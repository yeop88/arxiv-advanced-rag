"""
rag/stage2_rerank.py — 2단계: 검색 + 리랭킹

파이프라인:
  (a) 1단계가 만든 여러 질의로 벡터검색 → 후보를 넉넉히(top_k_retrieve) 모음
  (b) 중복 제거(여러 질의가 같은 청크를 가져올 수 있음)
  (c) 크로스인코더 리랭커로 (질문, 청크) 관련도를 다시 채점 → 상위 top_k_rerank 만 선택

왜 리랭킹인가?
  벡터검색(bi-encoder)은 빠르지만 질문-문서를 '따로' 임베딩해 정밀도가 낮습니다.
  크로스인코더는 질문과 문서를 '함께' 보고 점수를 매겨 훨씬 정확합니다.
  → 빠른 검색으로 넓게 뽑고, 느리지만 정확한 리랭커로 좁히는 2단 구조가 정석.
"""
from __future__ import annotations

from functools import lru_cache

from langchain_core.documents import Document

from config import settings
from rag.llm import get_reranker
from rag.schemas import RetrievedChunk
from rag.vectorstore import VectorStoreLike, get_all_documents, get_vectors


def _chunk_key(content: str) -> str:
    """청크 중복 판별 키 (내용 앞부분)."""
    return content[:200]


def retrieve(vectorstore: VectorStoreLike, queries: list[str]) -> list[RetrievedChunk]:
    """여러 질의로 벡터검색 후 중복 제거한 후보 청크 리스트 반환."""
    # 질의가 여러 개이므로, 질의당 가져올 개수를 나눠 전체가 top_k_retrieve 근처가 되게
    per_query = max(3, settings.top_k_retrieve // max(1, len(queries)))

    seen: dict[str, RetrievedChunk] = {}
    for q in queries:
        # score_threshold 없이 유사도 점수와 함께 검색
        results = vectorstore.similarity_search_with_relevance_scores(q, k=per_query)
        for doc, score in results:
            key = _chunk_key(doc.page_content)  # 내용 앞부분으로 중복 판별
            if key in seen:
                # 더 높은 점수로 갱신
                if score > seen[key].score:
                    seen[key].score = score
                continue
            seen[key] = RetrievedChunk(
                content=doc.page_content,
                id=doc.id,
                source=doc.metadata.get("source", ""),
                title=doc.metadata.get("title", ""),
                section=doc.metadata.get("section", ""),
                url=doc.metadata.get("url", ""),
                score=float(score),
            )
    return list(seen.values())


# ---------------------------------------------------------------------------
# 하이브리드 검색: BM25(키워드) + 벡터(의미) 를 RRF 로 융합
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _get_bm25(vectorstore_id: int):
    """BM25 리트리버를 1회만 구축해 캐싱.

    BM25 는 전체 코퍼스가 메모리에 있어야 하므로 스토어에서 모든 청크를 꺼내 인덱싱
    (Pinecone 은 로컬 docstore 에서 꺼낸다 — Chroma 때와 같은 원문).
    (vectorstore_id 는 lru_cache 키 용도 — 같은 세션에선 재사용)
    """
    from langchain_community.retrievers import BM25Retriever

    docs = get_all_documents(_BM25_STORE["vs"])
    retriever = BM25Retriever.from_documents(docs)
    retriever.k = settings.top_k_retrieve
    return retriever


_BM25_STORE: dict = {}  # vectorstore 참조 보관 (lru_cache 는 해시 가능 인자만 받으므로)


def warm_bm25(vectorstore: VectorStoreLike) -> None:
    """BM25 색인을 미리 구축(챗봇 시작 시 백그라운드 로딩용). 첫 질문 지연을 없앤다."""
    _BM25_STORE["vs"] = vectorstore
    _get_bm25(id(vectorstore))


def reset_bm25() -> None:
    """코퍼스에 문서가 추가·삭제된 뒤 호출 → 다음 검색 때 새 원문으로 BM25 를 다시 만든다."""
    _get_bm25.cache_clear()


def _reciprocal_rank_fusion(
    rankings: list[list[Document]], k: int
) -> list[tuple[Document, float]]:
    """여러 랭킹 리스트를 RRF 로 융합.

    RRF 점수 = Σ 1/(k + rank).  rank 는 0부터. 점수가 아니라 '순위'만 쓰므로
    BM25 점수와 코사인 유사도처럼 척도가 다른 결과도 공정하게 합칠 수 있음.
    """
    scores: dict[str, float] = {}
    docs_by_key: dict[str, Document] = {}
    for ranking in rankings:
        for rank, doc in enumerate(ranking):
            key = _chunk_key(doc.page_content)
            if key not in docs_by_key or (doc.id and not docs_by_key[key].id):
                docs_by_key[key] = doc
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
    fused = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return [(docs_by_key[key], score) for key, score in fused]


def retrieve_hybrid(vectorstore: VectorStoreLike, queries: list[str]) -> list[RetrievedChunk]:
    """BM25 + 벡터 결과를 RRF 로 융합한 후보 리스트 반환."""
    _BM25_STORE["vs"] = vectorstore
    bm25 = _get_bm25(id(vectorstore))

    rankings: list[list[Document]] = []
    per_query = max(3, settings.top_k_retrieve // max(1, len(queries)))
    for q in queries:
        # 벡터 랭킹
        vec_docs = vectorstore.similarity_search(q, k=per_query)
        rankings.append(vec_docs)
        # BM25 랭킹
        bm25_docs = bm25.invoke(q)[:per_query]
        rankings.append(bm25_docs)

    fused = _reciprocal_rank_fusion(rankings, settings.rrf_k)[: settings.top_k_retrieve]
    return [
        RetrievedChunk(
            content=doc.page_content,
            id=doc.id,
            source=doc.metadata.get("source", ""),
            title=doc.metadata.get("title", ""),
            section=doc.metadata.get("section", ""),
            url=doc.metadata.get("url", ""),
            score=float(rrf_score),  # 여기선 RRF 점수를 score 로 보관
        )
        for doc, rrf_score in fused
    ]


def _mmr_select(
    candidates: list[RetrievedChunk],
    k: int,
    lambda_: float,
    same_source_penalty: float,
    max_per_source: int = 0,
    vectors: dict[str, list[float]] | None = None,
) -> list[RetrievedChunk]:
    """MMR(Maximal Marginal Relevance)로 관련도와 다양성을 함께 고려해 k개 선택.

    각 단계에서 다음을 최대화하는 후보를 고른다:
        MMR(i) = λ·rel(i) − (1−λ)·max_{j∈선택됨} sim(i, j)
    - rel : 크로스인코더 점수를 [0,1]로 정규화한 관련도
    - sim : 청크 임베딩 코사인 유사도. **같은 논문(source)끼리는** 유사도 하한을
            same_source_penalty 로 올려(=중복으로 간주) 한 논문의 독점을 억제한다.

    max_per_source>0 이면, 한 논문이 그 개수만큼 선택되면 더는 못 뽑게 **보장**한다.
    (MMR의 soft 벌점만으로는 관련도 격차가 큰 경우 쏠림을 못 막기 때문. 단, 후보 논문이
     부족해 k를 못 채우면 상한을 완화해 빈 슬롯을 채운다.)
    → 다중 홉 질문에서 서로 다른 논문의 근거가 상위에 고르게 올라온다.

    vectors: {청크 ID: 저장된 임베딩}. 주어지면 그대로 쓰고, 없는 청크만 새로 임베딩한다.
    (저장된 벡터는 색인 때 같은 모델로 만든 것이라 결과가 같고, CPU 재임베딩 수 초를 아낀다)
    """
    import numpy as np

    from rag.llm import get_embeddings

    if len(candidates) <= 1:
        return candidates[:k]

    # 관련도 정규화 (min-max). 전부 같으면 1.0
    scores = np.array([c.rerank_score or 0.0 for c in candidates], dtype=float)
    span = scores.max() - scores.min()
    rel = (scores - scores.min()) / span if span > 1e-9 else np.ones_like(scores)

    # 청크 임베딩 → 정규화(코사인 = 내적)
    vectors = vectors or {}
    missing = [i for i, c in enumerate(candidates) if not (c.id and c.id in vectors)]
    fresh = get_embeddings().embed_documents([candidates[i].content for i in missing]) if missing else []
    fresh_by_idx = dict(zip(missing, fresh))
    embs = np.array([fresh_by_idx[i] if i in fresh_by_idx else vectors[c.id]
                     for i, c in enumerate(candidates)], dtype=float)
    embs /= (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-9)
    sim = embs @ embs.T  # (N, N) 코사인 유사도

    # 같은 논문 청크는 유사도 하한을 올려 '중복'으로 취급
    sources = [c.source for c in candidates]
    for i in range(len(candidates)):
        for j in range(len(candidates)):
            if i != j and sources[i] and sources[i] == sources[j]:
                sim[i, j] = max(sim[i, j], same_source_penalty)

    k = min(k, len(candidates))
    selected: list[int] = []
    src_count: dict[str, int] = {}

    def pick_from(pool: set[int]) -> int | None:
        best_i, best_score = None, -1e9
        for i in pool:
            redundancy = max((sim[i, j] for j in selected), default=0.0)
            mmr = lambda_ * rel[i] - (1.0 - lambda_) * redundancy
            if mmr > best_score:
                best_score, best_i = mmr, i
        return best_i

    remaining = set(range(len(candidates)))
    while len(selected) < k and remaining:
        # 상한을 넘지 않은 논문의 후보만 우선 고려
        allowed = {i for i in remaining
                   if not max_per_source or src_count.get(sources[i], 0) < max_per_source}
        pool = allowed or remaining  # 모두 상한에 걸리면 완화해 슬롯을 채움
        i = pick_from(pool)
        if i is None:
            break
        selected.append(i)
        remaining.discard(i)
        src_count[sources[i]] = src_count.get(sources[i], 0) + 1

    return [candidates[i] for i in selected]


def rerank(question: str, candidates: list[RetrievedChunk],
           vectorstore: VectorStoreLike | None = None) -> list[RetrievedChunk]:
    """크로스인코더로 재정렬 후 상위 top_k_rerank 반환.

    config.rerank_diversity 가 True 면 단순 점수순 대신 MMR로 다양성까지 고려한다.
    """
    if not candidates:
        return []

    reranker = get_reranker()
    pairs = [(question, c.content) for c in candidates]
    scores = reranker.predict(pairs)  # 각 쌍의 관련도 점수

    for c, s in zip(candidates, scores):
        c.rerank_score = float(s)

    if settings.rerank_diversity:
        # 관련도순 정렬된 후보에 MMR 적용 (내부에서 다양성 재배치)
        candidates = sorted(candidates, key=lambda c: c.rerank_score or 0.0, reverse=True)
        vectors = get_vectors(vectorstore, [c.id for c in candidates if c.id]) if vectorstore else None
        return _mmr_select(
            candidates, settings.top_k_rerank, settings.mmr_lambda,
            settings.same_source_penalty, settings.rerank_max_per_source, vectors=vectors,
        )

    ranked = sorted(candidates, key=lambda c: c.rerank_score or 0.0, reverse=True)
    return ranked[: settings.top_k_rerank]


def retrieve_and_rerank(
    vectorstore: VectorStoreLike, question: str, queries: list[str]
) -> tuple[list[RetrievedChunk], list[RetrievedChunk]]:
    """검색 + 리랭킹을 한 번에. (검색결과, 리랭킹결과) 튜플 반환 — 시각화용.

    config.use_hybrid 에 따라 순수 벡터검색 또는 하이브리드(BM25+벡터) 사용.
    어느 쪽이든 결과를 크로스인코더로 재정렬하는 흐름은 동일.
    """
    if settings.use_hybrid:
        retrieved = retrieve_hybrid(vectorstore, queries)
    else:
        retrieved = retrieve(vectorstore, queries)
    reranked = rerank(question, retrieved, vectorstore)
    return retrieved, reranked
