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

import re
from functools import lru_cache

from langchain_core.documents import Document

from config import settings
from rag.llm import get_reranker
from rag.schemas import VISUAL_KINDS, RetrievedChunk
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
                url=doc.metadata.get("url", ""), origin=doc.metadata.get("origin", ""),
                kind=doc.metadata.get("kind", ""),
                score=float(score),
            )
    return list(seen.values())


# ---------------------------------------------------------------------------
# 하이브리드 검색: BM25(키워드) + 벡터(의미) 를 RRF 로 융합
# ---------------------------------------------------------------------------
_LATIN = re.compile(r"[a-z0-9]+")
_HANGUL = re.compile(r"[가-힣]+")
# 한국어 질문에서 영어 용어 뒤에 붙는 조사·어미("HyDE의", "RETRO는")와 한글 단어 끝의 흔한 조사
_PARTICLES = ("에서는", "으로는", "에서", "으로", "에게", "까지", "부터", "보다", "처럼", "이란", "이나",
              "은", "는", "이", "가", "을", "를", "의", "에", "와", "과", "로", "도", "만", "란")


def _strip_particle(word: str) -> str:
    for p in _PARTICLES:
        if word.endswith(p) and len(word) > len(p) + 1:
            return word[: -len(p)]
    return word


@lru_cache(maxsize=1)
def _kiwi():
    from kiwipiepy import Kiwi

    return Kiwi()


def bm25_tokenize(text: str) -> list[str]:
    """BM25 단어 나누기(설정 BM25_TOKENIZER).

    whitespace: 공백만(이전 방식) — "HyDE의"·"RETRO," 처럼 조사·문장부호가 붙어 영어 용어가 안 맞았다.
    regex: 영문·숫자와 한글을 따로 떼고 한글 끝 조사를 뗀다.
    kiwi: 영문·숫자는 regex 와 같고, 한글은 형태소 분석으로 명사·동사 어간 등 실질 형태소만.
    """
    mode = settings.bm25_tokenizer
    if mode == "whitespace":
        return text.split()
    low = (text or "").lower()
    tokens = _LATIN.findall(low)
    if mode == "kiwi":
        for hangul in _HANGUL.findall(low):
            tokens += [t.form for t in _kiwi().tokenize(hangul) if t.tag[0] in "NVX" or t.tag in ("SL", "SN")]
    else:
        tokens += [_strip_particle(w) for w in _HANGUL.findall(low)]
    return [t for t in tokens if t not in _PARTICLES]  # 영어 용어에서 떨어져 나온 조사만 남은 토큰은 버린다


@lru_cache(maxsize=1)
def _get_bm25(vectorstore_id: int):
    """BM25 리트리버를 1회만 구축해 캐싱.

    BM25 는 전체 코퍼스가 메모리에 있어야 하므로 스토어에서 모든 청크를 꺼내 인덱싱
    (Pinecone 은 로컬 docstore 에서 꺼낸다 — Chroma 때와 같은 원문).
    (vectorstore_id 는 lru_cache 키 용도 — 같은 세션에선 재사용)
    """
    from langchain_community.retrievers import BM25Retriever

    docs = get_all_documents(_BM25_STORE["vs"])
    # 본문과 표·그림은 색인을 따로 만든다 — 표 청크가 본문 색인의 단어 통계(IDF·평균 길이)를 바꿔
    # 본문 순위가 흔들리지 않게(§5.37). 반환: (본문 리트리버, 표·그림 리트리버 또는 None)
    body = [d for d in docs if not _is_visual(d)]
    visual = [d for d in docs if _is_visual(d)]
    retrievers = []
    for part in (body, visual):
        r = BM25Retriever.from_documents(part, preprocess_func=bm25_tokenize) if part else None
        if r is not None:
            r.k = settings.top_k_retrieve
        retrievers.append(r)
    return tuple(retrievers)


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


def _is_visual(doc: Document) -> bool:
    return (doc.metadata or {}).get("kind") in VISUAL_KINDS


def _split_visual(docs: list[Document], n_body: int) -> tuple[list[Document], list[Document]]:
    """랭킹 앞에서부터 본문 청크 n_body 개와, 그 사이에 나온 표·그림 청크를 나눠 돌려준다."""
    body: list[Document] = []
    visual: list[Document] = []
    for d in docs:
        if len(body) >= n_body:
            break
        (visual if _is_visual(d) else body).append(d)
    return body, visual


def retrieve_hybrid(vectorstore: VectorStoreLike, queries: list[str]) -> list[RetrievedChunk]:
    """BM25 + 벡터 결과를 RRF 로 융합한 후보 리스트 반환.

    본문 청크와 표·그림 청크(rag/visual.py)는 따로 융합한다. 본문 후보는 표·그림이 없던 때와 똑같이
    top_k_retrieve 개를 유지하고, 표·그림은 visual_candidates 개까지 따로 더한다 — 표 청크가 후보 자리를
    차지해 리랭커가 높게 볼 본문 근거가 아예 후보에서 빠지는 일을 막는다(이슈 20, §5.37).
    """
    _BM25_STORE["vs"] = vectorstore
    bm25_body, bm25_visual = _get_bm25(id(vectorstore))

    body_rankings: list[list[Document]] = []
    visual_rankings: list[list[Document]] = []
    per_query = max(3, settings.top_k_retrieve // max(1, len(queries)))
    for q in queries:
        # 벡터 랭킹 — 표·그림을 걸러내도 본문이 per_query 개 남게 2배로 받아 나눈다
        body, visual = _split_visual(vectorstore.similarity_search(q, k=per_query * 2), per_query)
        body_rankings.append(body)
        visual_rankings.append(visual)
        # BM25 랭킹 — 본문·표 색인이 따로 있다
        body_rankings.append(bm25_body.invoke(q)[:per_query])
        if bm25_visual is not None:
            visual_rankings.append(bm25_visual.invoke(q)[:per_query])

    fused = _reciprocal_rank_fusion(body_rankings, settings.rrf_k)[: settings.top_k_retrieve]
    if settings.visual_candidates > 0:
        fused += _reciprocal_rank_fusion(visual_rankings, settings.rrf_k)[: settings.visual_candidates]
    return [
        RetrievedChunk(
            content=doc.page_content,
            id=doc.id,
            source=doc.metadata.get("source", ""),
            title=doc.metadata.get("title", ""),
            section=doc.metadata.get("section", ""),
            url=doc.metadata.get("url", ""), origin=doc.metadata.get("origin", ""),
            kind=doc.metadata.get("kind", ""),
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
    # 표·그림 청크는 같은 논문의 본문과 따로 센다 — 본문 2개가 이미 뽑힌 논문의 표도 들어올 수 있게(§5.37)
    sources = [c.source + ("#visual" if c.kind in VISUAL_KINDS else "") for c in candidates]
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
        # 자동 수집 논문은 가중치만큼 낮춰, 주변 논문이 무관한 질문의 상위를 차지하지 않게 한다(1.0 = 그대로)
        c.rerank_score = float(s) * (settings.auto_rerank_weight if c.origin == "auto" else 1.0)
    # 표·그림은 리랭커가 질문과 확실히 관련 있다고 본 것만 근거로 쓴다 — 빈 자리를 채우러 무관한 표가 들어오지 않게(§5.37)
    candidates = [c for c in candidates
                  if c.kind not in VISUAL_KINDS or (c.rerank_score or 0.0) >= settings.visual_min_score]

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
