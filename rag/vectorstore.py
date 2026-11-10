"""
rag/vectorstore.py — 벡터 스토어 로드/구축 헬퍼 (Chroma | Pinecone)

- load_vectorstore(): 파이프라인이 쓰는 스토어 로드 (config.vector_backend 로 선택)
- build_vectorstore(docs): 문서를 임베딩해 추가 (ingest 스크립트에서 사용)
- get_all_documents(vs): 전체 청크 → BM25 인덱스용
- delete_sources(sources) / reset_vectorstore(): 재수집·초기화용

설계: 파이프라인(stage2_rerank)은 아래 네 가지만 쓴다. 두 백엔드가 이 모양을 똑같이 제공하므로
`VECTOR_BACKEND` 만 바꾸면 나머지 코드는 그대로 동작한다.
    similarity_search(query, k) / similarity_search_with_relevance_scores(query, k)
    get() -> {"ids", "documents", "metadatas"} / delete(ids=[...])
"""
from __future__ import annotations

import shutil
import uuid
from pathlib import Path
from typing import Any, Protocol

from langchain_core.documents import Document

from config import settings
from rag.docstore import DocStore
from rag.llm import get_embeddings


class VectorStoreLike(Protocol):
    """파이프라인이 기대하는 벡터 스토어 인터페이스(Chroma 와 PineconeStore 공통)."""

    def similarity_search(self, query: str, k: int = 4) -> list[Document]: ...
    def similarity_search_with_relevance_scores(self, query: str, k: int = 4) -> list[tuple[Document, float]]: ...
    def get(self) -> dict: ...
    def delete(self, ids: list[str]) -> Any: ...


# ---------------------------------------------------------------------------
# Pinecone 어댑터
# ---------------------------------------------------------------------------
_PC_ALLOWED = (str, int, float, bool)


def _pc_metadata(meta: dict, text: str) -> dict:
    """Pinecone 메타데이터 제약에 맞춤: None 불가, 값은 문자열·숫자·불리언·문자열 리스트만."""
    out = {}
    for k, v in (meta or {}).items():
        if v is None:
            continue
        if isinstance(v, _PC_ALLOWED) or (isinstance(v, list) and all(isinstance(x, str) for x in v)):
            out[k] = v
        else:
            out[k] = str(v)
    out["text"] = text  # docstore 가 없어도 본문을 복구할 수 있게 함께 보관(청크 ≤ 1000자, 한도 40KB 이내)
    return out


class PineconeStore:
    """Pinecone 인덱스(벡터) + 로컬 DocStore(원문)를 묶어 Chroma 와 같은 모양으로 제공.

    검색 결과의 본문·메타데이터는 DocStore 에서 ID 로 찾아 붙인다 → Chroma 때와 내용이 동일.
    """

    UPSERT_BATCH = 100   # 1024차원 벡터 + 메타데이터 기준 요청당 2MB 한도 안쪽
    DELETE_BATCH = 1000  # Pinecone 삭제 요청당 최대 ID 수

    def __init__(self, index, namespace: str, embeddings, docstore: DocStore):
        self.index = index
        self.namespace = namespace
        self.embeddings = embeddings
        self.docstore = docstore

    # --- 검색 ---
    def similarity_search_with_relevance_scores(self, query: str, k: int = 4) -> list[tuple[Document, float]]:
        vec = self.embeddings.embed_query(query)
        res = self.index.query(vector=vec, top_k=k, include_metadata=True, namespace=self.namespace)
        out = []
        for m in res.matches:
            row = self.docstore.get(m.id)
            if row is not None:
                text, meta = row["text"], dict(row["metadata"])
            else:  # docstore 에 없으면 Pinecone 에 같이 넣어둔 본문으로 복구
                meta = dict(m.metadata or {})
                text = meta.pop("text", "")
            # cosine 점수(-1~1)를 0~1 관련도로. 순위는 그대로이며 점수는 표시·중복 처리에만 쓰인다.
            out.append((Document(page_content=text, metadata=meta, id=m.id), (m.score + 1) / 2))
        return out

    def similarity_search(self, query: str, k: int = 4) -> list[Document]:
        return [d for d, _ in self.similarity_search_with_relevance_scores(query, k)]

    # --- 조회·삭제 (Chroma 와 같은 반환 모양) ---
    def get(self) -> dict:
        rows = self.docstore.all()
        return {
            "ids": [r["id"] for r in rows],
            "documents": [r["text"] for r in rows],
            "metadatas": [r["metadata"] for r in rows],
        }

    def delete(self, ids: list[str]) -> None:
        ids = list(ids)
        for i in range(0, len(ids), self.DELETE_BATCH):
            self.index.delete(ids=ids[i:i + self.DELETE_BATCH], namespace=self.namespace)
        self.docstore.delete(ids)

    def delete_all(self) -> None:
        try:
            self.index.delete(delete_all=True, namespace=self.namespace)
        except Exception as e:  # noqa: BLE001 — 비어 있는 namespace 는 404 일 수 있음
            if "404" not in str(e) and "not found" not in str(e).lower():
                raise
        self.docstore.clear()

    def count(self) -> int:
        stats = self.index.describe_index_stats()
        ns = stats.namespaces.get(self.namespace) if stats.namespaces else None
        if ns is None:
            return 0
        return ns["vector_count"] if isinstance(ns, dict) else ns.vector_count

    # --- 추가 ---
    def upsert_vectors(self, ids: list[str], vectors: list[list[float]],
                       texts: list[str], metadatas: list[dict]) -> None:
        """이미 계산된 벡터를 그대로 넣는다(마이그레이션: Chroma 벡터를 복사 → 임베딩 동일성 보장)."""
        payload = [
            {"id": i, "values": list(map(float, v)), "metadata": _pc_metadata(m, t)}
            for i, v, t, m in zip(ids, vectors, texts, metadatas)
        ]
        for s in range(0, len(payload), self.UPSERT_BATCH):
            self.index.upsert(vectors=payload[s:s + self.UPSERT_BATCH],
                              namespace=self.namespace, show_progress=False)
        self.docstore.add(ids, texts, metadatas)

    def add_documents(self, docs: list[Document], ids: list[str] | None = None) -> list[str]:
        ids = ids or [str(uuid.uuid4()) for _ in docs]
        texts = [d.page_content for d in docs]
        vectors = self.embeddings.embed_documents(texts)
        self.upsert_vectors(ids, vectors, texts, [d.metadata for d in docs])
        return ids


def ensure_pinecone_index(dimension: int):
    """인덱스가 없으면 cosine 으로 생성, 있으면 차원·척도가 맞는지 확인. Index 객체 반환."""
    from pinecone import Pinecone, ServerlessSpec

    if not settings.pinecone_api_key:
        raise RuntimeError("PINECONE_API_KEY 가 설정되지 않았습니다.")
    pc = Pinecone(api_key=settings.pinecone_api_key)
    name = settings.pinecone_index
    if not pc.has_index(name):
        pc.create_index(
            name=name, dimension=dimension, metric="cosine",
            spec=ServerlessSpec(cloud=settings.pinecone_cloud, region=settings.pinecone_region),
        )
    desc = pc.describe_index(name)
    if desc.dimension != dimension:
        raise RuntimeError(f"Pinecone 인덱스 차원 {desc.dimension} ≠ 임베딩 차원 {dimension}")
    if str(desc.metric) not in ("cosine", "dotproduct"):
        # 정규화 벡터에서 cosine·dotproduct 는 Chroma(L2) 와 순위가 같지만 euclidean 은 점수 해석이 달라 막는다
        raise RuntimeError(f"Pinecone 인덱스 척도가 {desc.metric} 입니다. cosine 으로 만들어야 순위가 같습니다.")
    return pc.Index(name)


def _load_pinecone(create: bool = False) -> PineconeStore:
    from pinecone import Pinecone

    if not settings.pinecone_api_key:
        raise RuntimeError("PINECONE_API_KEY 가 설정되지 않았습니다.")
    emb = get_embeddings()
    if create:
        index = ensure_pinecone_index(len(emb.embed_query("dimension probe")))
    else:
        pc = Pinecone(api_key=settings.pinecone_api_key)
        if not pc.has_index(settings.pinecone_index):
            raise FileNotFoundError(
                f"Pinecone 인덱스가 없습니다: {settings.pinecone_index}\n"
                f"먼저 `python scripts/migrate_to_pinecone.py` 로 옮기거나 수집 스크립트를 실행하세요."
            )
        index = pc.Index(settings.pinecone_index)
    return PineconeStore(index, settings.pinecone_namespace, emb, DocStore(settings.docstore_path))


# ---------------------------------------------------------------------------
# Chroma
# ---------------------------------------------------------------------------
def _load_chroma():
    from langchain_chroma import Chroma

    if not Path(settings.chroma_dir).exists():
        raise FileNotFoundError(
            f"벡터 DB 가 없습니다: {settings.chroma_dir}\n"
            f"먼저 `python scripts/ingest_arxiv.py` 로 논문을 수집하세요."
        )
    return Chroma(
        collection_name=settings.collection_name,
        embedding_function=get_embeddings(),
        persist_directory=settings.chroma_dir,
    )


# ---------------------------------------------------------------------------
# 공통 진입점
# ---------------------------------------------------------------------------
def load_vectorstore(backend: str | None = None) -> VectorStoreLike:
    """설정된 백엔드의 스토어를 로드. backend 를 주면 설정과 무관하게 그 백엔드(비교 측정용)."""
    backend = backend or settings.vector_backend
    if backend == "pinecone":
        return _load_pinecone()
    if backend == "chroma":
        return _load_chroma()
    raise ValueError(f"알 수 없는 VECTOR_BACKEND: {backend} (chroma|pinecone)")


def get_all_documents(vs: VectorStoreLike) -> list[Document]:
    """스토어의 모든 청크를 Document 로 반환 (BM25 인덱스 구축용)."""
    raw = vs.get()  # {'documents': [...], 'metadatas': [...], ...}
    docs = []
    for content, meta in zip(raw.get("documents", []), raw.get("metadatas", [])):
        docs.append(Document(page_content=content, metadata=meta or {}))
    return docs


def delete_sources(sources: set[str]) -> int:
    """주어진 source(논문 ID)들의 기존 청크를 모두 삭제. 삭제한 청크 수 반환.

    같은 논문을 초록→전문(full-text)으로 다시 넣을 때, 옛 청크를 지우고
    새로 넣기 위해 사용한다. 스토어가 없으면 0.
    """
    if not sources:
        return 0
    try:
        vs = load_vectorstore()
    except FileNotFoundError:
        return 0
    raw = vs.get()
    ids = raw.get("ids", [])
    metas = raw.get("metadatas", [])
    to_del = [i for i, m in zip(ids, metas) if (m or {}).get("source") in sources]
    if to_del:
        vs.delete(ids=to_del)
    return len(to_del)


def build_vectorstore(docs: list[Document]) -> VectorStoreLike:
    """문서 리스트를 임베딩해 스토어에 추가(없으면 생성)."""
    if settings.vector_backend == "pinecone":
        store = _load_pinecone(create=True)
        store.add_documents(docs)
        return store
    from langchain_chroma import Chroma

    return Chroma.from_documents(
        documents=docs,
        embedding=get_embeddings(),
        collection_name=settings.collection_name,
        persist_directory=settings.chroma_dir,
    )


def reset_vectorstore() -> None:
    """현재 백엔드의 저장 내용을 모두 지운다(ingest_local --reset 용)."""
    if settings.vector_backend == "pinecone":
        try:
            _load_pinecone().delete_all()
        except FileNotFoundError:
            pass
        return
    if Path(settings.chroma_dir).exists():
        shutil.rmtree(settings.chroma_dir)
