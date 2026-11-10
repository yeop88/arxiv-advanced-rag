"""
rag/vectorstore.py — Chroma 벡터 스토어 로드/구축 헬퍼

- build_vectorstore(docs): 문서로 새 컬렉션 생성 (ingest 스크립트에서 사용)
- load_vectorstore(): 기존 컬렉션 로드 (파이프라인에서 사용)
"""
from __future__ import annotations

from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document

from config import settings
from rag.llm import get_embeddings


def load_vectorstore() -> Chroma:
    """디스크에 저장된 Chroma 컬렉션을 로드."""
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


def get_all_documents(vs: Chroma) -> list[Document]:
    """Chroma 에 저장된 모든 청크를 Document 로 반환 (BM25 인덱스 구축용)."""
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


def build_vectorstore(docs: list[Document]) -> Chroma:
    """문서 리스트로부터 Chroma 컬렉션을 생성/추가."""
    return Chroma.from_documents(
        documents=docs,
        embedding=get_embeddings(),
        collection_name=settings.collection_name,
        persist_directory=settings.chroma_dir,
    )
