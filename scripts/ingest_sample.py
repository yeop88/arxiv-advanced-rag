"""
scripts/ingest_sample.py — 번들 샘플 논문으로 벡터DB 구축 (네트워크 불필요)

arXiv API 접속이 막힌 환경(사내 프록시 등)이나, 재현 가능한 데모를 원할 때 사용.
data/sample_papers.json 의 RAG 관련 대표 논문 초록을 임베딩해 Chroma 에 저장합니다.

사용:
    python scripts/ingest_sample.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.documents import Document  # noqa: E402
from langchain_text_splitters import RecursiveCharacterTextSplitter  # noqa: E402

from config import settings  # noqa: E402
from rag.vectorstore import build_vectorstore  # noqa: E402

SAMPLE_PATH = Path(__file__).resolve().parent.parent / "data" / "sample_papers.json"


def _existing_sources() -> set[str]:
    """이미 벡터DB에 저장된 source 집합. 재실행 시 같은 논문 중복 적재를 막는다."""
    try:
        from rag.vectorstore import load_vectorstore

        raw = load_vectorstore().get()
        return {
            (m or {}).get("source")
            for m in raw.get("metadatas", [])
            if (m or {}).get("source")
        }
    except Exception:  # noqa: BLE001 — DB 미존재 등: 중복제거 없이 진행
        return set()


def load_sample_docs() -> list[Document]:
    papers = json.loads(SAMPLE_PATH.read_text(encoding="utf-8"))
    seen = _existing_sources()
    skipped = 0
    docs = []
    for p in papers:
        if p["source"] in seen:
            skipped += 1
            continue
        seen.add(p["source"])
        text = f"{p['title']}\n\n{p['abstract']}"
        docs.append(
            Document(
                page_content=text,
                metadata={
                    "source": p["source"],
                    "title": p["title"],
                    "authors": p.get("authors", ""),
                    "published": p.get("published", ""),
                },
            )
        )
    if skipped:
        print(f"      (이미 저장된 {skipped}편은 건너뜀 - 중복 방지)")
    return docs


def main() -> None:
    print(f"[1/3] 샘플 논문 로드: {SAMPLE_PATH.name}")
    docs = load_sample_docs()
    print(f"      {len(docs)}편 로드")

    print(f"[2/3] 청킹 (chunk_size={settings.chunk_size})")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    chunks = splitter.split_documents(docs)
    print(f"      {len(docs)}편 → {len(chunks)}개 청크")

    print(f"[3/3] 임베딩 + Chroma 저장 → {settings.chroma_dir}")
    build_vectorstore(chunks)
    print(f"완료! 총 {len(chunks)}개 청크 저장.")


if __name__ == "__main__":
    main()
