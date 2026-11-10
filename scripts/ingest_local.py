"""
scripts/ingest_local.py — 로컬 PDF/텍스트 폴더 수집 (한국어 논문 등)

왜 이 방식인가?
  RISS·DBpia 등 국내 논문 DB 는 기관 로그인이 필요해 자동 수집이 어렵고,
  무단 크롤링은 이용약관 위반 소지가 있습니다. 가장 현실적·합법적인 경로는
  "본인 권한으로 내려받은 PDF 를 폴더에 넣어 로컬 수집"하는 것입니다.
  임베딩(bge-m3)은 한국어를 지원하므로 한국어 논문이 그대로 검색됩니다.

지원 형식: .pdf, .txt, .md
섹션 인식 청킹(rag/chunking.py)을 적용해 각 청크에 section 메타를 붙입니다.

사용:
    python scripts/ingest_local.py --dir ./my_papers
    python scripts/ingest_local.py --dir ./my_papers --reset   # 기존 DB 비우고 새로
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.documents import Document  # noqa: E402

from config import settings  # noqa: E402
from rag.chunking import section_aware_split  # noqa: E402
from rag.vectorstore import build_vectorstore, reset_vectorstore  # noqa: E402

SUPPORTED = {".pdf", ".txt", ".md"}


def read_file(path: Path) -> str:
    """파일에서 텍스트 추출."""
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    # txt / md
    return path.read_text(encoding="utf-8", errors="ignore")


def load_dir(directory: Path) -> list[Document]:
    """폴더 내 지원 파일을 모두 읽어 섹션 청킹된 Document 리스트로."""
    files = [p for p in sorted(directory.rglob("*")) if p.suffix.lower() in SUPPORTED]
    if not files:
        print(f"  지원 파일(.pdf/.txt/.md)이 없습니다: {directory}")
        return []

    chunks: list[Document] = []
    for path in files:
        text = read_file(path).strip()
        if not text:
            print(f"  (건너뜀, 텍스트 없음) {path.name}")
            continue
        # 파일명을 제목/출처로. 첫 비어있지 않은 줄을 제목 후보로 쓸 수도 있음.
        title = path.stem
        meta = {"source": path.name, "title": title, "origin": "local"}
        paper_chunks = section_aware_split(text, meta)
        chunks.extend(paper_chunks)
        n_sec = len({c.metadata.get("section") for c in paper_chunks})
        print(f"  수집: {path.name} ({len(paper_chunks)}청크/{n_sec}섹션)")
    return chunks


def main() -> None:
    parser = argparse.ArgumentParser(description="로컬 PDF/텍스트 폴더 → 벡터DB")
    parser.add_argument("--dir", required=True, help="수집할 폴더 경로")
    parser.add_argument("--reset", action="store_true", help="기존 벡터DB 삭제 후 재구축")
    args = parser.parse_args()

    directory = Path(args.dir)
    if not directory.is_dir():
        print(f"폴더를 찾을 수 없습니다: {directory}")
        return

    if args.reset:
        reset_vectorstore()
        print(f"[reset] 기존 벡터DB 삭제 ({settings.vector_backend})")

    print(f"[1/2] 로컬 수집·청킹: {directory}")
    chunks = load_dir(directory)
    if not chunks:
        return

    print(f"[2/2] 임베딩 + 벡터DB 저장 ({settings.vector_backend})")
    build_vectorstore(chunks)
    print(f"완료! 총 {len(chunks)}개 청크 저장. `streamlit run app.py` 로 실행하세요.")


if __name__ == "__main__":
    main()
