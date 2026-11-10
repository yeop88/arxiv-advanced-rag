"""
scripts/manage_auto_papers.py — 챗봇이 자동 수집한 논문(origin=auto) 보기·지우기

자동 수집은 근거가 부족할 때 arXiv 논문을 DB 에 추가한다. 관련 없는 논문이 들어갔거나
Pinecone 무료 플랜 한도(약 30만 레코드)에 가까워지면 이 스크립트로 정리한다.
수동 수집(ingest_*.py)한 논문은 건드리지 않는다.

사용:
    python scripts/manage_auto_papers.py                  # 목록
    python scripts/manage_auto_papers.py --remove 2404.19756 2205.14135
    python scripts/manage_auto_papers.py --remove-all     # 자동 수집 논문 전부 삭제
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import settings  # noqa: E402

settings.exclude_auto_papers = False  # 관리 대상이 자동 수집분이므로 숨기지 않는다
from rag.vectorstore import delete_sources, load_vectorstore  # noqa: E402


def auto_papers() -> dict[str, dict]:
    raw = load_vectorstore().get()
    papers: dict[str, dict] = defaultdict(lambda: {"chunks": 0})
    for m in raw.get("metadatas", []):
        m = m or {}
        if m.get("origin") != "auto":
            continue
        p = papers[m.get("source", "?")]
        p["chunks"] += 1
        p.update(title=m.get("title", ""), fetched_at=m.get("fetched_at", ""), query=m.get("fetch_query", ""))
    return dict(papers)


def main() -> None:
    ap = argparse.ArgumentParser(description="자동 수집 논문(origin=auto) 관리")
    ap.add_argument("--remove", nargs="+", metavar="ARXIV_ID", help="지울 논문 ID")
    ap.add_argument("--remove-all", action="store_true", help="자동 수집 논문 전부 삭제")
    args = ap.parse_args()

    papers = auto_papers()
    print(f"벡터 DB: {settings.vector_backend} | 자동 수집 논문 {len(papers)}편, "
          f"청크 {sum(p['chunks'] for p in papers.values())}개")
    for pid, p in sorted(papers.items(), key=lambda kv: kv[1]["fetched_at"]):
        print(f"  [{pid}] {p['title'][:70]} | {p['chunks']}청크 | {p['fetched_at']} | 질문: {p['query'][:40]}")

    targets = set(papers) if args.remove_all else set(args.remove or [])
    if not targets:
        return
    unknown = targets - set(papers)
    if unknown:
        print(f"⚠️ 자동 수집 목록에 없는 ID 는 건너뜀(수동 수집 논문 보호): {sorted(unknown)}")
    targets &= set(papers)
    removed = delete_sources(targets)
    print(f"삭제: 논문 {len(targets)}편, 청크 {removed}개. BM25 색인은 챗봇 재시작 시 다시 만들어집니다.")


if __name__ == "__main__":
    main()
