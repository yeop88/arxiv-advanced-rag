"""
scripts/watch_papers.py — 관심 분야 알림(새 논문 소식지)

    python scripts/watch_papers.py --add "검색 증강 생성" [--query "retrieval augmented generation"]
    python scripts/watch_papers.py --list
    python scripts/watch_papers.py --remove "검색 증강 생성"
    python scripts/watch_papers.py --run [--days 7]      # 새 논문 확인 → storage/digests/YYYY-MM-DD.md

LLM: 경량 모델(주제 등록 시 검색어 1회, 논문당 요약 1회). 검색은 arXiv(주제당 1회, 요청 사이 3초).
주기적으로 받으려면 Windows 작업 스케줄러에 --run 을 등록하세요(README 참고).
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import settings  # noqa: E402
from rag import watch  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--add", metavar="주제")
    ap.add_argument("--query", help="영어 arXiv 검색어(생략하면 경량 LLM 이 만든다)")
    ap.add_argument("--remove", metavar="주제")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--days", type=int, default=7, help="처음 확인하는 주제는 최근 며칠 치를 볼지")
    args = ap.parse_args()
    if args.add:
        watch.add_topic(args.add, args.query)
    if args.remove:
        watch.remove_topic(args.remove)
    if args.run:
        from rag.vectorstore import load_vectorstore

        settings.exclude_auto_papers = False
        md, path = watch.run(load_vectorstore(), lookback_days=args.days,
                             progress=lambda ev, **kw: print(f"  [{kw['i']}/{kw['n']}] {kw['topic']}", flush=True))
        print(md)
        if path:
            print(f"\n저장 → {path}")
    if args.list or not (args.add or args.remove or args.run):
        for t in watch.load()["topics"]:
            print(f"- {t['topic']}  (검색어: {t['query']}, 마지막 확인: {t.get('last_checked') or '아직 없음'})")


if __name__ == "__main__":
    main()
