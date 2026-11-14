"""
scripts/export_feedback.py — 실제 사용 질문 기록에서 평가셋 후보를 뽑는다

챗봇 서버가 남긴 storage/query_log.db(server/querylog.py)를 읽어
eval/eval_set.jsonl 과 같은 모양의 후보 파일을 만든다. 기본은 👎 받은 질문만.
expected_sources 는 답변에 쓰인 근거 논문으로 미리 채워 두니, 사람이 보고 고친 뒤 평가셋에 옮긴다.
출력은 storage/ 아래(공개 저장소에 올라가지 않음) — 실제 사용자 질문이기 때문.

사용:
  python scripts/export_feedback.py                 # 👎 질문 → storage/eval_candidates.jsonl
  python scripts/export_feedback.py --all           # 피드백 없는 질문까지
  python scripts/export_feedback.py --stats         # 개수만
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import settings  # noqa: E402
from server.querylog import QueryLog  # noqa: E402


def to_candidate(row: dict) -> dict:
    sources = list(dict.fromkeys(s["source"] for s in row.get("sources") or [] if s.get("source")))
    return {
        "id": f"use_{row['id'][:8]}",
        "question": row.get("standalone") or row["question"],
        "category": "usage",
        "difficulty": "",            # 사람이 채움
        "multi_hop": len(sources) > 1,
        "expected_sources": sources,  # 답변에 쓰인 근거 — 맞는지 사람이 확인
        # 아래는 검토용(평가셋에 옮길 땐 지워도 됨)
        "_rating": row.get("rating"), "_note": row.get("note"),
        "_answer": row.get("answer"), "_verification": row.get("verification"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="피드백 없는 질문까지(👍 제외)")
    ap.add_argument("--stats", action="store_true")
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--out", default="./storage/eval_candidates.jsonl")
    args = ap.parse_args()

    if not Path(settings.query_log_path).exists():
        sys.exit(f"기록이 없습니다: {settings.query_log_path} (챗봇 서버를 쓰면 생깁니다)")
    qlog = QueryLog(settings.query_log_path)
    print("기록:", qlog.stats())
    if args.stats:
        return
    rows = qlog.list(limit=args.limit) if args.all else qlog.list(rating=-1, limit=args.limit)
    rows = [r for r in rows if r.get("rating") != 1 and not r.get("error")]
    seen, out = set(), []
    for r in rows:
        c = to_candidate(r)
        if c["question"] in seen:
            continue
        seen.add(c["question"])
        out.append(c)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("".join(json.dumps(c, ensure_ascii=False) + "\n" for c in out), encoding="utf-8")
    print(f"후보 {len(out)}개 → {args.out}")


if __name__ == "__main__":
    main()
