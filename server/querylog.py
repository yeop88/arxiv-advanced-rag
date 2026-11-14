"""
server/querylog.py — 실제 사용 질문 기록(로컬 SQLite)

질문마다 질문·답변·근거·검증 결과·설정을 한 줄로 남기고, 화면의 👍/👎 피드백을 붙인다.
👎 받은 질문은 평가셋 후보가 된다(scripts/export_feedback.py). 저장 위치는 storage/ 라
공개 저장소에 올라가지 않는다. 기록 실패는 답변을 막지 않는다(경고만 출력).
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS queries (
    id            TEXT PRIMARY KEY,
    ts            REAL NOT NULL,          -- 기록 시각(유닉스 초)
    question      TEXT NOT NULL,
    standalone    TEXT,                   -- 멀티턴 문맥화 뒤 질문
    history_turns INTEGER DEFAULT 0,
    answer        TEXT,
    sources       TEXT,                   -- 근거 목록 JSON(source·title·section·kind·rerank_score)
    verification  TEXT,                   -- 4단계 검증 JSON
    citation_check TEXT,                  -- 문장 단위 인용 확인 JSON
    auto_fetched  TEXT,                   -- 자동 수집 논문 JSON
    timings       TEXT,                   -- 단계별 시간 JSON
    options       TEXT,                   -- 이 질문에 쓴 설정 JSON
    error         TEXT,                   -- 실패했으면 메시지
    rating        INTEGER,                -- 1(👍) / -1(👎) / NULL
    note          TEXT,                   -- 👎 이유(선택)
    rated_ts      REAL
);
CREATE INDEX IF NOT EXISTS queries_rating ON queries(rating);
"""

_JSON_FIELDS = ("sources", "verification", "citation_check", "auto_fetched", "timings", "options")


class QueryLog:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        with self._connect() as db:
            db.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        return db

    def record(self, question: str, *, result: dict | None = None, options: dict | None = None,
               history_turns: int = 0, error: str | None = None) -> str:
        """질문 하나를 기록하고 id 를 돌려준다. result 는 server.schemas.result_json 모양."""
        rid = uuid.uuid4().hex[:16]
        r = result or {}
        sources = [{k: s.get(k) for k in ("source", "title", "section", "kind", "rerank_score")}
                   for s in r.get("sources", [])]
        row = {
            "id": rid, "ts": time.time(), "question": question,
            "standalone": r.get("standalone_question"), "history_turns": history_turns,
            "answer": r.get("answer"), "sources": sources, "verification": r.get("verification"),
            "citation_check": r.get("citation_check"), "auto_fetched": r.get("auto_fetched"),
            "timings": r.get("timings"), "options": options, "error": error,
        }
        for k in _JSON_FIELDS:
            row[k] = json.dumps(row[k], ensure_ascii=False) if row[k] is not None else None
        cols = ", ".join(row)
        with self._lock, self._connect() as db:
            db.execute(f"INSERT INTO queries ({cols}) VALUES ({', '.join('?' * len(row))})", tuple(row.values()))
        return rid

    def rate(self, rid: str, rating: int, note: str | None = None) -> bool:
        """피드백을 붙인다(0 이면 취소). 없는 id 면 False."""
        with self._lock, self._connect() as db:
            cur = db.execute("UPDATE queries SET rating = ?, note = ?, rated_ts = ? WHERE id = ?",
                             (rating or None, (note or "").strip() or None, time.time(), rid))
            return cur.rowcount > 0

    def list(self, rating: int | None = None, limit: int = 100) -> list[dict]:
        """최근 기록부터. rating 을 주면 그 피드백만."""
        sql, args = "SELECT * FROM queries", []
        if rating is not None:
            sql += " WHERE rating = ?"
            args.append(rating)
        sql += " ORDER BY ts DESC LIMIT ?"
        args.append(limit)
        with self._connect() as db:
            rows = [dict(r) for r in db.execute(sql, args)]
        for r in rows:
            for k in _JSON_FIELDS:
                if r.get(k):
                    r[k] = json.loads(r[k])
        return rows

    def stats(self) -> dict:
        with self._connect() as db:
            n, up, down, err = db.execute(
                "SELECT COUNT(*), SUM(rating = 1), SUM(rating = -1), SUM(error IS NOT NULL) FROM queries").fetchone()
        return {"total": n, "up": up or 0, "down": down or 0, "errors": err or 0}


STAGES = [("0_contextualize", "문맥화"), ("1_query_transform", "질의 변환"), ("2_retrieve_rerank", "검색·리랭크"),
          ("2b_auto_fetch", "자동 수집"), ("3_generate", "생성"), ("4_verify", "검증"), ("4b_citation_check", "인용 확인")]


def metrics(rows: list[dict], days: int = 14) -> dict:
    """질문 기록 → 화면 지표(LLM 없음). rows 는 QueryLog.list() 결과(최근 순)."""
    import datetime as dt

    ok = [r for r in rows if not r.get("error")]
    n = len(rows)

    def mean(xs):
        xs = [x for x in xs if x is not None]
        return round(sum(xs) / len(xs), 3) if xs else None

    total_sec = [sum((r.get("timings") or {}).values()) for r in ok if r.get("timings")]
    cc = [r["citation_check"]["summary"] for r in ok if r.get("citation_check")]
    stages = [{"key": k, "label": lab, "mean_sec": mean([(r.get("timings") or {}).get(k) for r in ok]),
               "runs": sum(k in (r.get("timings") or {}) for r in ok)} for k, lab in STAGES]
    today = dt.date.today()
    by_day = {(today - dt.timedelta(days=i)).isoformat(): 0 for i in range(days)}
    for r in rows:
        d = dt.date.fromtimestamp(r["ts"]).isoformat()
        if d in by_day:
            by_day[d] += 1
    v = [r["verification"] for r in ok if r.get("verification")]
    return {
        "total": n, "errors": n - len(ok),
        "error_rate": round((n - len(ok)) / n, 3) if n else None,
        "up": sum(r.get("rating") == 1 for r in rows), "down": sum(r.get("rating") == -1 for r in rows),
        "mean_total_sec": mean(total_sec),
        "verify_grounded_rate": round(sum(bool(x.get("grounded")) for x in v) / len(v), 3) if v else None,
        "citation_precision": mean([s.get("precision") for s in cc]),
        "number_mismatch_rate": round(sum(s.get("number_mismatch", 0) > 0 for s in cc) / len(cc), 3) if cc else None,
        "auto_fetch_rate": round(sum(bool(r.get("auto_fetched")) for r in ok) / len(ok), 3) if ok else None,
        "stages": stages,
        "by_day": [{"date": d, "n": c} for d, c in sorted(by_day.items())],
        "recent": [{"id": r["id"], "ts": r["ts"], "question": r["question"][:120], "rating": r.get("rating"),
                    "error": r.get("error"), "sec": round(sum((r.get("timings") or {}).values()), 1),
                    "citation": (r.get("citation_check") or {}).get("summary"),
                    "grounded": (r.get("verification") or {}).get("grounded")} for r in rows[:20]],
    }
