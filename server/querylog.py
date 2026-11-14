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
