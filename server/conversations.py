"""
server/conversations.py — 대화를 서버(로컬 SQLite)에 저장

지금까지 대화는 브라우저(localStorage)에만 남아 다른 브라우저·기기에서 이어 볼 수 없었고, 지우면 사라졌다.
대화마다 id·제목·메시지 목록(화면의 ChatMessage 모양 그대로, JSON)을 저장한다. 사용자 구분은 아직 없다(한 사람용).
저장 위치는 storage/ 라 공개 저장소에 올라가지 않는다. CONVERSATIONS=0 이면 끈다(화면은 브라우저 저장만 씀).
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id        TEXT PRIMARY KEY,
    title     TEXT NOT NULL,
    created   REAL NOT NULL,
    updated   REAL NOT NULL,
    n         INTEGER NOT NULL,        -- 메시지 수
    messages  TEXT NOT NULL            -- JSON 배열
);
CREATE INDEX IF NOT EXISTS conversations_updated ON conversations(updated);
"""
MAX_MESSAGES = 400


def title_of(messages: list[dict]) -> str:
    first = next((m.get("content", "") for m in messages if m.get("role") == "user" and m.get("content")), "")
    first = " ".join(first.split())
    return (first[:40] + "…") if len(first) > 40 else (first or "새 대화")


class ConversationStore:
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

    def list(self, limit: int = 100) -> list[dict]:
        with self._connect() as db:
            rows = db.execute("SELECT id, title, created, updated, n FROM conversations ORDER BY updated DESC LIMIT ?",
                              (limit,)).fetchall()
        return [dict(r) for r in rows]

    def get(self, cid: str) -> dict | None:
        with self._connect() as db:
            r = db.execute("SELECT * FROM conversations WHERE id = ?", (cid,)).fetchone()
        if r is None:
            return None
        d = dict(r)
        d["messages"] = json.loads(d["messages"])
        return d

    def put(self, cid: str, messages: list[dict], title: str | None = None) -> dict:
        """대화 전체를 저장(있으면 덮어씀). 빈 대화는 지운다."""
        messages = messages[-MAX_MESSAGES:]
        if not messages:
            self.delete(cid)
            return {"id": cid, "deleted": True}
        now = time.time()
        t = (title or "").strip() or title_of(messages)
        with self._lock, self._connect() as db:
            db.execute(
                "INSERT INTO conversations (id, title, created, updated, n, messages) VALUES (?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(id) DO UPDATE SET title = excluded.title, updated = excluded.updated, n = excluded.n, "
                "messages = excluded.messages",
                (cid, t, now, now, len(messages), json.dumps(messages, ensure_ascii=False)))
        return {"id": cid, "title": t, "updated": now, "n": len(messages)}

    def delete(self, cid: str) -> bool:
        with self._lock, self._connect() as db:
            return db.execute("DELETE FROM conversations WHERE id = ?", (cid,)).rowcount > 0
