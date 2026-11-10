"""
rag/docstore.py — 청크 원문을 로컬 JSONL 로 보관하는 문서 저장소

왜 필요한가?
  Pinecone 같은 원격 벡터 DB 는 "벡터로 찾기"는 빠르지만 "저장된 문서를 전부 꺼내기"가
  약하다. 그런데 하이브리드 검색의 BM25 는 전체 코퍼스 원문이 메모리에 있어야 한다.
  그래서 원문·메타데이터는 이 로컬 파일을 기준(source of truth)으로 두고,
  Pinecone 에는 벡터와 조회용 최소 정보만 둔다. 검색 결과의 본문도 여기서 ID 로 찾아 붙여
  Chroma 를 쓰던 때와 내용이 완전히 같게 만든다.

형식: 한 줄에 {"id": ..., "text": ..., "metadata": {...}}
"""
from __future__ import annotations

import json
import os
from pathlib import Path


class DocStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._rows: dict[str, dict] = {}
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    r = json.loads(line)
                    self._rows[r["id"]] = r

    def __len__(self) -> int:
        return len(self._rows)

    def get(self, doc_id: str) -> dict | None:
        return self._rows.get(doc_id)

    def all(self) -> list[dict]:
        return list(self._rows.values())

    def add(self, ids: list[str], texts: list[str], metadatas: list[dict]) -> None:
        for i, t, m in zip(ids, texts, metadatas):
            self._rows[i] = {"id": i, "text": t, "metadata": dict(m or {})}
        self.save()

    def delete(self, ids: list[str]) -> None:
        for i in ids:
            self._rows.pop(i, None)
        self.save()

    def clear(self) -> None:
        self._rows.clear()
        self.save()

    def save(self) -> None:
        """임시 파일에 쓴 뒤 교체 — 쓰는 도중 중단돼도 기존 파일이 깨지지 않게."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            for r in self._rows.values():
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        os.replace(tmp, self.path)
