"""
tests/fake_pinecone.py — Pinecone Index 의 메모리 내 대역(테스트·오프라인 동일성 검증용)

PineconeStore 가 쓰는 메서드(upsert / query / delete / describe_index_stats)만 같은 모양으로
흉내 낸다. 검색은 정확한 코사인 전수 탐색이라, 실제 Pinecone(근사 탐색)보다 엄격한 기준이다.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class _Match:
    id: str
    score: float
    metadata: dict | None


@dataclass
class _QueryResponse:
    matches: list


@dataclass
class _Vec:
    values: list


@dataclass
class _FetchResponse:
    vectors: dict


@dataclass
class _Stats:
    namespaces: dict = field(default_factory=dict)


class FakePineconeIndex:
    def __init__(self):
        self._ns: dict[str, dict[str, tuple[np.ndarray, dict]]] = {}

    def upsert(self, *, vectors, namespace: str = "", **_):
        store = self._ns.setdefault(namespace, {})
        for v in vectors:
            vec = np.asarray(v["values"], dtype=float)
            store[v["id"]] = (vec / (np.linalg.norm(vec) + 1e-12), dict(v.get("metadata") or {}))

    def query(self, *, top_k: int, vector, namespace: str = "", include_metadata: bool = False, **_):
        store = self._ns.get(namespace, {})
        if not store:
            return _QueryResponse(matches=[])
        ids = list(store)
        mat = np.stack([store[i][0] for i in ids])
        q = np.asarray(vector, dtype=float)
        q = q / (np.linalg.norm(q) + 1e-12)
        scores = mat @ q
        order = np.argsort(-scores)[:top_k]
        return _QueryResponse(matches=[
            _Match(ids[j], float(scores[j]), store[ids[j]][1] if include_metadata else None) for j in order
        ])

    def fetch(self, *, ids, namespace: str = "", **_):
        store = self._ns.get(namespace, {})
        return _FetchResponse(vectors={i: _Vec(list(store[i][0])) for i in ids if i in store})

    def delete(self, *, ids=None, delete_all: bool = False, namespace: str = "", **_):
        if delete_all:
            self._ns.pop(namespace, None)
            return
        store = self._ns.get(namespace, {})
        for i in ids or []:
            store.pop(i, None)

    def describe_index_stats(self, **_):
        return _Stats(namespaces={ns: {"vector_count": len(s)} for ns, s in self._ns.items()})
