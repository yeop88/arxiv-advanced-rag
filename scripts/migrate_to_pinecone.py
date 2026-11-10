"""
scripts/migrate_to_pinecone.py — Chroma → Pinecone 이전 (임베딩 재계산 없이 벡터를 그대로 복사)

성능 저하 없이 옮기기 위한 원칙:
  1) 벡터를 다시 만들지 않고 Chroma 에 저장된 벡터를 그대로 복사 → 임베딩이 완전히 동일.
  2) ID 를 그대로 유지 → 재실행해도 덮어쓰기(upsert)라 중복이 생기지 않음.
  3) 인덱스 척도는 cosine. 벡터가 길이 1로 정규화돼 있어 Chroma 의 L2 와 순위가 같다.
  4) 원문·메타데이터는 로컬 docstore(JSONL)에도 저장 → BM25(하이브리드 검색)가 Chroma 때와 같은 원문을 쓴다.

사용:
    python scripts/migrate_to_pinecone.py --dry-run   # 키 없이 점검만(데이터·제약 확인)
    python scripts/migrate_to_pinecone.py             # 실제 이전 (PINECONE_API_KEY 필요)
    python scripts/migrate_to_pinecone.py --reset     # 대상 namespace 를 비우고 다시 이전
이전 후:
    python eval/vector_parity.py                      # Chroma 와 검색 결과가 같은지 확인
    .env 에 VECTOR_BACKEND=pinecone                   # 전환
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from config import settings  # noqa: E402
from rag.docstore import DocStore  # noqa: E402
from rag.llm import get_embeddings  # noqa: E402
from rag.vectorstore import PineconeStore, _load_chroma, _pc_metadata, ensure_pinecone_index  # noqa: E402

PAGE = 500


def export_chroma() -> dict:
    """Chroma 의 모든 청크를 벡터·원문·메타데이터와 함께 꺼낸다."""
    col = _load_chroma()._collection
    total = col.count()
    out = {"ids": [], "embeddings": [], "documents": [], "metadatas": []}
    for off in range(0, total, PAGE):
        page = col.get(limit=PAGE, offset=off, include=["embeddings", "documents", "metadatas"])
        out["ids"] += page["ids"]
        out["embeddings"] += [list(map(float, e)) for e in page["embeddings"]]
        out["documents"] += page["documents"]
        out["metadatas"] += [m or {} for m in page["metadatas"]]
    return out


def check(data: dict) -> dict:
    """이전 전 점검: 개수·차원·정규화·ID 중복·메타데이터 크기(Pinecone 40KB/벡터)."""
    emb = np.asarray(data["embeddings"])
    norms = np.linalg.norm(emb, axis=1)
    meta_bytes = [len(json.dumps(_pc_metadata(m, t), ensure_ascii=False).encode("utf-8"))
                  for m, t in zip(data["metadatas"], data["documents"])]
    return {
        "count": len(data["ids"]),
        "unique_ids": len(set(data["ids"])),
        "dimension": int(emb.shape[1]) if emb.size else 0,
        "norm_min": float(norms.min()) if emb.size else 0.0,
        "norm_max": float(norms.max()) if emb.size else 0.0,
        "query_embedding_dim": len(get_embeddings().embed_query("dimension probe")),
        "meta_bytes_max": max(meta_bytes) if meta_bytes else 0,
        "sources": len({m.get("source") for m in data["metadatas"]}),
    }


def copy_into(store: PineconeStore, data: dict, batch: int = 500) -> None:
    """준비된 데이터를 스토어에 넣는다(벡터 그대로). 진행 상황 출력."""
    n = len(data["ids"])
    for s in range(0, n, batch):
        e = min(s + batch, n)
        store.upsert_vectors(data["ids"][s:e], data["embeddings"][s:e],
                             data["documents"][s:e], data["metadatas"][s:e])
        print(f"  업로드 {e}/{n}")


def wait_until_count(store: PineconeStore, expected: int, timeout: int = 180) -> int:
    """Pinecone 은 반영이 약간 늦을 수 있어(eventual consistency) 개수가 맞을 때까지 기다린다."""
    t0 = time.time()
    while True:
        c = store.count()
        if c >= expected or time.time() - t0 > timeout:
            return c
        time.sleep(3)


def main() -> None:
    ap = argparse.ArgumentParser(description="Chroma → Pinecone 이전")
    ap.add_argument("--dry-run", action="store_true", help="Pinecone 에 쓰지 않고 점검만")
    ap.add_argument("--reset", action="store_true", help="대상 namespace·docstore 를 비우고 이전")
    args = ap.parse_args()

    print(f"[1/3] Chroma 내보내기: {settings.chroma_dir}")
    data = export_chroma()
    info = check(data)
    print("      " + json.dumps(info, ensure_ascii=False))

    problems = []
    if info["unique_ids"] != info["count"]:
        problems.append("ID 중복")
    if info["dimension"] != info["query_embedding_dim"]:
        problems.append(f"저장 벡터 차원 {info['dimension']} ≠ 현재 임베딩 차원 {info['query_embedding_dim']}")
    if not (0.99 < info["norm_min"] and info["norm_max"] < 1.01):
        problems.append("벡터가 정규화돼 있지 않음 → cosine 과 L2 순위가 달라질 수 있음")
    if info["meta_bytes_max"] > 40_000:
        problems.append(f"메타데이터 최대 {info['meta_bytes_max']}B > Pinecone 한도 40KB")
    if problems:
        print("⚠️ 점검 실패: " + "; ".join(problems))
        sys.exit(1)
    print("      점검 통과: ID 고유·차원 일치·정규화·메타데이터 크기 OK")

    if args.dry_run:
        print("(dry-run: Pinecone 에 쓰지 않았습니다)")
        return

    print(f"[2/3] Pinecone 인덱스 준비: {settings.pinecone_index} / namespace={settings.pinecone_namespace}")
    index = ensure_pinecone_index(info["dimension"])
    store = PineconeStore(index, settings.pinecone_namespace, get_embeddings(), DocStore(settings.docstore_path))
    if args.reset:
        store.delete_all()
        print("      기존 namespace·docstore 비움")

    print(f"[3/3] 벡터 복사 ({info['count']}개)")
    copy_into(store, data)
    got = wait_until_count(store, info["count"])
    print(f"완료: Pinecone {got}개 / Chroma {info['count']}개, docstore {len(store.docstore)}개")
    if got != info["count"]:
        print("⚠️ 개수 불일치 — 잠시 뒤 다시 확인하거나 --reset 으로 재실행하세요.")
    print("다음: python eval/vector_parity.py 로 검색 동일성 확인 → .env 에 VECTOR_BACKEND=pinecone")


if __name__ == "__main__":
    main()
