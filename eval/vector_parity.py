"""
eval/vector_parity.py — Chroma 와 Pinecone 의 검색 결과가 같은지 확인 (전환 전 필수 점검)

"성능 저하 없이" 를 숫자로 보이기 위해 평가셋 전체 질문에 대해 두 백엔드를 비교한다.
LLM 은 쓰지 않는다(질의변환 none) → 비용 0, 순수하게 벡터 스토어 차이만 본다.

비교 항목:
  - 벡터 검색 상위 N 겹침률(dense overlap@N): 두 백엔드가 같은 청크를 찾는가
  - 최종 리랭크 결과 일치율: 하이브리드+리랭킹+MMR 을 거친 최종 근거 5개가 완전히 같은가
  - recall@k · MRR: 정답 논문을 얼마나 잘 찾는가 (Pinecone 이 Chroma 보다 낮으면 '저하')

사용:
    python eval/vector_parity.py --fake      # 키 없이: 메모리 내 Pinecone 대역에 같은 벡터를 넣어 어댑터 검증
    python eval/vector_parity.py             # 실제 Pinecone (migrate_to_pinecone.py 로 이전한 뒤)
    python eval/vector_parity.py --n 50      # 질문 수 제한
"""
from __future__ import annotations

import argparse
import sys
import tempfile
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))

from config import settings  # noqa: E402

settings.query_transform_mode = "none"  # LLM 미사용: 벡터 스토어 차이만 측정
settings.auto_fetch = False  # 평가는 코퍼스를 고정한다(근거 부족 시 논문 자동 수집 끔)

from eval.dataset import load_eval_set  # noqa: E402
from eval.retrieval_metrics import recall_at_k, reciprocal_rank  # noqa: E402
from rag.stage2_rerank import _chunk_key, retrieve_and_rerank  # noqa: E402
from rag.vectorstore import load_vectorstore  # noqa: E402


def _fake_pinecone_store():
    """실제 계정 없이: Chroma 벡터를 그대로 복사한 메모리 내 Pinecone 대역 스토어."""
    from fake_pinecone import FakePineconeIndex
    from migrate_to_pinecone import copy_into, export_chroma

    from rag.docstore import DocStore
    from rag.llm import get_embeddings
    from rag.vectorstore import PineconeStore

    tmp = Path(tempfile.mkdtemp()) / "docstore.jsonl"
    store = PineconeStore(FakePineconeIndex(), "parity", get_embeddings(), DocStore(tmp))
    print("  (fake) Chroma 벡터를 메모리 내 Pinecone 대역에 복사 중...")
    copy_into(store, export_chroma(), batch=1000)
    return store


def measure(vs, records, n_dense: int) -> dict:
    k = settings.top_k_rerank
    out = {}
    for r in records:
        q = r["question"]
        t0 = time.perf_counter()
        dense = [_chunk_key(d.page_content) for d in vs.similarity_search(q, k=n_dense)]
        dense_ms = (time.perf_counter() - t0) * 1000
        _, rr = retrieve_and_rerank(vs, q, [q])
        exp = r.get("expected_sources") or []
        out[r["id"]] = {
            "dense": dense, "final": [_chunk_key(c.content) for c in rr], "dense_ms": dense_ms,
            "recall": recall_at_k(rr, exp, k) if exp else None,
            "mrr": reciprocal_rank(rr, exp) if exp else None,
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fake", action="store_true", help="실제 Pinecone 대신 메모리 내 대역 사용")
    ap.add_argument("--n", type=int, default=None, help="질문 수 제한")
    ap.add_argument("--dense-k", type=int, default=settings.top_k_retrieve)
    args = ap.parse_args()

    records = load_eval_set(limit=args.n)
    print(f"질문 {len(records)}개 | hybrid={settings.use_hybrid} diversity={settings.rerank_diversity} "
          f"dense top-{args.dense_k} / final top-{settings.top_k_rerank}")

    print("[1/2] Chroma 측정")
    a = measure(load_vectorstore("chroma"), records, args.dense_k)
    print("[2/2] Pinecone 측정" + (" (fake)" if args.fake else ""))
    pc = _fake_pinecone_store() if args.fake else load_vectorstore("pinecone")
    b = measure(pc, records, args.dense_k)

    def avg(xs):
        xs = [x for x in xs if x is not None]
        return sum(xs) / len(xs) if xs else 0.0

    ids = list(a)
    overlap = avg([len(set(a[i]["dense"]) & set(b[i]["dense"])) / max(1, len(a[i]["dense"])) for i in ids])
    dense_same = avg([float(a[i]["dense"] == b[i]["dense"]) for i in ids])
    final_same = avg([float(a[i]["final"] == b[i]["final"]) for i in ids])
    ra, rb = avg([a[i]["recall"] for i in ids]), avg([b[i]["recall"] for i in ids])
    ma, mb = avg([a[i]["mrr"] for i in ids]), avg([b[i]["mrr"] for i in ids])
    k = settings.top_k_rerank

    print("\n| 항목 | Chroma | Pinecone" + (" (fake)" if args.fake else "") + " | 차이 |")
    print("|---|---|---|---|")
    print(f"| recall@{k} | {ra:.4f} | {rb:.4f} | {rb - ra:+.4f} |")
    print(f"| MRR | {ma:.4f} | {mb:.4f} | {mb - ma:+.4f} |")
    print(f"| 벡터 검색 1회 평균(ms) | {avg([a[i]['dense_ms'] for i in ids]):.1f} | "
          f"{avg([b[i]['dense_ms'] for i in ids]):.1f} | |")
    print(f"\n벡터 검색 상위 {args.dense_k} 겹침률: {overlap:.2%} | 순서까지 동일: {dense_same:.2%}")
    print(f"최종 근거 {k}개 완전 일치: {final_same:.2%}")
    diff = [i for i in ids if a[i]["final"] != b[i]["final"]]
    if diff:
        print(f"최종 근거가 다른 질문 {len(diff)}개: {diff[:10]}")

    tol = 0.005
    ok = rb >= ra - tol and mb >= ma - tol
    if ok and rb >= ra and mb >= ma:
        verdict = "✅ 저하 없음 (recall·MRR 이 Chroma 이상)"
    elif ok:
        verdict = f"✅ 허용 범위 (차이 recall {rb - ra:+.4f} / MRR {mb - ma:+.4f}, 허용 오차 {tol} 이내 — 근사 탐색 수준)"
    else:
        verdict = f"⚠️ 저하 있음 (허용 오차 {tol} 초과) — 원인 확인 필요"
    print("\n판정:", verdict)
    if args.fake:
        print("※ fake 는 정확 탐색이라 '어댑터·이전 과정이 결과를 바꾸지 않는지'를 검증한다. "
              "실제 Pinecone 은 근사 탐색이므로 키를 넣고 다시 실행해 최종 확인할 것.")


if __name__ == "__main__":
    main()
