"""
eval/retrieval_variants.py — 검색 설정 변형 비교 (평가셋 294문항, LLM 없음) — 정확도 실험 §5.56

recall@5 가 0.97 로 포화돼 작은 차이가 안 보이므로 더 촘촘한 지표를 함께 본다:
  recall@1·@3·@5, MRR, 다중 홉 질문에서 기대 논문을 '모두' 근거에 넣은 비율, 평균 근거 수.
질의 변환은 끄고(원 질문 1개) 자동 수집 논문은 숨긴다 — 검색 설정 차이만 본다.

  python eval/retrieval_variants.py --tag base
  python eval/retrieval_variants.py --tag qwen06 --set rerank_model=tomaarsen/Qwen3-Reranker-0.6B-seq-cls
  python eval/retrieval_variants.py --tag t512 --set pgvector_table=rag_chunks_t512
결과는 eval/results/retrieval_variants.json 에 태그별로 쌓인다.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import settings  # noqa: E402

OUT = ROOT / "eval" / "results" / "retrieval_variants.json"


def metrics(rows: list[dict]) -> dict:
    def rec(r, k):
        exp = set(r["exp"])
        return len(exp & set(r["srcs"][:k])) / len(exp)
    scored = [r for r in rows if r["exp"]]
    mh = [r for r in scored if r["multi_hop"]]
    n = len(scored)
    out = {f"recall@{k}": round(sum(rec(r, k) for r in scored) / n, 4) for k in (1, 3, 5)}
    out["MRR"] = round(sum(next((1 / (i + 1) for i, s in enumerate(r["srcs"]) if s in r["exp"]), 0) for r in scored) / n, 4)
    out["multi_hop_all_found"] = round(sum(set(r["exp"]) <= set(r["srcs"]) for r in mh) / len(mh), 4) if mh else None
    out["avg_evidence"] = round(sum(len(r["srcs"]) for r in scored) / n, 2)
    out["n"], out["n_multi_hop"] = n, len(mh)
    return out


def main() -> None:
    from eval.answer_quality import apply_sets

    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--set", action="append")
    ap.add_argument("--n", type=int)
    a = ap.parse_args()
    settings.query_transform_mode = "none"
    settings.auto_fetch = False
    settings.exclude_auto_papers = True
    changed = apply_sets(a.set)

    from eval.dataset import load_eval_set
    from rag.stages.stage2_rerank import retrieve_and_rerank
    from rag.store.vectorstore import load_vectorstore

    vs = load_vectorstore()
    rows, t0 = [], time.perf_counter()
    for r in load_eval_set(limit=a.n):
        _, rr = retrieve_and_rerank(vs, r["question"], [r["question"]])
        rows.append({"id": r["id"], "exp": r.get("expected_sources") or [], "multi_hop": r.get("multi_hop", False),
                     "srcs": [c.source for c in rr], "scores": [round(float(c.rerank_score or 0), 4) for c in rr]})
    m = metrics(rows)
    m["sec_per_q"] = round((time.perf_counter() - t0) / len(rows), 3)
    m["settings"] = {k: str(v) for k, v in changed.items()}
    res = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    res[a.tag] = m
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    (ROOT / "eval" / "results" / f"retrieval_variants_{a.tag}_rows.json").write_text(
        json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    print(a.tag, json.dumps(m, ensure_ascii=False))


if __name__ == "__main__":
    main()
