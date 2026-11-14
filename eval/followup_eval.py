"""
eval/followup_eval.py — 보충 검색(rag/stages/stage2_rerank.py, §5.45) 효과 측정 (LLM 없음)

여러 대상을 이름으로 묻는 질문에서 근거에 없는 대상을 따로 다시 찾는 보충 검색을 끄고/켜고(1회·2회)
평가셋 294문항(기준 코퍼스, 질의 변환 없이)의 recall@5·MRR·다중 홉 recall, 보충 검색이 일어난 문항 수를 비교한다.

사용: python eval/followup_eval.py
결과: eval/results/followup_eval.json
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import settings  # noqa: E402

settings.auto_fetch = False
settings.query_transform_mode = "none"
settings.use_hybrid = True

from eval.corpus import set_corpus  # noqa: E402
from eval.dataset import load_eval_set  # noqa: E402
from eval.retrieval_metrics import recall_at_k, reciprocal_rank  # noqa: E402
from rag.stages import stage2_rerank as s2  # noqa: E402
from rag.store.vectorstore import load_vectorstore  # noqa: E402

OUT = Path(__file__).resolve().parent / "results" / "followup_eval.json"


def run(vs, rows, on: bool, rounds: int) -> dict:
    settings.followup_retrieval, settings.followup_rounds = on, rounds
    per, triggered = {}, 0
    for r in rows:
        q = r["question"]
        _, rr = s2.retrieve_and_rerank(vs, q, [q])
        exp = r.get("expected_sources") or []
        per[r["id"]] = (recall_at_k(rr, exp, settings.top_k_rerank), reciprocal_rank(rr, exp))
    return per


def main() -> None:
    rows = [r for r in load_eval_set() if r.get("expected_sources")]
    multi = {r["id"] for r in rows if r.get("multi_hop")}
    set_corpus("base")
    vs = load_vectorstore()
    # 보충 검색이 일어날 문항(기준 결과에서 빠진 이름이 있는 것)
    settings.followup_retrieval = False
    trig = []
    for r in rows:
        _, rr = s2.retrieve_and_rerank(vs, r["question"], [r["question"]])
        if s2.missing_names(r["question"], rr):
            trig.append(r["id"])
    out = {"questions": len(rows), "multi_hop": len(multi), "would_trigger": len(trig), "variants": {}}
    base = None
    for name, on, rounds in [("끔", False, 0), ("1회", True, 1), ("2회", True, 2)]:
        t = time.perf_counter()
        per = run(vs, rows, on, rounds)
        n = len(per)
        row = {"recall": round(sum(v[0] for v in per.values()) / n, 4), "mrr": round(sum(v[1] for v in per.values()) / n, 4),
               "multi_hop_recall": round(sum(per[k][0] for k in multi) / len(multi), 4),
               "triggered_recall": round(sum(per[k][0] for k in trig) / len(trig), 4) if trig else None,
               "sec": round(time.perf_counter() - t)}
        if base is not None:
            row["better"] = [k for k in per if per[k] > base[k]]
            row["worse"] = [k for k in per if per[k] < base[k]]
        else:
            base = per
        out["variants"][name] = row
        print(name, {k: v for k, v in row.items() if k not in ("better", "worse")},
              "좋아짐", len(row.get("better", [])), "나빠짐", len(row.get("worse", [])), flush=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"보충 검색이 일어나는 문항 {len(trig)}개")


if __name__ == "__main__":
    main()
