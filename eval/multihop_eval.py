"""
eval/multihop_eval.py — 다중 홉 분해(MULTIHOP_DECOMPOSE) 검색 효과 (§5.48)

다중 홉 질문(평가셋 67 + 기준 세트 18)에서 기대 논문이 최종 근거(top_k_rerank)에 몇 편 들어가는지를
분해 끔/켬으로 비교한다. 질의 변환(multi_query)은 LLM 캐시에서 나오고, 켬일 때만 하위 질문 분해(가벼운 LLM 1회)가 더 든다.
정답 판정에 LLM 을 쓰지 않는다(기대 논문 포함 여부).

사용: python eval/multihop_eval.py
결과: eval/results/multihop_eval.json
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

OUT = Path(__file__).resolve().parent / "results" / "multihop_eval.json"


def questions() -> list[dict]:
    from eval.dataset import load_eval_set

    rows = [{"id": r["id"], "set": "eval", "question": r["question"], "expected": r.get("expected_sources") or []}
            for r in load_eval_set() if r.get("multi_hop")]
    hs = Path(settings.human_set_path)
    if hs.exists():
        seen = {r["id"] for r in rows}
        for x in hs.read_text(encoding="utf-8").splitlines():
            r = json.loads(x)
            if r["kind"] == "multi_hop" and r["id"] not in seen:
                rows.append({"id": r["id"], "set": "human", "question": r["question"], "expected": r["expected_sources"]})
    return rows


def run(vs, rows: list[dict], decompose: bool) -> dict:
    from rag.pipeline import sub_questions
    from rag.stages.stage1_query_transform import transform_query
    from rag.stages.stage2_rerank import retrieve_and_rerank

    settings.multihop_decompose = decompose
    per = {}
    t0 = time.perf_counter()
    for i, r in enumerate(rows, 1):
        q = r["question"]
        try:
            queries = transform_query(q, settings.query_transform_mode)
        except Exception:  # noqa: BLE001
            queries = [q]
        subs = sub_questions(q)
        if subs:
            queries = list(dict.fromkeys([*queries, *subs]))
        _, reranked = retrieve_and_rerank(vs, q, queries, sub_questions=subs)
        got = {c.source for c in reranked}
        exp = set(r["expected"])
        per[r["id"]] = {"hit": len(exp & got), "n": len(exp), "subs": len(subs), "set": r["set"]}
        if i % 10 == 0:
            print(f"  {'켬' if decompose else '끔'} {i}/{len(rows)}", flush=True)

    def agg(keys):
        ks = [k for k in per if per[k]["set"] in keys]
        hit = sum(per[k]["hit"] for k in ks)
        tot = sum(per[k]["n"] for k in ks)
        return {"questions": len(ks), "recall": round(hit / tot, 4) if tot else None,
                "all_found": sum(per[k]["hit"] == per[k]["n"] for k in ks),
                "decomposed": sum(per[k]["subs"] >= 2 for k in ks)}

    return {"eval": agg({"eval"}), "human": agg({"human"}), "all": agg({"eval", "human"}),
            "sec": round(time.perf_counter() - t0), "per": per}


def main() -> None:
    from rag.store.vectorstore import load_vectorstore

    if settings.llm_provider != "local":
        raise SystemExit("로컬 LLM(LLM_PROVIDER=local)에서만 실행합니다.")
    settings.local_thinking = settings.local_thinking_light = "off"
    vs = load_vectorstore()
    rows = questions()
    print(f"다중 홉 {len(rows)}문항", flush=True)
    off, on = run(vs, rows, False), run(vs, rows, True)
    changed = {k: [off["per"][k]["hit"], on["per"][k]["hit"]] for k in off["per"] if off["per"][k]["hit"] != on["per"][k]["hit"]}
    res = {"off": {k: v for k, v in off.items() if k != "per"}, "on": {k: v for k, v in on.items() if k != "per"},
           "changed": changed}
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
