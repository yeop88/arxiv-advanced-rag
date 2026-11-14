"""
eval/judge_eval.py — 답변 판정기 비교: 어느 자동 판정이 사람 쪽 판정('맞음' 여부)과 맞는가

기준 세트 답변(또는 eval/answer_quality.py 로 다시 쓴 답변)에 대해 판정기마다 '맞음/아님'을 내고
판정 파일(기본 storage/human_labels.json)의 overall == correct 와 일치율·Cohen's κ 를 잰다.

판정기
  llm_v1 : 지금 4단계 검증(grounded) — 기준 세트를 만들 때 저장한 값(LLM 호출 없음)
  rule   : 답변에 근거 부족 표현("찾을 수 없습니다" 등)이 없으면 맞음(LLM 없음)
  llm_v2 : 검증 v2 — 질문을 함께 주고 빠뜨린 부분까지 확인. grounded·completeness=full·missed 없음이면 맞음
           (문항당 LLM 1회, 결과는 storage/answer_runs/judge_TAG_llm_v2[_think].json 에 저장해 다시 쓰지 않음)

사용: python eval/judge_eval.py [--tag base] [--llm] [--think] [--annotator claude]
결과: eval/results/judge_eval.json (수치만)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import settings  # noqa: E402
from eval.answer_quality import RUNS, chunks_of, load_items, load_judge, load_run  # noqa: E402
from eval.human_agreement import kappa  # noqa: E402

OUT = Path(__file__).resolve().parent / "results" / "judge_eval.json"


def rule_ok(answer: str) -> bool:
    from rag.ingest.auto_fetch import answer_lacks_evidence
    return not answer_lacks_evidence(answer)


def llm_v2(tag: str, think: bool) -> dict[str, dict]:
    """검증 v2 결과(캐시). 없는 문항만 LLM 으로 채운다."""
    from rag.stages import stage4_verify as s4

    if settings.llm_provider != "local":
        raise SystemExit("로컬 LLM(LLM_PROVIDER=local)에서만 실행합니다.")
    settings.local_thinking = "on" if think else "off"
    path = RUNS / f"judge_{tag}_llm_v2{'_think' if think else ''}.json"
    cache = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    items, run = load_items(), load_run(tag)
    todo = [k for k in run if k not in cache]
    for i, k in enumerate(todo, 1):
        it, t = items[k], time.perf_counter()
        chunks = chunks_of({"sources": run[k].get("sources") or it["sources"]})
        v = s4.verify(run[k]["answer"], chunks, question=it["question"], version="v2")
        cache[k] = {"grounded": v.grounded, "completeness": v.completeness, "missed": v.missed,
                    "unsupported": v.unsupported_claims, "reasoning": v.reasoning}
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"[{i}/{len(todo)}] {k} {time.perf_counter() - t:.0f}s {v.grounded} {v.completeness} 빠뜨림 {len(v.missed)}",
              flush=True)
    return cache


def v2_ok(v: dict) -> bool:
    return bool(v["grounded"]) and v["completeness"] == "full" and not v["missed"]


def score(truth: dict[str, bool], pred: dict[str, bool]) -> dict:
    ks = [k for k in truth if k in pred]
    h, p = [truth[k] for k in ks], [pred[k] for k in ks]
    if not ks:
        return {}
    return {"n": len(ks), "agreement": round(sum(a == b for a, b in zip(h, p)) / len(ks), 3), "kappa": kappa(h, p),
            "pred_correct_rate": round(sum(p) / len(ks), 3), "truth_correct_rate": round(sum(h) / len(ks), 3),
            "missed_problems": sum((not a) and b for a, b in zip(h, p)),   # 문제 답을 맞다고 함
            "false_alarms": sum(a and (not b) for a, b in zip(h, p))}      # 맞는 답을 문제라고 함


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="base")
    ap.add_argument("--annotator", help="이 주체의 판정만 기준으로(기본: 판정 파일 전부)")
    ap.add_argument("--llm", action="store_true", help="검증 v2 를 돌린다(문항당 LLM 1회, 캐시)")
    ap.add_argument("--think", action="store_true", help="검증 v2 에 thinking 켜기")
    a = ap.parse_args()
    items, run, judge = load_items(), load_run(a.tag), load_judge(a.tag)
    truth = {k: v["overall"] == "correct" for k, v in judge.items()
             if k in run and v.get("overall") and (not a.annotator or v.get("annotator", "human") == a.annotator)}
    if not truth:
        raise SystemExit(f"{a.tag} 판정이 없습니다.")
    res = {"tag": a.tag, "annotators": sorted({judge[k].get("annotator", "human") for k in truth}),
           "rule": score(truth, {k: rule_ok(r["answer"]) for k, r in run.items()})}
    if a.tag == "base":
        res["llm_v1"] = score(truth, {k: bool(items[k]["llm_verification"]["grounded"]) for k in run})
    for think in ([False, True] if a.think else [False]):
        name = f"llm_v2{'_think' if think else ''}"
        path = RUNS / f"judge_{a.tag}_{name}.json"
        if a.llm and (think == a.think):
            cache = llm_v2(a.tag, think)
        else:
            cache = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        if cache:
            res[name] = score(truth, {k: v2_ok(v) for k, v in cache.items()})
            res[name + "_or_rule"] = score(truth, {k: v2_ok(v) and rule_ok(run[k]["answer"]) for k, v in cache.items()})
    allres = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    allres[a.tag] = res
    OUT.write_text(json.dumps(allres, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
