"""
eval/nli_eval.py — 문장 단위 인용 확인에 함의(NLI) 판정을 더할지 측정 (LLM 없음)

리랭커(관련도)는 '근거와 같은 주제지만 다른 말'을 거의 그대로 통과시킨다. 다국어 NLI 모델
(MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7, 한국어 문장 ↔ 영어 근거)이 그런 문장을 잡는지 본다.

자료: 수집 신호 기록(eval/results/fetch_signal_record.jsonl)의 실제 답변 문장과 인용한 근거.
  원래 문장      : 대부분 맞게 쓴 문장 → 경보가 적어야 함(오경보)
  숫자 바꿈      : 문장의 숫자 하나 마지막 자리 변경(78.4 → 78.5)
  방향 뒤집기    : 높/낮, 증가/감소, 빠르/느리, 향상/저하, 많/적, 크/작 … 반대말로 바꿈
  다른 논문 근거 : 같은 질문의 근거 중 인용한 논문과 다른 논문 청크와 짝지음
각 경우의 '경보율'(리랭커: 점수 < 기준, NLI: 모순 확률 ≥ 기준 또는 함의 확률 < 기준)을 비교한다.
사람 판정 기준 세트가 쌓이면 eval/human_agreement.py --extra 로 다시 잰다.

사용: python eval/nli_eval.py            # 위 네 경우 비교
      python eval/nli_eval.py --human    # 사람 판정 기준 세트의 문장별 NLI 경보 → storage/nli_flags.json
                                         #   → python eval/human_agreement.py --extra storage/nli_flags.json
결과: eval/results/nli_eval.json
"""
from __future__ import annotations

import json
import random
import re
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import settings  # noqa: E402
from rag.stages import citation_check as cc  # noqa: E402

RECORD = Path(__file__).resolve().parent / "results" / "fetch_signal_record.jsonl"
OUT = Path(__file__).resolve().parent / "results" / "nli_eval.json"

# 방향 뒤집기: (원래, 반대) — 앞쪽이 먼저 맞는 것부터
FLIPS = [("높습니다", "낮습니다"), ("낮습니다", "높습니다"), ("높은", "낮은"), ("낮은", "높은"), ("증가", "감소"), ("감소", "증가"),
         ("빠르", "느리"), ("느리", "빠르"), ("향상", "저하"), ("개선", "악화"), ("많은", "적은"), ("적은", "많은"),
         ("큰 ", "작은 "), ("작은 ", "큰 "), ("우수", "열등"), ("효율적", "비효율적"), ("가능합니다", "불가능합니다"),
         ("줄", "늘"), ("능가", "못 미치")]


def flip(text: str) -> str | None:
    for a, b in FLIPS:
        if a in text:
            return text.replace(a, b, 1)
    return None


def perturb_number(text: str) -> str | None:
    nums = cc.answer_numbers(text)
    if not nums or nums[0] not in text:
        return None
    n = nums[0]
    return text.replace(n, n[:-1] + str((int(n[-1]) + 1) % 10), 1)


def human_flags(out_path: Path = Path("storage/nli_flags.json")) -> None:
    """기준 세트 문장마다 '인용한 근거 모두와 모순 확률 ≥ 기준'이면 경보(파이프라인의 contradicted 와 같은 규칙)."""
    from rag.stages.citation_check import nli_scores

    rows = [json.loads(x) for x in Path(settings.human_set_path).read_text(encoding="utf-8").splitlines() if x.strip()]
    flags = {}
    for r in rows:
        sents = r["citation_check"]["sentences"]
        jobs = [(i, n) for i, s in enumerate(sents) for n in s["cites"] if 1 <= n <= len(r["sources"])]
        _, con = nli_scores([(r["sources"][n - 1]["content"], sents[i]["text"]) for i, n in jobs])
        worst: dict[int, float] = {}
        for (i, _), c in zip(jobs, con):
            worst[i] = min(worst.get(i, 1.0), c)
        flags[r["id"]] = [worst.get(i, 0.0) >= settings.citation_nli_contradiction for i in range(len(sents))]
    out_path.write_text(json.dumps(flags, ensure_ascii=False), encoding="utf-8")
    print(f"{len(flags)}문항 → {out_path} (경보 문장 {sum(sum(v) for v in flags.values())}개)")


def main() -> None:
    if "--human" in sys.argv:
        human_flags()
        return
    from rag.core.llm import get_reranker
    from rag.stages.citation_check import nli_scores

    rows = [json.loads(x) for x in RECORD.read_text(encoding="utf-8").splitlines() if x.strip()]
    rng = random.Random(0)
    cases: dict[str, list[tuple[str, str]]] = {"원래 문장": [], "숫자 바꿈": [], "방향 뒤집기": [], "다른 논문 근거": []}
    for r in rows:
        ev = r.get("evidence") or []
        for s in cc.split_sentences(r.get("answer") or ""):
            if not cc._is_claim(s):
                continue
            cites = [n for n in cc.parse_cites(s) if 1 <= n <= len(ev)]
            if not cites:
                continue
            claim = cc.claim_text(s)
            prem = ev[cites[0] - 1]["content"]
            cases["원래 문장"].append((prem, claim))
            if (x := perturb_number(claim)):
                cases["숫자 바꿈"].append((prem, x))
            if (x := flip(claim)):
                cases["방향 뒤집기"].append((prem, x))
            others = [e for e in ev if e["source"] not in {ev[n - 1]["source"] for n in cites}]
            if others:
                cases["다른 논문 근거"].append((rng.choice(others)["content"], claim))

    rr = get_reranker()
    thr = settings.citation_support_threshold
    out: dict = {"threshold_rerank": thr, "cases": {}}
    grid_c, grid_e = [0.5, 0.7, 0.9], [0.1, 0.3, 0.5]
    for name, pairs in cases.items():
        rer = [float(x) for x in rr.predict([(h, p) for p, h in pairs], batch_size=32)]
        ent, con = nli_scores(pairs)
        row = {"n": len(pairs),
               "rerank_flag": round(sum(x < thr for x in rer) / len(pairs), 3),
               "nli_contradiction": {f">={c}": round(sum(x >= c for x in con) / len(pairs), 3) for c in grid_c},
               "nli_not_entailed": {f"<{e}": round(sum(x < e for x in ent) / len(pairs), 3) for e in grid_e},
               "mean_entail": round(sum(ent) / len(ent), 3), "mean_contra": round(sum(con) / len(con), 3)}
        out["cases"][name] = row
        print(name, row, flush=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
