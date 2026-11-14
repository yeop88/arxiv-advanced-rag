"""
eval/citation_check_eval.py — 문장 단위 인용 확인(rag/stages/citation_check.py)의 기준 점수 보정 (LLM 없음)

수집 신호 평가 기록(eval/results/fetch_signal_record.jsonl — 실제 첫 답변 81개와 그때의 근거 원문)을 다시 쓴다.
사람이 붙인 정답 판정은 없으므로 대리 지표로 본다(숫자 대조는 숫자 하나를 바꾼 문장을 잡는지도 본다):
  양성  : 문장 ↔ 그 문장이 인용한 근거(대부분 맞게 달린 인용)
  어려운 음성 : 문장 ↔ 같은 질문 근거 중 '인용한 논문과 다른 논문'의 청크(그럴듯하지만 다른 근거)
  쉬운 음성   : 문장 ↔ 다른 질문의 근거 청크
리랭커 점수가 양성과 음성을 얼마나 가르는지(AUC)와 기준값별 통과율을 계산하고, 기준값에서 실제 답변의 문장 상태 분포를 낸다.

사용: python eval/citation_check_eval.py [--threshold 0.1]
결과: eval/results/citation_check_eval.json (문장 원문 없이 수치만)
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.core.schemas import RetrievedChunk  # noqa: E402
from rag.stages import citation_check as cc  # noqa: E402

RECORD = Path(__file__).resolve().parent / "results" / "fetch_signal_record.jsonl"
OUT = Path(__file__).resolve().parent / "results" / "citation_check_eval.json"
VISUAL_QA = Path(__file__).resolve().parent / "results" / "visual_qa_refactor.jsonl"
GRID = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]


def auc(pos: list[float], neg: list[float]) -> float:
    """양성 점수가 음성보다 클 확률(동점 0.5)."""
    if not pos or not neg:
        return float("nan")
    allv = sorted([(v, 1) for v in pos] + [(v, 0) for v in neg])
    rank_sum, i = 0.0, 0
    while i < len(allv):   # 동점은 평균 순위
        j = i
        while j < len(allv) and allv[j][0] == allv[i][0]:
            j += 1
        avg = (i + j + 1) / 2
        rank_sum += avg * sum(lbl for _, lbl in allv[i:j])
        i = j
    return (rank_sum - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def pct(xs: list[float], q: float) -> float:
    xs = sorted(xs)
    return round(xs[min(len(xs) - 1, int(q * len(xs)))], 4) if xs else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=None, help="상태 분포를 낼 기준값(기본 config)")
    ap.add_argument("--visual-qa", default=str(VISUAL_QA), help="숫자 대조를 볼 표·수식 답변 기록(빈 값이면 생략)")
    args = ap.parse_args()
    rows = [json.loads(line) for line in RECORD.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = [r for r in rows if r.get("answer") and r.get("evidence")]
    rng = random.Random(0)
    from rag.core.llm import get_reranker
    reranker = get_reranker()

    pos, hard, easy = [], [], []
    num_alarms: list = []    # 원래 답변에서 숫자가 인용 근거에 없다고 나온 문장
    perturbed: list = []     # 숫자 하나를 바꾼 문장을 잡았는지
    statuses: Counter = Counter()
    per_answer = []
    for qi, r in enumerate(rows):
        chunks = [RetrievedChunk(content=e["content"], source=e.get("source", ""), title=e.get("title", ""))
                  for e in r["evidence"]]
        others = [e["content"] for j, o in enumerate(rows) if j != qi for e in o["evidence"]]
        sents = [s for s in cc.split_sentences(r["answer"]) if cc._is_claim(s)]
        pairs, meta = [], []
        for si, s in enumerate(sents):
            claim = cc.claim_text(s)
            for ci, c in enumerate(chunks):
                pairs.append((claim, c.content))
                meta.append(("chunk", si, ci))
            pairs.append((claim, rng.choice(others)))
            meta.append(("easy", si, -1))
        if not pairs:
            continue
        scores = [float(x) for x in reranker.predict(pairs, batch_size=32)]
        table: dict[int, dict[int, float]] = {}
        for (kind, si, ci), v in zip(meta, scores):
            if kind == "easy":
                easy.append(v)
            else:
                table.setdefault(si, {})[ci] = v
        for si, s in enumerate(sents):
            cites = [n for n in cc.parse_cites(s) if 1 <= n <= len(chunks)]
            if not cites:
                continue
            pos.append(max(table[si][n - 1] for n in cites))
            cited_src = {chunks[n - 1].source for n in cites}
            neg = [v for ci, v in table[si].items() if chunks[ci].source not in cited_src]
            if neg:
                hard.append(max(neg))   # 다른 논문 중 가장 그럴듯한 근거
        rep = cc.check(r["answer"], chunks, scorer=lambda p: _rescore(p, table, sents),
                       threshold=args.threshold, question=r.get("standalone") or r["question"])
        statuses.update(s.status for s in rep.sentences)
        per_answer.append(rep.summary())
        # --- 숫자 대조: 원래 답변의 경보(오경보 후보)와, 숫자 하나를 바꾼 문장을 잡는지 ---
        chunk_nums = [cc.chunk_numbers(c.content) for c in chunks]
        for si, s in enumerate(sents):
            sc = rep.sentences[si]
            if sc.numbers_missing:
                num_alarms.append((sc.text, sc.numbers_missing, sc.numbers_found_in))
            cites = [n for n in cc.parse_cites(s) if 1 <= n <= len(chunks)]
            nums = cc.answer_numbers(cc.claim_text(s))
            if not cites or not nums or sc.numbers_missing:
                continue
            have = set().union(*(chunk_nums[n - 1] for n in cites))
            fake = _perturb(nums[0], have)
            bad = s.replace(nums[0], fake, 1) if nums[0] in s else None
            if not bad:
                continue
            missing, _, _ = cc.check_numbers(bad, cites, chunk_nums)
            perturbed.append({"caught": fake in missing, "pair": (cc.claim_text(s), cc.claim_text(bad),
                                                                  chunks[cites[0] - 1].content)})

    grid = [{"threshold": t,
             "pos_pass": round(sum(v >= t for v in pos) / len(pos), 3),
             "hard_neg_pass": round(sum(v >= t for v in hard) / len(hard), 3),
             "easy_neg_pass": round(sum(v >= t for v in easy) / len(easy), 3)} for t in GRID]
    cited_answers = [a for a in per_answer if a["precision"] is not None]
    # 숫자를 바꾼 문장에 리랭커는 어떻게 반응하나(숫자 대조가 따로 필요한 이유)
    if perturbed:
        ps = [float(x) for x in reranker.predict(
            [(a, c) for p in perturbed for a, c in ((p["pair"][0], p["pair"][2]), (p["pair"][1], p["pair"][2]))],
            batch_size=32)]
        orig, fake = ps[0::2], ps[1::2]
        thr = args.threshold if args.threshold is not None else cc.settings.citation_support_threshold
        rerank_on_fake = {"mean_score_original": round(sum(orig) / len(orig), 3),
                          "mean_score_changed_number": round(sum(fake) / len(fake), 3),
                          "changed_still_passes": round(sum(f >= thr for f in fake) / len(fake), 3)}
    else:
        rerank_on_fake = None
    print("\n[숫자 경보 — 원래 답변]")
    for text, miss, where in num_alarms:
        print(f" - {miss} (다른 근거 {where}) | {text[:160]}")
    out = {
        "answers": len(per_answer), "cited_sentences": len(pos),
        "auc_pos_vs_hard": round(auc(pos, hard), 4), "auc_pos_vs_easy": round(auc(pos, easy), 4),
        "percentiles": {name: {f"p{int(q * 100)}": pct(xs, q) for q in (0.05, 0.1, 0.25, 0.5, 0.75, 0.9)}
                        for name, xs in (("pos", pos), ("hard_neg", hard), ("easy_neg", easy))},
        "grid": grid,
        "threshold_used": args.threshold if args.threshold is not None else cc.settings.citation_support_threshold,
        "status_counts": dict(statuses),
        "mean_precision": round(sum(a["precision"] for a in cited_answers) / len(cited_answers), 3),
        "answers_all_supported": sum(a["supported"] == a["claims"] for a in per_answer),
        "numbers": {
            "numeric_sentences": sum(a["numeric"] for a in per_answer),
            "alarms_on_original": len(num_alarms),
            "alarms_with_number_in_other_chunk": sum(bool(w) for _, _, w in num_alarms),
            "changed_number_tested": len(perturbed),
            "changed_number_caught": round(sum(p["caught"] for p in perturbed) / len(perturbed), 3) if perturbed else None,
            "reranker_on_changed_number": rerank_on_fake,
        },
    }
    if args.visual_qa:
        out["numbers_visual_qa"] = visual_numbers(Path(args.visual_qa))
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=1))


def visual_numbers(path: Path) -> dict:
    """표·수식 질문의 맞은 답변(visual_qa 기록)으로 숫자 대조를 본다. 기록에 근거가 없어 질의 변환 없이(LLM 없음)
    검색만 다시 해 근거를 만든다 — 답변 때와 근거가 조금 다를 수 있어, 경보는 '오경보 또는 근거 차이'로 읽는다."""
    from config import settings
    from rag.stages import stage2_rerank as s2
    from rag.store.vectorstore import load_vectorstore

    settings.auto_fetch = False
    vs = load_vectorstore()
    rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    rows = [r for r in rows if r.get("correct") and r.get("answer")]
    alarms, tested, caught, numeric = [], 0, 0, 0
    for r in rows:
        q = r["question"]
        _, chunks = s2.retrieve_and_rerank(vs, q, [q])
        chunk_nums = [cc.chunk_numbers(c.content) for c in chunks]
        q_nums = set(cc.answer_numbers(q))
        for s in cc.split_sentences(r["answer"]):
            if not cc._is_claim(s):
                continue
            cites = cc.parse_cites(s)
            missing, where, did = cc.check_numbers(s, cites, chunk_nums, q_nums)
            numeric += did
            if missing:
                alarms.append((r["kind"], missing, where, cc.claim_text(s)[:140]))
                continue
            nums = cc.answer_numbers(cc.claim_text(s), q_nums)
            valid = [n for n in cites if 1 <= n <= len(chunks)] or list(range(1, len(chunks) + 1))
            if nums and nums[0] in s:
                have = set().union(*(chunk_nums[n - 1] for n in valid))
                fake = _perturb(nums[0], have)
                miss, _, _ = cc.check_numbers(s.replace(nums[0], fake, 1), cites, chunk_nums, q_nums)
                tested += 1
                caught += fake in miss
    print(f"\n[표·수식 답변 {len(rows)}개 — 숫자 경보]")
    for a in alarms:
        print(" -", a)
    return {"answers": len(rows), "numeric_sentences": numeric, "alarms": len(alarms),
            "changed_number_tested": tested, "changed_number_caught": round(caught / tested, 3) if tested else None}


def _perturb(num: str, have: set[str]) -> str:
    """마지막 자리를 바꾼 그럴듯한 틀린 숫자(근거에 우연히 있는 값은 피함). 예: 78.4 → 78.5"""
    for d in range(1, 10):
        last = (int(num[-1]) + d) % 10
        cand = num[:-1] + str(last)
        if cand not in have and cand != num and not cand.endswith(".0"):
            return cand
    return num + "1"


def _rescore(pairs, table, sents):
    """check() 가 같은 (문장, 근거) 순서로 묻는 점수를 이미 계산한 표에서 돌려준다(리랭커 재호출 없이)."""
    k = len(table[0]) if table else 0
    return [table[i // k][i % k] for i in range(len(pairs))]


if __name__ == "__main__":
    main()
