"""
eval/citation_check_eval.py — 문장 단위 인용 확인(rag/stages/citation_check.py)의 기준 점수 보정 (LLM 없음)

수집 신호 평가 기록(eval/results/fetch_signal_record.jsonl — 실제 첫 답변 81개와 그때의 근거 원문)을 다시 쓴다.
사람이 붙인 정답 판정은 없으므로 대리 지표로 본다:
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
    args = ap.parse_args()
    rows = [json.loads(line) for line in RECORD.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = [r for r in rows if r.get("answer") and r.get("evidence")]
    rng = random.Random(0)
    from rag.core.llm import get_reranker
    reranker = get_reranker()

    pos, hard, easy = [], [], []
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
                       threshold=args.threshold)
        statuses.update(s.status for s in rep.sentences)
        per_answer.append(rep.summary())

    grid = [{"threshold": t,
             "pos_pass": round(sum(v >= t for v in pos) / len(pos), 3),
             "hard_neg_pass": round(sum(v >= t for v in hard) / len(hard), 3),
             "easy_neg_pass": round(sum(v >= t for v in easy) / len(easy), 3)} for t in GRID]
    cited_answers = [a for a in per_answer if a["precision"] is not None]
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
    }
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=1))


def _rescore(pairs, table, sents):
    """check() 가 같은 (문장, 근거) 순서로 묻는 점수를 이미 계산한 표에서 돌려준다(리랭커 재호출 없이)."""
    k = len(table[0]) if table else 0
    return [table[i // k][i % k] for i in range(len(pairs))]


if __name__ == "__main__":
    main()
