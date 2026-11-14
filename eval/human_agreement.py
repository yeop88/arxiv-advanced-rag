"""
eval/human_agreement.py — 자동 판정이 사람 판정과 얼마나 맞는가 (LLM 없음)

기준 세트(storage/human_set.jsonl, eval/human_set.py)와 사람 판정(storage/human_labels.json, 챗봇 '사람 판정' 화면)으로
  문장 단위 : 4b 인용 확인 경보(근거 약함·번호 잘못·숫자 불일치 …)가 사람의 '근거 없음/번호 틀림'을 얼마나 잡는지
  답변 단위 : 4단계 LLM 검증(grounded·faithfulness)이 사람의 '맞음/일부 틀림/틀림'과 얼마나 맞는지
를 잰다. 우연히 맞는 비율을 뺀 Cohen's κ 를 함께 낸다(0.6 이상이면 상당한 일치로 본다).
판정마다 annotator(human 또는 판정한 모델 이름)가 있다 — 사람이 아닌 모델이 판정했다면 결과는 '독립 모델 판정과의 일치율'로 읽는다.
--extra 에 '문항 id → 문장별 경보(bool) 목록' JSON 을 주면 다른 자동 판정기(예: 함의 판정)도 같은 기준으로 잰다.

사용: python eval/human_agreement.py [--extra storage/nli_flags.json]
결과: eval/results/human_agreement.json (문장 원문 없이 수치만)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import settings  # noqa: E402

OUT = Path(__file__).resolve().parent / "results" / "human_agreement.json"


def kappa(a: list[bool], b: list[bool]) -> float | None:
    """두 판정(참/거짓)의 Cohen's κ."""
    n = len(a)
    if not n:
        return None
    po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return round((po - pe) / (1 - pe), 3) if pe < 1 else None


def flag_scores(human_bad: list[bool], auto_flag: list[bool]) -> dict:
    """자동 경보가 사람의 '문제 있음'을 잡는 정도."""
    tp = sum(h and f for h, f in zip(human_bad, auto_flag))
    fp = sum((not h) and f for h, f in zip(human_bad, auto_flag))
    fn = sum(h and (not f) for h, f in zip(human_bad, auto_flag))
    tn = sum((not h) and (not f) for h, f in zip(human_bad, auto_flag))
    n = len(human_bad)
    return {
        "n": n, "human_bad": tp + fn, "auto_flagged": tp + fp, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "flag_precision": round(tp / (tp + fp), 3) if tp + fp else None,   # 경보 중 실제 문제 비율
        "flag_recall": round(tp / (tp + fn), 3) if tp + fn else None,      # 실제 문제 중 잡은 비율
        "accuracy": round((tp + tn) / n, 3) if n else None,
        "kappa": kappa(human_bad, auto_flag),
    }


_OK = ("supported", "inherited", "summary", "table_row")   # citation_check.OK_STATUSES (모델을 올리지 않으려 복사)


def citation_flag(s: dict) -> bool:
    return s["status"] not in _OK or bool(s.get("numbers_missing"))


def recheck(items: dict, labels: dict) -> tuple[dict, dict]:
    """판정한 문장(기준 세트에 저장된 문장 글)을 지금 코드의 인용 확인 결과와 글로 맞춰 비교한다.
    지금 코드가 주장이 아니라고 빼는 문장(소제목 등)은 '경보 없음'으로 센다."""
    from rag.core.schemas import RetrievedChunk
    from rag.stages import citation_check as cc

    hb, af, h_ok, a_ok = [], [], [], []
    for iid, lab in labels.items():
        it = items[iid]
        chunks = [RetrievedChunk(content=x["content"], source=x["source"], title=x["title"], section=x["section"])
                  for x in it["sources"]]
        rep = cc.check(it["answer"], chunks, question=it["question"], nli=False)
        now = {s.text: s.flagged for s in rep.sentences}
        flags = []
        for i, s in enumerate(it["citation_check"]["sentences"]):
            h = lab["sentences"].get(str(i))
            if h is None:
                continue
            hb.append(h != "supported")
            flags.append(now.get(s["text"], False))
            af.append(flags[-1])
        h_ok.append(lab["overall"] == "correct")
        a_ok.append(not any(flags) and not any(r.numbers_missing for r in rep.table_rows))
    ans = {"agreement": round(sum(x == y for x, y in zip(h_ok, a_ok)) / len(h_ok), 3), "kappa": kappa(h_ok, a_ok)}
    return flag_scores(hb, af), ans


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--extra", help="다른 자동 판정기의 문장별 경보 JSON: {문항 id: [bool, ...]}")
    ap.add_argument("--recheck", action="store_true",
                    help="저장된 답변을 지금 코드의 인용 확인으로 다시 채점해 같은 판정과 비교(리랭커, LLM 없음)")
    ap.add_argument("--annotator", default="human",
                    help="이 주체의 판정만 쓴다(기본 human — 사람이 검토해 저장한 것. 모델 초안만 보려면 그 이름)")
    args = ap.parse_args()
    set_path, label_path = Path(settings.human_set_path), Path(settings.human_labels_path)
    if not set_path.exists() or not label_path.exists():
        raise SystemExit("기준 세트나 사람 판정이 없습니다 — eval/human_set.py --build 후 챗봇 '사람 판정' 화면에서 판정하세요.")
    items = {r["id"]: r for r in (json.loads(x) for x in set_path.read_text(encoding="utf-8").splitlines() if x.strip())}
    labels = {k: v for k, v in json.loads(label_path.read_text(encoding="utf-8")).items()
              if v.get("overall") and k in items and v.get("annotator", "human") == args.annotator}
    if not labels:
        raise SystemExit(f"'{args.annotator}' 판정이 아직 없습니다 — 챗봇 '판정' 화면에서 검토·저장하세요.")
    extra = json.loads(Path(args.extra).read_text(encoding="utf-8")) if args.extra else {}

    human_bad, auto_cc, auto_extra, miscite_h, miscite_a = [], [], [], [], []
    by_kind: dict[str, tuple[list, list]] = {}
    ans_h_correct, ans_llm_grounded, ans_cc_clean, faith = [], [], [], {"correct": [], "partial": [], "wrong": []}
    for iid, lab in labels.items():
        it = items[iid]
        sents = it["citation_check"]["sentences"]
        ex = extra.get(iid)
        for i, s in enumerate(sents):
            h = lab["sentences"].get(str(i))
            if h is None:
                continue
            bad = h != "supported"
            human_bad.append(bad)
            auto_cc.append(citation_flag(s))
            miscite_h.append(h == "miscited")
            miscite_a.append(s["status"] == "miscited" or bool(s.get("numbers_found_in")))
            hk, ak = by_kind.setdefault(it["kind"], ([], []))
            hk.append(bad)
            ak.append(citation_flag(s))
            if ex is not None and i < len(ex):
                auto_extra.append((bad, bool(ex[i])))
        v = it.get("llm_verification")
        ans_h_correct.append(lab["overall"] == "correct")
        ans_cc_clean.append(not any(citation_flag(s) for s in sents))
        if v is not None:
            ans_llm_grounded.append((lab["overall"] == "correct", bool(v["grounded"])))
            faith[lab["overall"]].append(v["faithfulness"])

    out = {
        "labeled_items": len(labels), "items": len(items),
        "annotators": sorted({v.get("annotator", "human") for v in labels.values()}),
        "sentence_citation_check": flag_scores(human_bad, auto_cc),
        "sentence_miscite": flag_scores(miscite_h, miscite_a),
        "sentence_by_kind": {k: flag_scores(h, a) for k, (h, a) in by_kind.items()},
        "answer_llm_verification": {
            "n": len(ans_llm_grounded),
            "agreement": round(sum(h == g for h, g in ans_llm_grounded) / len(ans_llm_grounded), 3) if ans_llm_grounded else None,
            "kappa": kappa([h for h, _ in ans_llm_grounded], [g for _, g in ans_llm_grounded]),
            "llm_grounded_rate": round(sum(g for _, g in ans_llm_grounded) / len(ans_llm_grounded), 3) if ans_llm_grounded else None,
            "human_correct_rate": round(sum(ans_h_correct) / len(ans_h_correct), 3) if ans_h_correct else None,
            "faithfulness_by_human": {k: round(sum(v) / len(v), 3) if v else None for k, v in faith.items()},
        },
        "answer_citation_check_clean_vs_human_correct": {
            "agreement": round(sum(a == b for a, b in zip(ans_h_correct, ans_cc_clean)) / len(ans_h_correct), 3) if ans_h_correct else None,
            "kappa": kappa(ans_h_correct, ans_cc_clean),
        },
    }
    if auto_extra:
        out["sentence_extra"] = flag_scores([h for h, _ in auto_extra], [a for _, a in auto_extra])
    drafts = [(v, v["model_draft"]) for v in labels.values() if v.get("model_draft")]
    if drafts:   # 사람이 모델 초안을 얼마나 고쳤나
        same_s = [d["sentences"].get(i) == s for v, d in drafts for i, s in v["sentences"].items()]
        out["human_vs_model_draft"] = {
            "items": len(drafts), "draft_annotators": sorted({d.get("annotator") for _, d in drafts}),
            "overall_same": round(sum(v["overall"] == d["overall"] for v, d in drafts) / len(drafts), 3),
            "sentence_same": round(sum(same_s) / len(same_s), 3) if same_s else None,
        }
    if args.recheck:
        out["sentence_citation_check_recheck"], out["answer_citation_check_recheck"] = recheck(items, labels)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
