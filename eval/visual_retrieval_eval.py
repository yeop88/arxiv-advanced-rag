"""
eval/visual_retrieval_eval.py — 표·그림 청크 리랭크 가중치(VISUAL_RERANK_WEIGHT) 비교 (LLM 없음)

가중치마다 두 가지를 잰다.
  - 평가셋 294문항 recall@5 / MRR: 표·그림 청크가 무관한 질문의 근거를 밀어내는지(이슈 20)
  - 표·그림·수식 질문(visual_qa_eval.QUESTIONS): 그 논문의 해당 종류 청크가 top-5 에 드는지
검색·리랭커 점수는 질문마다 한 번만 계산하고, 가중치만 바꿔 MMR 선택을 다시 한다(실제 rerank() 와 같은 계산).
    python eval/visual_retrieval_eval.py --weights 1.0 0.9 --floors 0 0.05 0.1 0.2 [--tag figures]
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import settings  # noqa: E402
from rag.schemas import VISUAL_KINDS  # noqa: E402

RESULTS = Path(__file__).resolve().parent / "results"


def _prepare(vs, question: str):
    """후보 청크와 리랭커 원점수(가중치 적용 전)."""
    from rag.llm import get_reranker
    from rag.stage2_rerank import retrieve_hybrid

    cands = retrieve_hybrid(vs, [question])
    raw = get_reranker().predict([(question, c.content) for c in cands]) if cands else []
    return cands, [float(s) for s in raw]


def _select(cands, raw, weight: float, vectors, floor: float = 0.0):
    from rag.stage2_rerank import _mmr_select

    for c, s in zip(cands, raw):
        c.rerank_score = s * (settings.auto_rerank_weight if c.origin == "auto" else 1.0)
        if c.kind in VISUAL_KINDS:
            c.rerank_score *= weight
    cands = [c for c in cands if c.kind not in VISUAL_KINDS or c.rerank_score >= floor]
    ranked = sorted(cands, key=lambda c: c.rerank_score or 0.0, reverse=True)
    return _mmr_select(ranked, settings.top_k_rerank, settings.mmr_lambda, settings.same_source_penalty,
                       settings.rerank_max_per_source, vectors=vectors)


def main() -> None:
    from eval.dataset import load_eval_set
    from eval.retrieval_metrics import recall_at_k, reciprocal_rank
    from eval.visual_qa_eval import QUESTIONS
    from rag.vectorstore import get_vectors, load_vectorstore

    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=float, nargs="+", default=[1.0, 0.9, 0.8, 0.7])
    ap.add_argument("--floors", type=float, nargs="+", default=[0.0], help="표·그림 최소 리랭크 점수(VISUAL_MIN_SCORE)")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    settings.exclude_auto_papers = False
    settings.use_hybrid = True
    vs = load_vectorstore()
    kinds = [(m or {}).get("kind") for m in vs.get()["metadatas"]]

    evalset = [r for r in load_eval_set() if r.get("expected_sources")]
    prepared = []
    for r in evalset:
        cands, raw = _prepare(vs, r["question"])
        prepared.append((r, cands, raw, get_vectors(vs, [c.id for c in cands if c.id])))
    table_prepared = []
    for q, _exp, src, kind in QUESTIONS:
        cands, raw = _prepare(vs, q)
        table_prepared.append((q, src, kind, cands, raw, get_vectors(vs, [c.id for c in cands if c.id])))

    out = {"tables": kinds.count("table"), "figures": kinds.count("figure"), "equations": kinds.count("equation"), "rows": []}
    for w, floor in [(w, f) for w in args.weights for f in args.floors]:
        recalls, mrrs, intrusions, hits = [], [], 0, []
        for r, cands, raw, vec in prepared:
            sel = _select(cands, raw, w, vec, floor)
            recalls.append(recall_at_k(sel, r["expected_sources"], settings.top_k_rerank))
            mrrs.append(reciprocal_rank(sel, r["expected_sources"]))
            # 정답이 아닌 논문의 표·그림 청크가 근거에 들어간 문항 수
            intrusions += any(c.kind in VISUAL_KINDS and c.source not in r["expected_sources"] for c in sel)
        for q, src, kind, cands, raw, vec in table_prepared:
            sel = _select(cands, raw, w, vec, floor)
            hits.append((kind, any(c.kind == kind and c.source == src for c in sel)))
        row = {"weight": w, "floor": floor, "recall": round(sum(recalls) / len(recalls), 4), "mrr": round(sum(mrrs) / len(mrrs), 4),
               "intrusion_questions": intrusions,
               "hits": {k: [sum(h for kk, h in hits if kk == k), sum(kk == k for kk, _ in hits)] for k in VISUAL_KINDS},
               "recalls": recalls}
        out["rows"].append(row)
        print(f"weight {w:.2f} floor {floor:.2f}: recall {row['recall']:.4f} MRR {row['mrr']:.4f} | 무관한 표·그림 끼어든 문항 {intrusions}"
              f" | 질문의 그 논문 표·그림·수식이 top-5 {row['hits']}", flush=True)
    name = f"visual_retrieval{'_' + args.tag if args.tag else ''}.json"
    (RESULTS / name).write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    print(f"표 {out['tables']}개·그림 {out['figures']}개·수식 {out['equations']}개, 결과 {name}")


if __name__ == "__main__":
    main()
