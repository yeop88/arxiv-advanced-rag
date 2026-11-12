"""
eval/bm25_tokenizer_eval.py — BM25 단어 나누기 방식 비교(LLM 없음). DB 는 읽기만 한다.

평가셋 294문항을 질의 변환 없이(원 한국어 질문 1개) 하이브리드 검색 → 리랭크해 recall@5·MRR 을 잰다.
BM25 만 단독으로 쓴 recall@20(후보 단계)도 함께 본다 — 리랭커가 가려 주기 전의 키워드 검색 자체 품질.
    python eval/bm25_tokenizer_eval.py [--corpus base|all]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import settings  # noqa: E402

OUT = Path(__file__).resolve().parent / "results" / "bm25_tokenizer_eval.json"
MODES = ["whitespace", "regex", "kiwi"]


def main() -> None:
    from eval.corpus import set_corpus
    from eval.dataset import load_eval_set
    from eval.retrieval_metrics import recall_at_k, reciprocal_rank
    from rag import stage2_rerank as s2
    from rag.vectorstore import load_vectorstore

    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="base", choices=["base", "all"])
    args = ap.parse_args()
    set_corpus(args.corpus)
    settings.use_hybrid = True
    vs = load_vectorstore()
    records = [r for r in load_eval_set() if r.get("expected_sources")]
    results = {}
    for mode in MODES:
        settings.bm25_tokenizer = mode
        s2.reset_bm25()
        t0 = time.perf_counter()
        bm25 = s2._get_bm25(id(vs)) if s2._BM25_STORE.update(vs=vs) is None else None
        build = time.perf_counter() - t0
        rec, mrr, bm_rec, hard_rec = [], [], [], []
        for r in records:
            exp = r["expected_sources"]
            _, rr = s2.retrieve_and_rerank(vs, r["question"], [r["question"]])
            rec.append(recall_at_k(rr, exp, settings.top_k_rerank))
            mrr.append(reciprocal_rank(rr, exp))
            got = {d.metadata.get("source") for d in bm25.invoke(r["question"])[:20]}
            bm_rec.append(len(got & set(exp)) / len(exp))
            if r.get("difficulty") == "hard":
                hard_rec.append(rec[-1])
        results[mode] = {"recall@5": sum(rec) / len(rec), "mrr": sum(mrr) / len(mrr),
                         "bm25_recall@20": sum(bm_rec) / len(bm_rec), "hard_recall@5": sum(hard_rec) / len(hard_rec),
                         "index_sec": round(build, 1)}
        print(f"{mode:<11} recall@5 {results[mode]['recall@5']:.4f}  MRR {results[mode]['mrr']:.4f}  "
              f"hard recall@5 {results[mode]['hard_recall@5']:.4f}  BM25 단독 recall@20 {results[mode]['bm25_recall@20']:.4f}  "
              f"색인 {build:.1f}초", flush=True)
    OUT.write_text(json.dumps({"corpus": args.corpus, "n": len(records), "results": results}, ensure_ascii=False, indent=1),
                   encoding="utf-8")


if __name__ == "__main__":
    main()
