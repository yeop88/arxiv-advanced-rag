"""
eval/compare_table_eval.py — 논문 비교표(C1) 품질 측정. DB 는 읽기만 한다.

논문 묶음마다 비교표를 만들어 채운 칸 비율, 근거 번호 없는 칸, 4단계 검증(grounded·faithfulness), 소요 시간을 기록한다.
LLM: 로컬(논문당 1회 + 검증 1회).
    python eval/compare_table_eval.py
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

OUT = Path(__file__).resolve().parent / "results" / "compare_table_eval.jsonl"
SETS = {
    "dense_vs_late": ["2004.04906", "2004.12832", "2107.05720"],      # DPR · ColBERT · SPLADE
    "retrieval_lm": ["2005.11401", "2310.11511", "2002.08909"],       # RAG · Self-RAG · REALM
    "bert_family": ["1810.04805", "1907.11692"],                      # BERT · RoBERTa
    "llm_tuning": ["2106.09685", "2203.02155", "2307.09288"],         # LoRA · InstructGPT · Llama 2
    "query_side": ["2404.16130", "2212.10496", "2305.14283"],         # GraphRAG · HyDE · Query Rewriting
}


def main() -> None:
    from rag.compare_table import build_table
    from rag.vectorstore import load_vectorstore

    if settings.llm_provider != "local":
        raise SystemExit("로컬 LLM(LLM_PROVIDER=local)에서만 실행합니다.")
    settings.local_thinking = "off"
    settings.exclude_auto_papers = False
    vs = load_vectorstore()
    OUT.unlink(missing_ok=True)
    rows = []
    for name, srcs in SETS.items():
        t0 = time.perf_counter()
        t = build_table(srcs, vs)
        v = t.verification
        row = {"set": name, "papers": [r.title[:50] for r in t.rows], "cells": t.cells_total,
               "filled": t.cells_filled, "uncited": t.cells_uncited,
               "uncited_cells": {r.title[:30]: r.uncited for r in t.rows if r.uncited},
               "grounded": v.grounded if v else None, "faithfulness": v.faithfulness if v else None,
               "unsupported": v.unsupported_claims if v else None,
               "sec": round(time.perf_counter() - t0, 1), "table": t.to_markdown()}
        rows.append(row)
        with OUT.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"{name:<14} 논문 {len(t.rows)} 칸 {row['filled']}/{row['cells']} 인용없음 {row['uncited']} "
              f"grounded={row['grounded']} faith={row['faithfulness']} {row['sec']}s", flush=True)
    cells = sum(r["cells"] for r in rows)
    print(f"\n합계: 채운 칸 {sum(r['filled'] for r in rows)}/{cells}, 근거 번호 없는 칸 {sum(r['uncited'] for r in rows)}, "
          f"검증 통과 {sum(bool(r['grounded']) for r in rows)}/{len(rows)}, "
          f"평균 {sum(r['sec'] for r in rows) / len(rows):.0f}초")


if __name__ == "__main__":
    main()
