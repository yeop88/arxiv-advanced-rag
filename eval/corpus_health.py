"""
eval/corpus_health.py — 코퍼스가 커져도 검색 품질이 유지되는지 점검 (LLM 없음)

평가셋 전체를 자동 수집 논문 제외(기준) / 포함(현재)으로 측정해 차이를 본다.
결과는 eval/results/corpus_health.jsonl 에 쌓아 추세를 볼 수 있다.

판정: recall·MRR 하락이 0.008 이상이면 주의, 0.01 이상이면 초과(종료 코드 1).
초과하면 AUTO_FETCH_MAX_CHUNKS 를 낮추거나 scripts/manage_auto_papers.py 로 정리한다.

원격 벡터 DB 는 대량 조회가 월 전송량 한도를 빠르게 소모하므로 로컬 백엔드에서만 실행한다.

사용:
    python eval/corpus_health.py
    python eval/corpus_health.py --no-record   # 기록 없이 확인만
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import settings  # noqa: E402

settings.auto_fetch = False
settings.query_transform_mode = "none"  # LLM 미사용
settings.use_hybrid = True

from eval.corpus import auto_stats, set_corpus  # noqa: E402
from eval.dataset import load_eval_set  # noqa: E402
from eval.retrieval_metrics import recall_at_k, reciprocal_rank  # noqa: E402
from rag.stage2_rerank import retrieve_and_rerank  # noqa: E402
from rag.vectorstore import load_vectorstore  # noqa: E402

OUT = Path(__file__).resolve().parent / "results" / "corpus_health.jsonl"
WARN, LIMIT = 0.008, 0.01


def measure(vs, records: list[dict]) -> tuple[float, float]:
    k = settings.top_k_rerank
    recalls, mrrs = [], []
    for r in records:
        exp = r.get("expected_sources") or []
        if not exp:
            continue
        _, rr = retrieve_and_rerank(vs, r["question"], [r["question"]])
        recalls.append(recall_at_k(rr, exp, k))
        mrrs.append(reciprocal_rank(rr, exp))
    return sum(recalls) / len(recalls), sum(mrrs) / len(mrrs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-record", action="store_true", help="결과를 기록하지 않음")
    args = ap.parse_args()
    if settings.vector_backend != "chroma":
        raise SystemExit("로컬 백엔드(VECTOR_BACKEND=chroma)에서만 실행합니다. 원격 DB 는 전송량 한도를 소모합니다.")

    records = load_eval_set()
    set_corpus("base")
    base_recall, base_mrr = measure(load_vectorstore(), records)
    set_corpus("all")
    vs = load_vectorstore()
    recall, mrr = measure(vs, records)
    papers, chunks = auto_stats(vs)

    drop = max(base_recall - recall, base_mrr - mrr)
    verdict = "초과" if drop >= LIMIT else "주의" if drop >= WARN else "정상"
    row = {"date": dt.date.today().isoformat(), "auto_papers": papers, "auto_chunks": chunks,
           "budget": settings.auto_fetch_max_chunks, "base_recall": round(base_recall, 4), "base_mrr": round(base_mrr, 4),
           "recall": round(recall, 4), "mrr": round(mrr, 4), "verdict": verdict}
    print(f"자동 수집 {papers}편 / {chunks}청크 (예산 {settings.auto_fetch_max_chunks})")
    print(f"recall@{settings.top_k_rerank}: 기준 {base_recall:.4f} → 현재 {recall:.4f} ({recall - base_recall:+.4f})")
    print(f"MRR: 기준 {base_mrr:.4f} → 현재 {mrr:.4f} ({mrr - base_mrr:+.4f})")
    print(f"판정: {verdict} (주의 {WARN}, 초과 {LIMIT})")
    if not args.no_record:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        with OUT.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    sys.exit(1 if verdict == "초과" else 0)


if __name__ == "__main__":
    main()
