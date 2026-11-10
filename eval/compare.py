"""
eval/compare.py — 설정 조합별 A/B 평가 자동화

여러 파이프라인 설정(하이브리드 ON/OFF, 질의변환 모드)을 같은 평가셋에 돌려
지표를 비교표로 출력합니다. "무엇이 실제로 품질을 올리는가"를 감이 아니라
숫자로 확인하기 위한 도구입니다.

두 종류의 지표:
  - 검색 지표:  recall@k, MRR  — expected_sources 와 검색 결과만으로 계산(결정적)
  - 생성 지표:  faithfulness / answer_rel / context_rel / answer_rate — LLM 심판 호출(유료)

비용 참고: `--retrieval-only` 는 생성·검증·LLM심판을 전부 생략해 질문당 비용을 크게 줄인다
(약 5회 LLM 호출 → 질의변환 1회만). 질의변환이 `none`(baseline)인 설정은 LLM 호출이 0이라
완전 무료다. 수백~수천 문항으로 '검색 품질 회귀'를 싸게 스크리닝할 때 쓴다.

평가셋은 eval/eval_set.jsonl 한 곳에서 로드합니다(코드 수정 없이 확장 가능).

사용:
    python eval/compare.py                       # 전체 평가셋, 전 지표
    python eval/compare.py --n 4                  # 질문 수 제한(비용 절감)
    python eval/compare.py --difficulty hard      # 난이도 필터
    python eval/compare.py --retrieval-only       # 검색 지표만 — 대량 스크리닝(저비용)
    python eval/compare.py --retrieval-only --only baseline --corpus both   # LLM 0회, 두 코퍼스 비교

코퍼스: 기본은 자동 수집 논문을 숨긴 기준 코퍼스(--corpus base) → 이전 수치와 그대로 비교된다.
--corpus all 은 자동 수집분 포함(실제 챗봇 상태), both 는 둘 다(eval/corpus.py 참고).
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")  # 캐시 등 라이브러리 DeprecationWarning 노이즈 억제

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import settings  # noqa: E402

settings.auto_fetch = False  # 평가는 코퍼스를 고정한다(근거 부족 시 논문 자동 수집 끔)
from eval.corpus import LABELS, add_corpus_arg, corpora, describe, set_corpus  # noqa: E402
from eval.dataset import load_eval_set  # noqa: E402
from eval.ragas_lite import evaluate_result, is_answered  # noqa: E402
from eval.retrieval_metrics import recall_at_k, reciprocal_rank  # noqa: E402
from rag.pipeline import RAGPipeline  # noqa: E402
from rag.stage1_query_transform import transform_query  # noqa: E402
from rag.stage2_rerank import retrieve_and_rerank  # noqa: E402
from rag.vectorstore import load_vectorstore  # noqa: E402

# 비교할 설정 조합 — (이름, {바꿀 설정들})
CONFIGS = [
    ("baseline (질의변환X · 벡터만)", {"query_transform_mode": "none", "use_hybrid": False}),
    ("multi_query · 벡터만",          {"query_transform_mode": "multi_query", "use_hybrid": False}),
    ("multi_query · 하이브리드",       {"query_transform_mode": "multi_query", "use_hybrid": True}),
    ("hyde · 하이브리드",             {"query_transform_mode": "hyde", "use_hybrid": True}),
]


def apply_config(overrides: dict) -> None:
    for k, v in overrides.items():
        setattr(settings, k, v)


def _retrieve_only(vs, question: str):
    """현재 설정(질의변환 모드/하이브리드)으로 검색+리랭킹만 수행 → reranked 청크.

    생성·검증·LLM심판을 전부 건너뛰므로 비용이 거의 없다(질의변환 light-LLM 1회뿐).
    질의변환 실패 시 원 질문으로 degrade.
    """
    try:
        queries = transform_query(question, settings.query_transform_mode)
    except Exception:  # noqa: BLE001
        queries = [question]
    _, reranked = retrieve_and_rerank(vs, question, queries)
    return reranked


def run_config(pipe, vs, records: list[dict], retrieval_only: bool) -> dict[str, float]:
    """한 설정으로 평가셋 전체를 돌려 평균 지표 반환."""
    k = settings.top_k_rerank
    recalls, mrrs = [], []
    rows, answered = [], 0

    for rec in records:
        q = rec["question"]
        exp = rec.get("expected_sources", [])

        if retrieval_only:
            reranked = _retrieve_only(vs, q)
        else:
            result = pipe.run(q)
            reranked = result.reranked
            answered += int(is_answered(result.answer))
            rows.append(evaluate_result(result))

        if exp:  # expected_sources 가 있는 질문만 검색 지표 집계
            recalls.append(recall_at_k(reranked, exp, k))
            mrrs.append(reciprocal_rank(reranked, exp))

    def avg(xs):
        return sum(xs) / len(xs) if xs else 0.0

    out = {
        f"recall@{k}": avg(recalls),
        "mrr": avg(mrrs),
    }
    if not retrieval_only:
        out.update({
            "faithfulness": avg([r["faithfulness"] for r in rows]),
            "answer_relevance": avg([r["answer_relevance"] for r in rows]),
            "context_relevance": avg([r["context_relevance"] for r in rows]),
            "answer_rate": answered / len(records) if records else 0.0,
        })
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=None, help="사용할 질문 수(앞에서부터)")
    parser.add_argument("--difficulty", default=None, help="난이도 필터: easy|medium|hard")
    parser.add_argument("--category", default=None, help="카테고리 필터")
    parser.add_argument("--retrieval-only", action="store_true",
                        help="검색 지표(recall@k·MRR)만 — 생성·검증·심판 생략(저비용 대량 스크리닝)")
    parser.add_argument("--only", nargs="+", default=None, metavar="TEXT",
                        help="이름에 TEXT 가 들어간 설정만 실행 (예: --only baseline → LLM 0회)")
    add_corpus_arg(parser, allow_both=True)
    args = parser.parse_args()

    records = load_eval_set(difficulty=args.difficulty, category=args.category, limit=args.n)
    if not records:
        print("평가셋이 비었습니다(필터 확인). eval/eval_set.jsonl")
        return
    configs = [c for c in CONFIGS if not args.only or any(t in c[0] for t in args.only)]
    if not configs:
        print(f"--only {args.only} 에 맞는 설정이 없습니다: {[c[0] for c in CONFIGS]}")
        return
    k = settings.top_k_rerank
    results: list[tuple[str, str, dict]] = []  # (코퍼스, 설정 이름, 지표)

    for corpus in corpora(args.corpus):
        set_corpus(corpus)
        vs = load_vectorstore()
        pipe = None if args.retrieval_only else RAGPipeline(vs)  # 유료 모드일 때만 전체 파이프라인
        print(f"\n■ {describe(corpus, vs)}")
        for name, overrides in configs:
            print(f"\n▶ 설정: {name}")
            apply_config(overrides)
            t0 = time.perf_counter()
            metrics = run_config(pipe, vs, records, args.retrieval_only)
            metrics["time"] = time.perf_counter() - t0
            results.append((corpus, name, metrics))
            line = f"  recall@{k}={metrics[f'recall@{k}']:.4f} mrr={metrics['mrr']:.4f}"
            if not args.retrieval_only:
                line += (f" | faith={metrics['faithfulness']:.2f} "
                         f"ans_rel={metrics['answer_relevance']:.2f} "
                         f"ctx_rel={metrics['context_relevance']:.2f} "
                         f"ans_rate={metrics['answer_rate']:.0%}")
            print(line)

    # --- 비교표 (markdown) ---
    print("\n" + "=" * 78)
    mode = "검색 지표만(무료)" if args.retrieval_only else "전 지표"
    print(f"A/B 평가 비교표 (질문 {len(records)}개, {mode})\n")
    if args.retrieval_only:
        print(f"| 코퍼스 | 설정 | recall@{k} | MRR |")
        print("|---|---|---|---|")
        for corpus, name, m in results:
            print(f"| {LABELS[corpus]} | {name} | {m[f'recall@{k}']:.4f} | {m['mrr']:.4f} |")
    else:
        print(f"| 코퍼스 | 설정 | recall@{k} | MRR | faithfulness | answer_rel | context_rel | answer_rate |")
        print("|---|---|---|---|---|---|---|---|")
        for corpus, name, m in results:
            print(f"| {LABELS[corpus]} | {name} | {m[f'recall@{k}']:.4f} | {m['mrr']:.4f} "
                  f"| {m['faithfulness']:.3f} | {m['answer_relevance']:.3f} "
                  f"| {m['context_relevance']:.3f} | {m['answer_rate']:.0%} |")
    print("=" * 78)
    print("※ 이 표를 docs/PROGRESS_REPORT.md 에 붙여 품질 변화를 추적하세요.")


if __name__ == "__main__":
    main()
