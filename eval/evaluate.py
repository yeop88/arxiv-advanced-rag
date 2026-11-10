"""
eval/evaluate.py — RAGAS 스타일 평가 하네스

왜 평가가 중요한가?
  "프롬프트를 바꿨더니 좋아진 것 같다"는 감(感)이 아니라 숫자로 확인해야
  제품급으로 발전시킬 수 있습니다. 작은 평가셋부터 시작하세요.

측정 지표 (eval/ragas_lite.py):
  - faithfulness       : 답변이 근거에 충실한가
  - answer_relevance   : 답변이 질문에 적절히 답하는가
  - context_relevance  : 검색된 근거가 질문에 관련 있는가
  - answer_rate        : 근거를 찾아 답한 비율

사용:
    python eval/evaluate.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import settings  # noqa: E402

settings.auto_fetch = False  # 평가는 코퍼스를 고정한다(근거 부족 시 논문 자동 수집 끔)
from eval.dataset import questions as load_questions  # noqa: E402
from eval.ragas_lite import evaluate_result, is_answered  # noqa: E402
from rag.pipeline import RAGPipeline  # noqa: E402

# 평가셋은 eval/eval_set.jsonl 한 곳에서 로드(코드 수정 없이 확장 가능).
EVAL_QUESTIONS = load_questions()


def main() -> None:
    pipe = RAGPipeline()

    rows: list[dict] = []
    answered = 0

    for q in EVAL_QUESTIONS:
        print(f"\n질문: {q}")
        result = pipe.run(q)

        answered += int(is_answered(result.answer))

        metrics = evaluate_result(result)
        rows.append(metrics)
        print(f"  faithfulness={metrics['faithfulness']:.2f}  "
              f"answer_rel={metrics['answer_relevance']:.2f}  "
              f"context_rel={metrics['context_relevance']:.2f}")

    n = len(EVAL_QUESTIONS)

    def avg(key: str) -> float:
        return sum(r[key] for r in rows) / len(rows) if rows else 0.0

    print("\n" + "=" * 52)
    print(f"RAGAS 스타일 평가 결과 (n={n})")
    print(f"  평균 faithfulness      : {avg('faithfulness'):.3f}")
    print(f"  평균 answer_relevance  : {avg('answer_relevance'):.3f}")
    print(f"  평균 context_relevance : {avg('context_relevance'):.3f}")
    print(f"  answer_rate            : {answered}/{n} ({answered / n:.0%})")
    print("=" * 52)
    print("※ config 의 use_hybrid / query_transform_mode 를 바꿔가며 점수를 비교해보세요.")


if __name__ == "__main__":
    main()
