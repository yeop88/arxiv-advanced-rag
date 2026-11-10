"""
eval/ragas_lite.py — RAGAS 스타일 지표의 경량 구현 (LLM-as-judge)

RAGAS 정식 패키지는 의존성이 무겁고 자체 LLM 래퍼가 필요합니다. 여기서는
프로젝트의 get_llm 을 재사용해 동일한 개념을 가볍게 측정합니다.

3대 지표:
  - faithfulness       : 답변이 근거에 충실한가 (4단계 검증관 재사용)
  - answer_relevance   : 답변이 질문에 적절히 답하는가
  - context_relevance  : 검색된 근거가 질문에 관련 있는가
"""
from __future__ import annotations

import json

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from rag import prompts
from rag.llm import get_llm
from rag.schemas import PipelineResult
from rag.stage3_generate import format_context
from rag.stage4_verify import _parse_json, verify


# 파이프라인이 근거 0건으로 종료할 때의 고정 메시지
_NO_EVIDENCE = "관련 근거를 찾지 못했습니다"


def is_answered(answer: str) -> bool:
    """답변이 '실질적으로 답했는지' 판정.

    주의: 생성 프롬프트는 근거가 부족한 부분을 '찾을 수 없음'으로 정직하게 명시하므로,
    단순 substring 매칭은 '단서를 단 긴 답변'을 미답변으로 오판한다. 따라서
    (1) 파이프라인의 근거-0건 메시지인지, (2) 답변이 충분히 긴지로 판정한다.
    """
    a = answer.strip()
    if _NO_EVIDENCE in a:
        return False
    return len(a) >= 150  # 전면 거부는 짧다 / 실질 답변은 길다


def _judge_score(prompt_template: str, score_key: str, **kwargs) -> float:
    """LLM 심판으로 점수 하나를 뽑는 공통 헬퍼."""
    prompt = ChatPromptTemplate.from_template(prompt_template)
    chain = prompt | get_llm(light=False) | StrOutputParser()
    raw = chain.invoke(kwargs)
    try:
        data = _parse_json(raw)
        return float(data.get(score_key, 0.0))
    except (json.JSONDecodeError, ValueError, KeyError):
        return 0.0


def answer_relevance(question: str, answer: str) -> float:
    return _judge_score(
        prompts.ANSWER_RELEVANCE_PROMPT, "answer_relevance",
        question=question, answer=answer,
    )


def context_relevance(question: str, chunks) -> float:
    return _judge_score(
        prompts.CONTEXT_RELEVANCE_PROMPT, "context_relevance",
        question=question, context=format_context(chunks),
    )


def evaluate_result(result: PipelineResult) -> dict[str, float]:
    """한 PipelineResult 에 대해 3대 지표를 계산해 dict 로 반환."""
    # faithfulness: 4단계 검증 결과가 있으면 재사용, 없으면 새로 계산
    if result.verification is not None:
        faith = result.verification.faithfulness
    else:
        faith = verify(result.answer, result.reranked).faithfulness

    return {
        "faithfulness": faith,
        "answer_relevance": answer_relevance(result.question, result.answer),
        "context_relevance": context_relevance(result.question, result.reranked),
    }
