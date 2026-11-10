"""
rag/stage4_verify.py — 4단계: 검증 (groundedness / faithfulness)

생성된 답변이 근거에 충실한지 LLM-as-a-judge 로 채점합니다.
이 단계가 Advanced RAG 의 핵심 차별점: "그럴듯하지만 틀린 답"을 걸러냅니다.

흐름:
  1) LLM 검증관이 (근거, 답변)을 비교해 JSON 평가 반환
  2) grounded=False 또는 faithfulness 낮으면 → 3단계로 돌아가 재생성(자기수정)
"""
from __future__ import annotations

import json

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from rag import prompts
from rag.llm import get_llm
from rag.schemas import RetrievedChunk, VerificationResult
from rag.stage3_generate import format_context


def _parse_json(text: str) -> dict:
    """LLM 이 코드펜스(```json ... ```)로 감싸는 경우까지 안전하게 파싱."""
    t = text.strip()
    if t.startswith("```"):
        t = t.split("```")[1] if "```" in t[3:] else t
        t = t.removeprefix("json").strip()
    # 첫 { 부터 마지막 } 까지만 취함
    start, end = t.find("{"), t.rfind("}")
    if start != -1 and end != -1:
        t = t[start : end + 1]
    return json.loads(t)


def verify(answer: str, chunks: list[RetrievedChunk]) -> VerificationResult:
    """답변의 근거 충실도를 검증."""
    context = format_context(chunks)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", prompts.VERIFICATION_SYSTEM),
            ("user", prompts.VERIFICATION_USER),
        ]
    )
    chain = prompt | get_llm(light=False) | StrOutputParser()
    raw = chain.invoke({"context": context, "answer": answer})

    try:
        data = _parse_json(raw)
        return VerificationResult(
            grounded=bool(data.get("grounded", False)),
            faithfulness=float(data.get("faithfulness", 0.0)),
            unsupported_claims=list(data.get("unsupported_claims", []) or []),
            reasoning=str(data.get("reasoning", "")),
        )
    except (json.JSONDecodeError, ValueError, KeyError) as e:
        # 파싱 실패 시 보수적으로 "검증 불가" 처리
        return VerificationResult(
            grounded=False,
            faithfulness=0.0,
            unsupported_claims=[],
            reasoning=f"검증 결과 파싱 실패: {e}",
        )
