"""
rag/contextualize.py — 0단계(전처리): 멀티턴 질의 문맥화

후속질문("그건 왜?", "더 자세히")을 대화 히스토리로 독립형 질문으로 재작성.
히스토리가 없으면 아무 것도 하지 않아 기존 단일턴 동작과 100% 호환됩니다.
"""
from __future__ import annotations

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from rag import prompts
from rag.llm import get_llm

# (role, content) 튜플 리스트로 히스토리를 받음
ChatHistory = list[tuple[str, str]]


def _format_history(history: ChatHistory, max_turns: int = 6) -> str:
    """최근 몇 턴만 '사용자/AI' 레이블로 포맷."""
    recent = history[-max_turns:]
    lines = []
    for role, content in recent:
        label = "사용자" if role == "user" else "AI"
        lines.append(f"{label}: {content}")
    return "\n".join(lines)


def contextualize(question: str, history: ChatHistory | None) -> str:
    """대화 히스토리를 참고해 질문을 독립형으로 재작성.

    Args:
        question: 마지막 사용자 질문
        history: [(role, content), ...] — 직전까지의 대화. 없으면 원문 반환.
    """
    if not history:
        return question

    prompt = ChatPromptTemplate.from_template(prompts.CONTEXTUALIZE_PROMPT)
    chain = prompt | get_llm(light=True) | StrOutputParser()  # 쉬운 작업 → 경량 모델
    standalone = chain.invoke(
        {"history": _format_history(history), "question": question}
    ).strip()
    # 빈 응답 방어
    return standalone or question
