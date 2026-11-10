"""
rag/stage1_query_transform.py — 1단계: 질의 변환

왜 필요한가?
  사용자의 질문 표현과 논문의 표현은 다릅니다(어휘 불일치).
  질문을 여러 각도로 바꾸거나(Multi-Query), 가상의 답변으로 바꾸거나(HyDE),
  하위 질문으로 쪼개면(Decompose) 검색 재현율이 크게 올라갑니다.

세 가지 전략을 모두 구현하고 config 로 선택합니다.
"""
from __future__ import annotations

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from config import settings
from rag import prompts
from rag.llm import get_llm


def _split_lines(text: str) -> list[str]:
    """LLM 출력(줄 단위)을 리스트로. 빈 줄·번호 접두 제거."""
    lines = []
    for raw in text.strip().splitlines():
        s = raw.strip().lstrip("0123456789.-) ").strip()
        if s:
            lines.append(s)
    return lines


def transform_query(question: str, mode: str | None = None) -> list[str]:
    """질문을 검색용 질의 리스트로 변환.

    Returns:
        검색에 사용할 질의 문자열 리스트. 원 질문도 항상 포함(안전장치).
    """
    mode = mode or settings.query_transform_mode
    llm = get_llm(light=True)  # 쉬운 작업이므로 가벼운 모델

    if mode == "none":
        return [question]

    if mode == "multi_query":
        prompt = ChatPromptTemplate.from_template(prompts.MULTI_QUERY_PROMPT)
        chain = prompt | llm | StrOutputParser()
        out = chain.invoke({"question": question, "n": settings.num_query_variations})
        variations = _split_lines(out)
        # 원 질문 + 변형들 (중복 제거, 순서 유지)
        return list(dict.fromkeys([question, *variations]))

    if mode == "hyde":
        prompt = ChatPromptTemplate.from_template(prompts.HYDE_PROMPT)
        chain = prompt | llm | StrOutputParser()
        hypothetical = chain.invoke({"question": question}).strip()
        # HyDE 는 가상답변으로 검색하되, 원 질문도 함께 써서 안정성 확보
        return [hypothetical, question]

    if mode == "decompose":
        prompt = ChatPromptTemplate.from_template(prompts.DECOMPOSE_PROMPT)
        chain = prompt | llm | StrOutputParser()
        out = chain.invoke({"question": question, "n": settings.num_query_variations})
        subs = _split_lines(out)
        return subs or [question]

    raise ValueError(f"알 수 없는 query_transform_mode: {mode}")
