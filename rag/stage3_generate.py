"""
rag/stage3_generate.py — 3단계: 생성 (인용 기반 답변)

리랭킹으로 고른 근거만 LLM 에 넣고, 각 문장에 [번호] 인용을 달아 답하게 합니다.
근거를 번호로 포맷해서 넣는 것이 환각 억제 + 출처 추적의 핵심.
"""
from __future__ import annotations

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from rag import prompts
from rag.llm import get_llm
from rag.schemas import RetrievedChunk


def format_context(chunks: list[RetrievedChunk]) -> str:
    """근거 청크들을 [번호] 형식의 문자열로 변환 (인용 추적 가능하게).

    섹션 정보(section)가 있으면 헤더에 포함해, 전문 기반일 때 인용이
    "어느 섹션에서 나온 근거인지"까지 드러나게 한다.
    """
    blocks = []
    for i, c in enumerate(chunks, start=1):
        sec = f" / {c.section}" if c.section else ""
        header = f"[{i}] ({c.title or 'unknown'}{sec} / arXiv:{c.source or 'N/A'})"
        blocks.append(f"{header}\n{c.content}")
    return "\n\n".join(blocks)


def generate(question: str, chunks: list[RetrievedChunk]) -> str:
    """근거 기반 답변 생성."""
    context = format_context(chunks)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", prompts.GENERATION_SYSTEM),
            ("user", prompts.GENERATION_USER),
        ]
    )
    chain = prompt | get_llm(light=False) | StrOutputParser()
    return chain.invoke({"context": context, "question": question}).strip()


def regenerate(question: str, chunks: list[RetrievedChunk], unsupported: list[str]) -> str:
    """검증 실패 시, 문제점 피드백을 넣어 재생성."""
    context = format_context(chunks)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", prompts.GENERATION_SYSTEM),
            ("user", prompts.REGENERATION_USER),
        ]
    )
    chain = prompt | get_llm(light=False) | StrOutputParser()
    return chain.invoke(
        {
            "context": context,
            "question": question,
            "unsupported": "; ".join(unsupported) or "(명시된 항목 없음)",
        }
    ).strip()
