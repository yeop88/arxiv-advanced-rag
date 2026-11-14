"""
rag/stages/stage3_generate.py — 3단계: 생성 (인용 기반 답변)

리랭킹으로 고른 근거만 LLM 에 넣고, 각 문장에 [번호] 인용을 달아 답하게 합니다.
근거를 번호로 포맷해서 넣는 것이 환각 억제 + 출처 추적의 핵심.
"""
from __future__ import annotations

from typing import Callable

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from rag.core import prompts
from rag.core.llm import get_llm
from rag.core.schemas import RetrievedChunk


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


TokenCallback = Callable[[str], None]


def _visible(text: str) -> str:
    """스트리밍 중 화면에 보일 부분: 추론(<think>) 구간은 숨기고 최종 답만."""
    if "</think>" in text:
        return text.split("</think>")[-1].lstrip()
    if "<think>" in text:
        return ""  # 아직 추론 중
    return text


def _chunk_text(content) -> str:
    """메시지 조각의 글자(문자열 또는 블록 목록 — StrOutputParser 와 같은 처리)."""
    if isinstance(content, str):
        return content
    return "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content or [])


def _run(prompt: ChatPromptTemplate, inputs: dict, on_token: TokenCallback | None,
         on_reasoning: TokenCallback | None = None) -> str:
    """on_token 이 없으면 한 번에 받고, 있으면 토큰을 받는 대로 누적 텍스트를 넘긴다.
    on_reasoning 을 주면 thinking 모델의 추론 조각(서버가 따로 보내는 것)을 받는 대로 넘긴다."""
    if on_token is None:
        return (prompt | get_llm(light=False) | StrOutputParser()).invoke(inputs).strip()
    text = ""
    for msg in (prompt | get_llm(light=False, raw=True)).stream(inputs):
        rc = getattr(msg, "additional_kwargs", {}).get("reasoning_content")
        if rc and on_reasoning:
            on_reasoning(rc)
        piece = _chunk_text(getattr(msg, "content", msg))
        if piece:
            text += piece
            on_token(_visible(text))
    return _visible(text).strip()


def generate(question: str, chunks: list[RetrievedChunk], on_token: TokenCallback | None = None,
             on_reasoning: TokenCallback | None = None) -> str:
    """근거 기반 답변 생성. on_token 을 주면 생성되는 대로 누적 답변을 콜백으로 전달(화면 스트리밍용)."""
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", prompts.generation_system()),
            ("user", prompts.GENERATION_USER),
        ]
    )
    return _run(prompt, {"context": format_context(chunks), "question": question}, on_token, on_reasoning)


def regenerate(question: str, chunks: list[RetrievedChunk], unsupported: list[str],
               on_token: TokenCallback | None = None, on_reasoning: TokenCallback | None = None) -> str:
    """검증 실패 시, 문제점 피드백을 넣어 재생성."""
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", prompts.generation_system()),
            ("user", prompts.REGENERATION_USER),
        ]
    )
    return _run(prompt, {
        "context": format_context(chunks),
        "question": question,
        "unsupported": "; ".join(unsupported) or "(명시된 항목 없음)",
    }, on_token, on_reasoning)
