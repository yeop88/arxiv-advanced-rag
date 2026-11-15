"""
server/schemas.py — API 요청 형식과, 파이프라인 결과를 화면용 JSON 으로 바꾸는 함수
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from rag.core.schemas import PipelineResult, RetrievedChunk, VerificationResult

SNIPPET_CHARS = 300


# --- 요청 ---
class ChatOptions(BaseModel):
    """질문마다 바꿀 수 있는 설정(없으면 서버 기본값). 이름은 config.Settings 필드와 같다."""
    query_transform_mode: Literal["multi_query", "hyde", "decompose", "none"] | None = None
    use_hybrid: bool | None = None
    top_k_rerank: int | None = Field(default=None, ge=1, le=10)
    enable_verification: bool | None = None
    auto_fetch: bool | None = None
    local_thinking: Literal["on", "off"] | None = None
    local_thinking_light: Literal["on", "off"] | None = None


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    history: list[ChatTurn] = Field(default_factory=list)
    options: ChatOptions = Field(default_factory=ChatOptions)


class FeedbackRequest(BaseModel):
    log_id: str = Field(min_length=1, max_length=64)
    rating: Literal[1, -1, 0]
    note: str | None = Field(default=None, max_length=1000)


class LabelRequest(BaseModel):
    """사람 판정: 문장 번호(문자열) → supported/unsupported/miscited, 답변 전체 correct/partial/wrong."""
    sentences: dict[str, Literal["supported", "unsupported", "miscited"]] = Field(default_factory=dict)
    overall: Literal["correct", "partial", "wrong"] | None = None
    note: str | None = Field(default=None, max_length=2000)
    annotator: str = Field(default="human", max_length=40)   # 판정한 주체(사람이면 human, 다른 모델이면 그 이름)


class ConversationPut(BaseModel):
    """대화 전체(화면의 메시지 목록 그대로). id 는 주소에서."""
    title: str | None = Field(default=None, max_length=200)
    messages: list[dict] = Field(default_factory=list, max_length=2000)


class CompareRequest(BaseModel):
    sources: list[str] = Field(min_length=2)


class SurveyRequest(BaseModel):
    topic: str = Field(min_length=1, max_length=300)


class WatchTopicRequest(BaseModel):
    topic: str = Field(min_length=1, max_length=200)


# --- 응답 ---
def chunk_json(c: RetrievedChunk, full: bool = False) -> dict:
    text = c.content.strip()
    return {
        "source": c.source, "title": c.title, "section": c.section, "link": c.link, "kind": c.kind,
        "origin": c.origin, "rerank_score": c.rerank_score,
        "snippet": text if full else text[:SNIPPET_CHARS] + ("..." if len(text) > SNIPPET_CHARS else ""),
    }


def verification_json(v: VerificationResult | None) -> dict | None:
    if v is None:
        return None
    return {"grounded": v.grounded, "faithfulness": v.faithfulness, "unsupported": v.unsupported_claims,
            "completeness": v.completeness, "missed": v.missed}


def result_json(r: PipelineResult) -> dict:
    return {
        "question": r.question,
        "standalone_question": r.standalone_question,
        "transformed_queries": r.transformed_queries,
        "answer": r.answer,
        "sources": [chunk_json(c) for c in r.reranked],
        "n_candidates": len(r.retrieved),
        "verification": verification_json(r.verification),
        "regenerated": r.regenerated,
        "auto_fetched": r.auto_fetched,
        "citation_check": r.citation_check,
        "timings": {k: round(v, 2) for k, v in r.timings.items()},
    }


def table_json(t) -> dict:
    from rag.stages.citation_check import check_table_numbers

    return {
        "markdown": t.to_markdown(marks=True),
        "number_check": check_table_numbers(t.to_markdown(), t.chunks),
        "verification": verification_json(t.verification),
        "cells_filled": t.cells_filled, "cells_total": t.cells_total,
        "chunks": [{"title": c.title, "section": c.section, "source": c.source} for c in t.chunks],
    }


def survey_json(s) -> dict:
    return {
        "topic": s.topic, "text": s.text, "references": s.references_md().splitlines(),
        "verification": verification_json(s.verification), "markdown": s.to_markdown(),
    }
