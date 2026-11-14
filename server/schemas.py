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
    return {"grounded": v.grounded, "faithfulness": v.faithfulness, "unsupported": v.unsupported_claims}


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
        "timings": {k: round(v, 2) for k, v in r.timings.items()},
    }


def table_json(t) -> dict:
    return {
        "markdown": t.to_markdown(marks=True),
        "verification": verification_json(t.verification),
        "cells_filled": t.cells_filled, "cells_total": t.cells_total,
        "chunks": [{"title": c.title, "section": c.section, "source": c.source} for c in t.chunks],
    }


def survey_json(s) -> dict:
    return {
        "topic": s.topic, "text": s.text, "references": s.references_md().splitlines(),
        "verification": verification_json(s.verification), "markdown": s.to_markdown(),
    }
