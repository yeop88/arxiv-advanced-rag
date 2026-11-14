"""질문 → 4단계 파이프라인 → 진행 알림·답변 토큰·최종 결과를 SSE 로.

이벤트 순서(예): waiting* → queued* → start → transform_start … generate_start → token* → generate_done
                 → verify_start → verify_done → result → end      (실패 시 fail)
token 은 새로 생긴 글자(delta)만 담는다.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends

from config import settings
from server.routes.deps import get_runtime
from server.runtime import Runtime
from server.schemas import ChatRequest, result_json
from server.streaming import sse_job

router = APIRouter(prefix="/api")


def thinking_label() -> str:
    """답변에 붙일 thinking 모드 표시(로컬 모델일 때만)."""
    if settings.llm_provider != "local":
        return ""
    return (f"🧠 thinking {settings.local_thinking or '서버 기본값'} "
            f"(질의변환 {settings.local_thinking_light or '기본값'})")


@router.post("/chat")
async def chat(req: ChatRequest, rt: Runtime = Depends(get_runtime)):
    history = [(t.role, t.content) for t in req.history]
    overrides = req.options.model_dump()
    if settings.llm_provider != "local":   # thinking 토글은 로컬 모델 전용
        overrides.pop("local_thinking")
        overrides.pop("local_thinking_light")

    def job(emit) -> None:
        pipe = rt.wait_ready(emit)
        with rt.session(emit, overrides):
            emit("start", mode=thinking_label())
            result = pipe.run(req.question, chat_history=history or None, progress=emit)
        emit("result", **result_json(result))

    return sse_job(job)
