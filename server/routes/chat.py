"""질문 → 4단계 파이프라인 → 진행 알림·답변 토큰·최종 결과를 SSE 로.

이벤트 순서(예): waiting* → queued* → start → transform_start … generate_start → token* → generate_done
                 → verify_start → verify_done → result → end      (실패 시 fail)
token 은 새로 생긴 글자(delta)만 담는다. result 에는 기록 id(log_id)가 붙어, 화면이 👍/👎 를 보낼 때 쓴다.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from config import settings
from server.querylog import QueryLog
from server.routes.deps import get_query_log, get_runtime
from server.runtime import Runtime
from server.schemas import ChatRequest, FeedbackRequest, result_json
from server.streaming import sse_job

router = APIRouter(prefix="/api")


def thinking_label() -> str:
    """답변에 붙일 thinking 모드 표시(로컬 모델일 때만)."""
    if settings.llm_provider != "local":
        return ""
    return (f"🧠 thinking {settings.local_thinking or '서버 기본값'} "
            f"(질의변환 {settings.local_thinking_light or '기본값'})")


@router.post("/chat")
async def chat(req: ChatRequest, rt: Runtime = Depends(get_runtime), qlog: QueryLog | None = Depends(get_query_log)):
    history = [(t.role, t.content) for t in req.history]
    overrides = req.options.model_dump()
    if settings.llm_provider != "local":   # thinking 토글은 로컬 모델 전용
        overrides.pop("local_thinking")
        overrides.pop("local_thinking_light")

    def record(**kw) -> str | None:
        if qlog is None:
            return None
        try:
            return qlog.record(req.question, options=req.options.model_dump(exclude_none=True),
                               history_turns=len(history), **kw)
        except Exception as e:  # noqa: BLE001 — 기록 실패가 답변을 막지 않게
            print(f"[경고] 질문 기록 실패: {e}")
            return None

    def job(emit) -> None:
        try:
            pipe = rt.wait_ready(emit)
            with rt.session(emit, overrides):
                emit("start", mode=thinking_label())
                result = pipe.run(req.question, chat_history=history or None, progress=emit)
        except Exception as e:  # 실패도 기록(중지·연결 끊김은 BaseException 이라 기록하지 않음)
            record(error=str(e) or type(e).__name__)
            raise
        data = result_json(result)
        emit("result", **data, log_id=record(result=data))
        if rt.enricher is not None:   # 새로 넣은 전문 논문의 표·그림·수식은 한가할 때 처리
            for a in result.auto_fetched:
                if a.get("fulltext"):
                    rt.enricher.enqueue(a["source"])

    return sse_job(job)


@router.post("/feedback")
def feedback(req: FeedbackRequest, qlog: QueryLog | None = Depends(get_query_log)) -> dict:
    """답변에 👍(1)/👎(-1)/취소(0)와 이유(선택)를 붙인다."""
    if qlog is None:
        raise HTTPException(404, "질문 기록이 꺼져 있습니다(QUERY_LOG=0).")
    if not qlog.rate(req.log_id, req.rating, req.note):
        raise HTTPException(404, "해당 기록이 없습니다.")
    return {"ok": True}


@router.get("/metrics")
def metrics(days: int = 14, qlog: QueryLog | None = Depends(get_query_log)) -> dict:
    """사용 지표(질문 수·피드백·단계별 시간·인용 확인 추세). 질문 기록에서 계산, LLM 없음."""
    from server.querylog import metrics as compute

    if qlog is None:
        return {"enabled": False}
    return {"enabled": True, **compute(qlog.list(limit=2000), days=max(1, min(days, 90)))}


@router.get("/logs")
def logs(rating: int | None = None, limit: int = 50, qlog: QueryLog | None = Depends(get_query_log)) -> dict:
    """최근 질문 기록(평가셋 후보 검토용). rating=-1 이면 👎 받은 것만."""
    if qlog is None:
        return {"enabled": False, "stats": None, "items": []}
    return {"enabled": True, "stats": qlog.stats(), "items": qlog.list(rating=rating, limit=min(limit, 500))}
