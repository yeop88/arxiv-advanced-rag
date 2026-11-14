"""관심 분야 알림: 주제 목록·최신 소식지 조회, 주제 추가·삭제, 새 논문 확인(SSE)."""
from __future__ import annotations

from fastapi import APIRouter, Depends

from server.routes.deps import get_runtime
from server.runtime import Runtime
from server.schemas import WatchTopicRequest
from server.streaming import sse_job

router = APIRouter(prefix="/api/watch")


def watch_state() -> dict:
    from rag.features import watch

    latest = watch.latest_digest()
    return {
        "topics": [{"topic": t["topic"], "query": t.get("query", ""), "last_checked": t.get("last_checked")}
                   for t in watch.load()["topics"]],
        "digest": {"date": latest[0], "markdown": latest[1]} if latest else None,
    }


@router.get("")
def get_watch() -> dict:
    return watch_state()


@router.post("/topics")
def add_topic(req: WatchTopicRequest) -> dict:
    """주제 추가 — 영어 검색어를 경량 LLM 으로 만든다(몇 초)."""
    from rag.features import watch

    watch.add_topic(req.topic.strip())
    return watch_state()


@router.delete("/topics/{topic}")
def remove_topic(topic: str) -> dict:
    from rag.features import watch

    watch.remove_topic(topic)
    return watch_state()


@router.post("/run")
async def run(rt: Runtime = Depends(get_runtime)):
    from rag.features import watch

    def job(emit) -> None:
        pipe = rt.wait_ready(emit)
        with rt.session(emit):
            watch.run(pipe.vectorstore, progress=emit)
        emit("result", **watch_state())

    return sse_job(job)
