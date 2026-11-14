"""
server/streaming.py — 오래 걸리는 작업을 스레드에서 돌리며 진행 알림을 SSE(Server-Sent Events)로 흘려보낸다

작업 함수는 emit(event, **data) 만 부르면 된다. 알림은 생기는 즉시 한 건씩 전송된다(모아 보내지 않음).
  - 답변 토큰도 이 경로로 가므로, 화면은 글자가 생기는 대로 받는다.
  - 알림이 없는 긴 구간(검증 20초 등)에는 15초마다 주석 줄(ping)을 보내 연결을 유지한다.
  - 화면이 연결을 끊으면(중지 버튼·창 닫기) 다음 emit 에서 Cancelled 를 던져 작업을 멈춘다.
    Cancelled 는 BaseException 이라 파이프라인의 'except Exception' 안전장치에 잡히지 않고 끝까지 올라간다.
끝은 항상 'end' 이벤트(성공) 또는 'fail' 이벤트(오류 메시지)다.
"""
from __future__ import annotations

import asyncio
import json
import threading
from typing import Any, Callable

from fastapi.responses import StreamingResponse

from server.runtime import NotReady

PING_SECONDS = 15.0


class Cancelled(BaseException):
    """화면이 연결을 끊어 작업을 멈춤."""


def format_sse(event: str, data: Any) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False, default=str)}\n\n"


def sse_job(job: Callable[[Callable[..., None]], None]) -> StreamingResponse:
    """job(emit) 을 백그라운드 스레드에서 실행하고, emit 된 알림을 SSE 로 내보내는 응답."""
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()
    cancel = threading.Event()

    def put(item) -> None:
        try:
            loop.call_soon_threadsafe(queue.put_nowait, item)
        except RuntimeError:  # 서버 종료로 이벤트 루프가 닫힘
            cancel.set()

    def emit(event: str, **data) -> None:
        if cancel.is_set():
            raise Cancelled()
        put((event, data))

    def worker() -> None:
        try:
            job(emit)
            put(("end", {}))
        except Cancelled:
            pass
        except NotReady as e:
            put(("fail", {"message": str(e)}))
        except Exception as e:  # noqa: BLE001 — 화면에 오류를 알리고 연결을 닫는다
            put(("fail", {"message": f"{type(e).__name__}: {e}"}))
        finally:
            put(None)

    threading.Thread(target=worker, daemon=True, name="sse-job").start()

    async def stream():
        try:
            while True:
                try:
                    item = await asyncio.wait_for(queue.get(), timeout=PING_SECONDS)
                except asyncio.TimeoutError:
                    yield ": ping\n\n"
                    continue
                if item is None:
                    break
                yield format_sse(*item)
        finally:
            cancel.set()

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
