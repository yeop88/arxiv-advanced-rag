"""
server/runtime.py — 서버 프로세스 하나가 공유하는 실행 상태

  - PipelineLoader: 서버가 켜지는 즉시 임베딩·리랭커·벡터DB·BM25 를 백그라운드로 올린다(rag/warmup.py).
  - LLM 서버 상태(/health)는 짧게 캐시한다 — 화면이 2초마다 묻기 때문.
  - 설정(config.settings)은 프로세스 전역 객체라, 요청마다 다른 옵션(질의 변환 방식·thinking 등)을
    잠금 안에서 적용하고 끝나면 되돌린다. 로컬 LLM 하나를 나눠 쓰므로 LLM 작업은 어차피 한 번에 하나씩이다.
"""
from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from typing import Any, Callable

from config import settings

Emit = Callable[..., None]


class NotReady(RuntimeError):
    """모델·DB 또는 LLM 서버를 쓸 수 없어 작업을 시작하지 못함(화면에 그대로 보여 줄 메시지)."""


class Runtime:
    def __init__(self, loader_factory: Callable[[], Any] | None = None,
                 llm_status_fn: Callable[[], tuple[str, str]] | None = None,
                 status_ttl: float = 1.5):
        from rag.warmup import PipelineLoader, llm_status

        self._loader_factory = loader_factory or PipelineLoader
        self._llm_status_fn = llm_status_fn or llm_status
        self._status_ttl = status_ttl
        self._status_cache: tuple[float, tuple[str, str]] | None = None
        self._status_lock = threading.Lock()
        self.loader = None
        self.lock = threading.Lock()   # LLM 을 쓰는 작업(질문·비교표·문헌조사·알림)은 한 번에 하나

    def start(self) -> None:
        if self.loader is None:
            self.loader = self._loader_factory()

    # --- 상태 ---
    def llm_status(self) -> tuple[str, str]:
        with self._status_lock:
            now = time.monotonic()
            if self._status_cache and now - self._status_cache[0] < self._status_ttl:
                return self._status_cache[1]
        st = self._llm_status_fn()
        with self._status_lock:
            self._status_cache = (time.monotonic(), st)
        return st

    def status(self) -> dict:
        ld = self.loader
        llm_state, llm_desc = self.llm_status()
        return {
            "loader": {
                "state": getattr(ld, "state", "loading"),
                "step": getattr(ld, "step", "시작 대기"),
                "done": [[n, round(s, 1)] for n, s in list(getattr(ld, "done", []))],
                "error": str(ld.error) if ld is not None and ld.error else None,
                "seconds": round(ld.since_start(), 1) if ld is not None else 0.0,
            },
            "llm": {"state": llm_state, "desc": llm_desc},
        }

    # --- 준비 대기 ---
    def wait_ready(self, emit: Emit, timeout: float = 1800, down_grace: float = 60, poll: float = 1.0):
        """모델·DB 와 LLM 서버가 준비될 때까지 기다리며 'waiting' 알림을 보낸다. 준비되면 파이프라인을 돌려준다."""
        self.start()
        t0 = time.monotonic()
        down_since = None
        announced = False
        while True:
            ld = self.loader
            if ld.state == "error":
                raise NotReady(f"모델·DB 준비 실패: {ld.error}")
            llm_state, llm_desc = self.llm_status()
            if ld.ready and llm_state in ("ok", "n/a"):
                if announced:
                    emit("ready", waited=round(time.monotonic() - t0, 1))
                return ld.pipe
            if llm_state == "down":
                down_since = down_since or time.monotonic()
                if time.monotonic() - down_since > down_grace:
                    raise NotReady(f"LLM 서버 {llm_desc}. FreeToken 이 실행 중인지, .env 의 LOCAL_BASE_URL(포트)을 확인하세요.")
            else:
                down_since = None
            waited = time.monotonic() - t0
            if waited > timeout:
                raise NotReady("준비 대기 시간을 넘었습니다.")
            emit("waiting", loader_state=ld.state, step=ld.step, done=[[n, round(s, 1)] for n, s in list(ld.done)],
                 llm_state=llm_state, llm_desc=llm_desc, waited=round(waited, 1))
            announced = True
            time.sleep(poll)

    # --- 잠금 + 요청별 설정 ---
    @contextmanager
    def session(self, emit: Emit, overrides: dict | None = None, poll: float = 0.5):
        """LLM 작업 잠금을 잡고(다른 작업 중이면 'queued' 알림을 보내며 기다림) 설정을 잠시 바꾼다."""
        if not self.lock.acquire(blocking=False):
            emit("queued")
            while not self.lock.acquire(timeout=poll):
                emit("queued")   # 화면이 닫히면 emit 이 Cancelled 를 던져 대기를 멈춘다
        saved = {}
        try:
            for k, v in (overrides or {}).items():
                if v is None:
                    continue
                if not hasattr(settings, k):
                    raise AttributeError(f"알 수 없는 설정: {k}")
                saved[k] = getattr(settings, k)
                setattr(settings, k, v)
            yield
        finally:
            for k, v in saved.items():
                setattr(settings, k, v)
            self.lock.release()
