"""
rag/warmup.py — 챗봇 시작 시 무거운 구성요소를 백그라운드로 미리 올리는 로더 + 상태 관리

왜 필요한가?
  임베딩 모델·리랭커·벡터 DB·BM25 색인은 처음 쓸 때 올리면 첫 질문이 10초 이상 늦어진다.
  앱이 켜지는 순간 별도 스레드에서 미리 올려 두고, 그 사이 들어온 질문은 준비될 때까지 기다리게 한다.
  로컬 LLM 서버(FreeToken)는 모델을 다시 올리는 동안 503 을 돌려주므로, 그 상태도 함께 확인한다.

상태: loading → ready | error   (현재 단계 이름·완료 단계·경과 시간 제공)
"""
from __future__ import annotations

import threading
import time
from urllib.parse import urlparse

from config import settings


class PipelineLoader:
    def __init__(self, start: bool = True):
        self.state = "loading"
        self.step = "시작 대기"
        self.done: list[tuple[str, float]] = []   # (단계 이름, 걸린 초)
        self.error: Exception | None = None
        self.pipe = None
        self.t0 = time.perf_counter()
        self.elapsed: float | None = None
        self._ready = threading.Event()
        if start:
            threading.Thread(target=self._load, daemon=True, name="rag-warmup").start()

    # --- 로딩 ---
    def _step(self, name: str, fn):
        self.step = name
        t = time.perf_counter()
        out = fn()
        self.done.append((name, time.perf_counter() - t))
        return out

    def _load(self) -> None:
        from rag import stage2_rerank as s2
        from rag.llm import get_embeddings, get_reranker
        from rag.pipeline import RAGPipeline

        try:
            self._step("임베딩 모델(bge-m3) 로딩", lambda: get_embeddings().embed_query("warmup"))
            self._step("리랭커(bge-reranker) 로딩", lambda: get_reranker().predict([("warmup", "warmup")]))
            pipe = self._step(f"벡터 DB 연결({settings.vector_backend})", RAGPipeline)
            self._step("BM25 키워드 색인 구축", lambda: s2.warm_bm25(pipe.vectorstore))
            self.pipe = pipe
            self.state = "ready"
            self.step = "준비 완료"
        except Exception as e:  # noqa: BLE001
            self.error = e
            self.state = "error"
            self.step = f"오류: {e}"
        finally:
            self.elapsed = time.perf_counter() - self.t0
            self._ready.set()

    # --- 조회 ---
    @property
    def ready(self) -> bool:
        return self.state == "ready"

    def wait(self, timeout: float | None = None) -> bool:
        """준비(또는 오류)될 때까지 최대 timeout 초 대기. 끝났으면 True."""
        return self._ready.wait(timeout)

    def since_start(self) -> float:
        return self.elapsed if self.elapsed is not None else time.perf_counter() - self.t0


def llm_status(timeout: float = 3.0) -> tuple[str, str]:
    """LLM 백엔드 준비 상태 → (state, 설명). state: ok | loading | down | n/a

    로컬 서버(FreeToken 등)의 /health 를 본다(인증 불필요 경로). 원격 API(Claude·OpenAI)는 확인하지 않는다.
    """
    if settings.llm_provider != "local" or not settings.local_base_url:
        return "n/a", f"{settings.llm_provider} API"
    import json
    import urllib.request

    u = urlparse(settings.local_base_url)
    url = f"{u.scheme}://{u.netloc}/health"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:  # noqa: S310 — 로컬 주소
            data = json.loads(r.read().decode("utf-8") or "{}")
    except Exception as e:  # noqa: BLE001
        return "down", f"응답 없음({type(e).__name__})"
    status = str(data.get("status", "unknown"))
    model = data.get("model", "")
    if status == "ok":
        mismatch = model and model not in (settings.llm_model, settings.llm_model_light)
        return "ok", f"{model} 준비됨" + (f" — ⚠️ .env 의 LLM_MODEL({settings.llm_model})과 이름이 다름" if mismatch else "")
    if status == "loading":
        prog = data.get("progress") or ""
        return "loading", f"{model} 모델 올리는 중 {prog}".strip()
    return "down", f"상태 {status}"
