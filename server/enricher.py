"""
server/enricher.py — 자동 수집 논문의 표·그림·수식을 백그라운드에서 근거로 추가

질문 중 자동 수집된 전문 논문(과 서버 시작 때 아직 처리 안 한 전문 논문)을 큐에 넣고, 사용자가 쓰지 않을 때만 처리한다.
  - 마지막 사용자 작업 뒤 idle 초가 지나고, 잠금을 기다리는 질문이 없을 때만 다음 페이지를 시작
  - LLM 작업 잠금은 페이지(모델 호출) 하나마다 잡고 놓는다 → 그사이 온 질문은 길어야 한 페이지(약 30~50초)만 기다림
  - 다 끝나면 잠금 안에서 청크를 추가하고 BM25 를 다시 만든다. 처리 기록에 남겨 정리 대상에서 빠진다(§5.39 이슈 22)
LLM: 캡션이 있는 페이지당 로컬 멀티모달 모델 1회. VISUAL_AUTO_ENRICH=0 이면 끔.
"""
from __future__ import annotations

import queue
import threading
import time
from contextlib import contextmanager

from config import settings


class Enricher:
    def __init__(self, rt, idle: float | None = None, poll: float = 2.0, scan_on_start: bool = True):
        self.rt = rt
        self.scan_on_start, self._scanned = scan_on_start, False
        self.idle = settings.visual_auto_idle if idle is None else idle
        self.poll = poll
        self.q: queue.Queue[tuple[str, dict]] = queue.Queue()
        self.queued: set[str] = set()
        self.current: dict | None = None     # {"source", "title", "page", "pages"}
        self.done: list[dict] = []           # 최근 처리 결과
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    # --- 큐 ---
    def enqueue(self, source: str, meta: dict | None = None) -> bool:
        """처리할 논문을 넣는다. meta 가 없으면 처리할 때 DB 에서 찾는다(전문 논문이 아니면 건너뜀)."""
        if source in self.queued:
            return False
        self.queued.add(source)
        self.q.put((source, meta or {}))
        self.start()
        return True

    def enqueue_pending(self, vs) -> int:
        """서버 시작 때: 전문이 있는 자동 수집 논문 중 아직 처리 안 한 것."""
        from rag.ingest import enrich, visual

        done = enrich.load_done()
        kinds = set(visual.default_kinds())
        n = 0
        for src, p in enrich.fulltext_papers(vs).items():
            if p["meta"].get("origin") == "auto" and kinds - enrich.done_kinds(done.get(src)):
                n += self.enqueue(src, p["meta"])
        return n

    def status(self) -> dict:
        return {"enabled": settings.visual_auto_enrich, "queue": self.q.qsize(), "current": self.current,
                "recent": self.done[-3:]}

    # --- 실행 ---
    def start(self) -> None:
        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(target=self._run, name="enricher", daemon=True)
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _quiet(self) -> bool:
        return self.rt.waiting == 0 and time.monotonic() - self.rt.last_active >= self.idle

    @contextmanager
    def _turn(self):
        """사용자가 조용할 때까지 기다렸다가 LLM 작업 잠금을 잡는다(페이지 하나 동안)."""
        got = False
        while not self._stop.is_set():
            if self._quiet() and self.rt.lock.acquire(blocking=False):
                got = True
                break
            time.sleep(self.poll)
        if not got:   # 서버 종료 — 남의 잠금을 풀지 않게 여기서 멈춘다
            raise RuntimeError("백그라운드 처리 중지")
        self.rt.background = "enrich"
        try:
            yield
        finally:
            self.rt.background = None
            self.rt.lock.release()

    def _pipe(self):
        return getattr(getattr(self.rt, "loader", None), "pipe", None)

    def _run(self) -> None:
        from rag.ingest import enrich, visual
        from rag.stages import stage2_rerank as s2

        if self.scan_on_start and not self._scanned:   # 모델·DB 가 준비되면 밀린 논문부터
            while self._pipe() is None and not self._stop.is_set():
                time.sleep(self.poll)
            self._scanned = True
            if self._pipe() is not None:
                self.enqueue_pending(self._pipe().vectorstore)
        while not self._stop.is_set():
            try:
                src, meta = self.q.get(timeout=self.poll)
            except queue.Empty:
                continue
            pipe = self._pipe()
            if pipe is None:            # 아직 모델·DB 준비 전 — 다시 넣고 기다림
                self.q.put((src, meta))
                time.sleep(self.poll)
                continue
            vs = pipe.vectorstore
            if not meta.get("source"):
                paper = enrich.fulltext_papers(vs).get(src)
                if paper is None:       # 초록만 있는 논문 — 표·그림·수식을 볼 PDF 페이지 정보가 없다
                    self.queued.discard(src)
                    continue
                meta = paper["meta"]
            kinds = tuple(k for k in visual.default_kinds() if k not in enrich.done_kinds(enrich.load_done().get(src)))
            if not kinds:
                self.queued.discard(src)
                continue
            self.current = {"source": src, "title": meta.get("title", ""), "page": 0, "pages": 0}
            t0 = time.perf_counter()

            def progress(ev, **kw):
                self.current.update(page=kw.get("i", 0), pages=kw.get("n", 0))

            def add(docs):
                with self._turn():       # 검색과 겹치지 않게 잠금 안에서 추가 + BM25 다시
                    vs.add_documents(docs)
                    s2.reset_bm25()

            try:
                items = enrich.enrich(src, meta, vs, kinds, progress=progress, guard=self._turn, add=add)
                enrich.mark_done(src, kinds, items)
                self.done.append({"source": src, "title": meta.get("title", ""), "added": len(items),
                                  "sec": round(time.perf_counter() - t0)})
            except Exception as e:  # noqa: BLE001 — 한 편 실패가 서버를 멈추지 않게
                print(f"[경고] 표·그림·수식 처리 실패 {src}: {type(e).__name__}: {str(e)[:150]}")
                self.done.append({"source": src, "title": meta.get("title", ""), "error": str(e)[:150]})
            finally:
                self.current = None
                self.queued.discard(src)
