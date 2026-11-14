"""
server/app.py — FastAPI 앱

  /api/*  : JSON·SSE API (server/routes)
  기록    : 질문·답변·피드백을 storage/query_log.db 에(server/querylog.py, QUERY_LOG=0 이면 끔)
  /       : 빌드된 React 화면(frontend/dist). 개발 중에는 Vite 개발 서버(npm run dev)가 /api 를 이쪽으로 넘긴다.
서버가 켜지면 바로 모델·DB 예열을 시작한다(첫 질문이 기다리지 않게).
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from config import settings
from server.conversations import ConversationStore
from server.querylog import QueryLog
from server.routes import ROUTERS
from server.runtime import Runtime

FRONTEND_DIST = Path(__file__).resolve().parent.parent / "frontend" / "dist"

_NO_BUILD = """<!doctype html><meta charset="utf-8"><title>arXiv RAG</title>
<body style="font-family:sans-serif;padding:2rem">
<h2>화면이 아직 빌드되지 않았습니다</h2>
<p><code>cd frontend</code> → <code>npm install</code> → <code>npm run build</code> 후 새로고침하세요.
API 문서는 <a href="/docs">/docs</a> 에 있습니다.</p></body>"""


def create_app(runtime: Runtime | None = None, warmup: bool = True, frontend_dist: Path | None = FRONTEND_DIST,
               query_log: QueryLog | None = None, conversations: ConversationStore | None = None) -> FastAPI:
    rt = runtime or Runtime()
    if runtime is None and settings.visual_auto_enrich and settings.llm_provider == "local":
        from server.enricher import Enricher

        rt.enricher = Enricher(rt)
    if query_log is None and settings.query_log:
        query_log = QueryLog(settings.query_log_path)
    if conversations is None and settings.conversations:
        conversations = ConversationStore(settings.conversations_path)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if warmup:
            rt.start()
            if rt.enricher is not None:
                rt.enricher.start()   # 준비되면 밀린 자동 수집 논문부터, 사용자가 조용할 때만
        yield
        if rt.enricher is not None:
            rt.enricher.stop()

    app = FastAPI(title="arXiv Advanced RAG", lifespan=lifespan)
    app.state.runtime = rt
    app.state.query_log = query_log   # None 이면 기록하지 않음
    app.state.conversations = conversations
    for r in ROUTERS:
        app.include_router(r)

    if frontend_dist is not None and (frontend_dist / "index.html").exists():
        app.mount("/", StaticFiles(directory=frontend_dist, html=True), name="frontend")
    else:
        @app.get("/", include_in_schema=False)
        def no_build() -> HTMLResponse:
            return HTMLResponse(_NO_BUILD)

    return app


app = create_app()
