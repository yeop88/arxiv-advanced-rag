"""
server/app.py — FastAPI 앱

  /api/*  : JSON·SSE API (server/routes)
  /       : 빌드된 React 화면(frontend/dist). 개발 중에는 Vite 개발 서버(npm run dev)가 /api 를 이쪽으로 넘긴다.
서버가 켜지면 바로 모델·DB 예열을 시작한다(첫 질문이 기다리지 않게).
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from server.routes import ROUTERS
from server.runtime import Runtime

FRONTEND_DIST = Path(__file__).resolve().parent.parent / "frontend" / "dist"

_NO_BUILD = """<!doctype html><meta charset="utf-8"><title>arXiv RAG</title>
<body style="font-family:sans-serif;padding:2rem">
<h2>화면이 아직 빌드되지 않았습니다</h2>
<p><code>cd frontend</code> → <code>npm install</code> → <code>npm run build</code> 후 새로고침하세요.
API 문서는 <a href="/docs">/docs</a> 에 있습니다.</p></body>"""


def create_app(runtime: Runtime | None = None, warmup: bool = True, frontend_dist: Path | None = FRONTEND_DIST) -> FastAPI:
    rt = runtime or Runtime()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if warmup:
            rt.start()
        yield

    app = FastAPI(title="arXiv Advanced RAG", lifespan=lifespan)
    app.state.runtime = rt
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
