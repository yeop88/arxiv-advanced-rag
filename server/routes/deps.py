"""라우터 공용 의존성."""
from __future__ import annotations

from fastapi import Request

from server.querylog import QueryLog
from server.runtime import Runtime


def get_runtime(request: Request) -> Runtime:
    return request.app.state.runtime


def get_query_log(request: Request) -> QueryLog | None:
    return request.app.state.query_log
