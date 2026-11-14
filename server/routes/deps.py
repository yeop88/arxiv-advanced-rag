"""라우터 공용 의존성."""
from __future__ import annotations

from fastapi import Request

from server.runtime import Runtime


def get_runtime(request: Request) -> Runtime:
    return request.app.state.runtime
