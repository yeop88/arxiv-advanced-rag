"""상태·기본 설정 조회(LLM 호출 없음)."""
from __future__ import annotations

from fastapi import APIRouter, Depends

from config import settings
from server.routes.deps import get_runtime
from server.runtime import Runtime

router = APIRouter(prefix="/api")


@router.get("/status")
def status(rt: Runtime = Depends(get_runtime)) -> dict:
    """모델·DB 로딩 단계와 LLM 서버 상태. 화면이 2초마다 묻는다."""
    return rt.status()


@router.get("/config")
def config() -> dict:
    """화면 설정 패널의 기본값과 사용 중인 모델 정보."""
    return {
        "provider": settings.llm_provider,
        "model": settings.llm_model,
        "embed_model": settings.embed_model,
        "vector_backend": settings.vector_backend,
        "openalex_enabled": settings.openalex_enabled,
        "defaults": {
            "query_transform_mode": settings.query_transform_mode,
            "use_hybrid": settings.use_hybrid,
            "top_k_rerank": settings.top_k_rerank,
            "enable_verification": settings.enable_verification,
            "auto_fetch": settings.auto_fetch,
            "local_thinking": settings.local_thinking or "off",
            "local_thinking_light": settings.local_thinking_light or "off",
        },
    }
