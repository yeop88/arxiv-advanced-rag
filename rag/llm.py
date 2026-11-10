"""
rag/llm.py — LLM / 임베딩 / 리랭커 팩토리

핵심 설계: "어떤 모델을 쓰는지"를 여기 한 곳에 가둬서,
각 단계 모듈은 provider(anthropic/openai)를 몰라도 되게 만듭니다.
→ 나중에 Claude ↔ OpenAI ↔ Ollama 교체가 이 파일 수정만으로 끝남.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from langchain_core.language_models.chat_models import BaseChatModel

from config import settings

_cache_initialized = False


def _init_llm_cache() -> None:
    """동일 (프롬프트, 모델) 응답을 SQLite 에 캐싱 → 반복 호출 비용·지연 제거.

    평가/디버깅처럼 같은 질문을 여러 번 돌릴 때 특히 효과적. 1회만 설정.
    """
    global _cache_initialized
    if _cache_initialized or not settings.enable_llm_cache:
        return
    from langchain_community.cache import SQLiteCache
    from langchain_core.globals import set_llm_cache

    Path(settings.llm_cache_path).parent.mkdir(parents=True, exist_ok=True)
    set_llm_cache(SQLiteCache(database_path=settings.llm_cache_path))
    _cache_initialized = True


def get_llm(light: bool = False) -> BaseChatModel:
    """생성/검증용 LLM 반환.

    Args:
        light: True 면 가벼운(저렴한) 모델 사용 — 질의변환처럼 쉬운 작업용.
    """
    _init_llm_cache()
    model = settings.llm_model_light if light else settings.llm_model

    if settings.llm_provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        # 주의: 최신 API 버전(2026-10-21~)은 temperature/top_p/top_k 를 거부(400).
        # 사실성은 프롬프트로 제어하고, temperature 는 넘기지 않는다.
        extra = {}
        if settings.llm_effort:
            # effort 는 요청 본문의 output_config 안에 들어간다(최상위 아님).
            extra["model_kwargs"] = {"output_config": {"effort": settings.llm_effort}}
        return ChatAnthropic(
            model=model,
            max_tokens=4096,
            max_retries=settings.llm_max_retries,  # 일시적 API 오류 자동 재시도
            **extra,
        )
    elif settings.llm_provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=model,
            temperature=settings.llm_temperature,
            max_retries=settings.llm_max_retries,
        )
    raise ValueError(f"알 수 없는 LLM_PROVIDER: {settings.llm_provider}")


@lru_cache(maxsize=1)
def get_embeddings():
    """HuggingFace 임베딩(bge-m3). 무거우므로 캐싱해서 1번만 로드."""
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name=settings.embed_model,
        model_kwargs={"device": "cpu"},  # GPU 있으면 "cuda"
        encode_kwargs={"normalize_embeddings": True},  # 코사인 유사도용 정규화
    )


@lru_cache(maxsize=1)
def get_reranker():
    """크로스인코더 리랭커(bge-reranker-v2-m3). 무거우므로 캐싱.

    CrossEncoder 는 (query, passage) 쌍을 함께 입력받아 관련도 점수를 매깁니다.
    벡터검색(bi-encoder)보다 느리지만 훨씬 정확 → 소수 후보 재정렬에 사용.
    """
    from sentence_transformers import CrossEncoder

    return CrossEncoder(settings.rerank_model, max_length=512)
