"""
rag/llm.py — LLM / 임베딩 / 리랭커 팩토리

핵심 설계: "어떤 모델을 쓰는지"를 여기 한 곳에 가둬서,
각 단계 모듈은 provider(anthropic/openai/local)를 몰라도 되게 만듭니다.
→ Claude ↔ OpenAI ↔ 로컬(OpenAI 호환 서버) 교체가 .env 설정만으로 끝남.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable, RunnableLambda

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


# provider 별 effort 허용값 — 이름도 값 체계도 다르므로 따로 검증한다.
ANTHROPIC_EFFORTS = {"low", "medium", "high", "xhigh", "max"}
OPENAI_EFFORTS = {"none", "minimal", "low", "medium", "high", "xhigh"}
LOCAL_THINKING = {"on", "off"}


def _choice(name: str, value: str, allowed: set[str]) -> str:
    """빈 값은 그대로(미전송), 그 외엔 허용값인지 확인해 오타를 일찍 잡는다."""
    v = (value or "").strip().lower()
    if v and v not in allowed:
        raise ValueError(f"{name}={value!r} 는 지원하지 않는 값입니다. 허용: {sorted(allowed)}")
    return v


def _strip_reasoning(msg: AIMessage) -> AIMessage:
    """로컬 thinking 모델(Qwen 등)이 본문에 섞어 내는 <think>...</think> 추론을 제거.

    서버에 reasoning parser 가 없으면 추론이 content 에 그대로 들어와, 검증 단계의 JSON
    파싱과 답변 본문을 오염시킨다. 여는 태그가 프롬프트 쪽에 있는 템플릿도 있어
    마지막 </think> 이후만 남긴다.
    """
    text = msg.content if isinstance(msg.content, str) else ""
    if "</think>" in text:
        return msg.model_copy(update={"content": text.split("</think>")[-1].strip()})
    return msg


def get_llm(light: bool = False, raw: bool = False) -> Runnable:
    """생성/검증용 LLM 반환. 모든 호출처는 `prompt | get_llm() | StrOutputParser()` 형태.

    Args:
        light: True 면 경량 모델·경량 effort 사용 — 질의변환처럼 쉬운 작업용.
        raw: True 면 후처리(<think> 제거) 래퍼 없이 모델 그대로 반환. 래퍼는 전체 응답을
             모아야 동작해 토큰 스트리밍을 막으므로, 스트리밍할 때 쓰고 제거는 호출부가 한다.
    """
    _init_llm_cache()
    model = settings.llm_model_light if light else settings.llm_model
    provider = settings.llm_provider

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        effort = _choice(
            "ANTHROPIC_EFFORT_LIGHT" if light else "ANTHROPIC_EFFORT",
            settings.anthropic_effort_light if light else settings.anthropic_effort,
            ANTHROPIC_EFFORTS,
        )
        extra = {}
        if effort:
            # effort 는 요청 본문의 output_config 안에 들어간다(최상위 아님).
            extra["model_kwargs"] = {"output_config": {"effort": effort}}
        # 주의: 최신 API 버전(2026-10-21~)은 temperature/top_p/top_k 를 거부(400) → 넘기지 않는다.
        return ChatAnthropic(
            model=model,
            max_tokens=settings.llm_max_tokens,
            max_retries=settings.llm_max_retries,  # 일시적 API 오류 자동 재시도
            **extra,
        )

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        effort = _choice(
            "OPENAI_REASONING_EFFORT_LIGHT" if light else "OPENAI_REASONING_EFFORT",
            settings.openai_reasoning_effort_light if light else settings.openai_reasoning_effort,
            OPENAI_EFFORTS,
        )
        extra = {"reasoning_effort": effort} if effort else {}
        return ChatOpenAI(
            model=model,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
            max_retries=settings.llm_max_retries,
            **extra,
        )

    if provider == "local":
        # OpenAI 호환 로컬 서버(예: FreeToken Local 의 Qwen3.6-35B-A3B).
        from langchain_openai import ChatOpenAI

        thinking = _choice(
            "LOCAL_THINKING_LIGHT" if light else "LOCAL_THINKING",
            settings.local_thinking_light if light else settings.local_thinking,
            LOCAL_THINKING,
        )
        extra = {}
        if thinking:
            extra["extra_body"] = {
                "chat_template_kwargs": {"enable_thinking": thinking == "on"}
            }
        llm = ChatOpenAI(
            model=model,
            base_url=settings.local_base_url,
            api_key=settings.local_api_key,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
            max_retries=settings.llm_max_retries,
            **extra,
        )
        return llm if raw else llm | RunnableLambda(_strip_reasoning)

    raise ValueError(f"알 수 없는 LLM_PROVIDER: {provider} (anthropic|openai|local)")


@lru_cache(maxsize=1)
def get_embeddings():
    """HuggingFace 임베딩(bge-m3). 무거우므로 캐싱해서 1번만 로드."""
    from langchain_huggingface import HuggingFaceEmbeddings

    device = settings.resolve_device(settings.embed_device)
    model_kwargs: dict = {"device": device}
    if device.startswith("cuda"):
        import torch

        # GPU 에선 fp16: 메모리 절반(bge-m3 약 1GB), 속도 수~수십 배. 저장 벡터(fp32)와 코사인 ≥0.9997.
        model_kwargs["model_kwargs"] = {"torch_dtype": torch.float16}
    return HuggingFaceEmbeddings(
        model_name=settings.embed_model,
        model_kwargs=model_kwargs,
        encode_kwargs={"normalize_embeddings": True},  # 코사인 유사도용 정규화
    )


@lru_cache(maxsize=1)
def get_reranker():
    """크로스인코더 리랭커(bge-reranker-v2-m3). 무거우므로 캐싱.

    CrossEncoder 는 (query, passage) 쌍을 함께 입력받아 관련도 점수를 매깁니다.
    벡터검색(bi-encoder)보다 느리지만 훨씬 정확 → 소수 후보 재정렬에 사용.
    """
    import torch
    from sentence_transformers import CrossEncoder

    device = settings.resolve_device(settings.rerank_device)
    if device.startswith("cuda") and settings.rerank_fp16:
        # GPU fp16: 메모리 절반, 20쌍 1.18s → 0.08s. 294문항에서 fp32 대비 1문항만 근소한 점수 뒤집힘
        # (recall@5 0.9711→0.9694). 로컬 LLM 과 GPU 를 나눠 쓸 때 느려지는 위험이 줄어든다.
        return CrossEncoder(settings.rerank_model, max_length=512, device=device,
                            model_kwargs={"torch_dtype": torch.float16})
    return CrossEncoder(settings.rerank_model, max_length=512, device=device)
