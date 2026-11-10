"""
config.py — 프로젝트 전역 설정 (한 곳에서 관리)

모든 하이퍼파라미터·경로·모델명을 여기 모았습니다.
각 단계 모듈은 이 Settings 객체만 보면 되므로, 실험(모델 교체·청크 크기 변경)이
코드 수정 없이 .env 와 이 파일만으로 끝납니다.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()  # .env 로드


@dataclass
class Settings:
    # --- LLM 백엔드 ---
    # "anthropic" | "openai" | "local"(OpenAI 호환 로컬 서버 — 예: FreeToken Local 의 Qwen)
    llm_provider: str = os.getenv("LLM_PROVIDER", "anthropic")
    llm_model: str = os.getenv("LLM_MODEL", "claude-opus-5-5")        # 생성·검증(메인)
    llm_model_light: str = os.getenv("LLM_MODEL_LIGHT", "claude-haiku-5-5")  # 질의변환(경량)
    # temperature 는 openai/local 전용(Claude 최신 API 는 temperature 를 400 으로 거부).
    llm_temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.0"))
    # thinking 모델은 추론 토큰까지 포함하므로 부족하면 답이 잘린다 → 로컬 thinking 사용 시 상향 권장.
    llm_max_tokens: int = int(os.getenv("LLM_MAX_TOKENS", "4096"))

    # --- 로컬(OpenAI 호환) 서버 접속 ---
    local_base_url: str = os.getenv("LOCAL_BASE_URL", "")    # 예: http://localhost:8000/v1
    local_api_key: str = os.getenv("LOCAL_API_KEY", "EMPTY")  # 인증 없는 서버는 아무 값이나

    # --- Effort: provider 마다 파라미터와 값 체계가 달라 분리, 각각 메인/경량 따로 ---
    # 빈 값("")이면 아무것도 보내지 않고 서버/모델 기본값을 쓴다.
    # Anthropic: output_config.effort = low|medium|high|xhigh|max (opus/haiku 5.x 기본 medium)
    anthropic_effort: str = os.getenv("ANTHROPIC_EFFORT", "")
    anthropic_effort_light: str = os.getenv("ANTHROPIC_EFFORT_LIGHT", "")
    # OpenAI: reasoning_effort = none|minimal|low|medium|high|xhigh (추론 모델에서만 의미, 지원값은 모델별 상이)
    openai_reasoning_effort: str = os.getenv("OPENAI_REASONING_EFFORT", "")
    openai_reasoning_effort_light: str = os.getenv("OPENAI_REASONING_EFFORT_LIGHT", "")
    # Local(Qwen3 계열): 등급형 effort 가 아니라 thinking on|off
    # → chat_template_kwargs.enable_thinking 로 전달(vLLM·SGLang 방식, 서버가 무시할 수도 있음)
    local_thinking: str = os.getenv("LOCAL_THINKING", "")
    local_thinking_light: str = os.getenv("LOCAL_THINKING_LIGHT", "")

    # --- 임베딩 / 리랭커 (HuggingFace 로컬) ---
    embed_model: str = os.getenv("EMBED_MODEL", "BAAI/bge-m3")
    rerank_model: str = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")

    # --- 벡터 스토어 ---
    chroma_dir: str = os.getenv("CHROMA_DIR", "./storage/chroma")
    collection_name: str = "arxiv_papers"

    # --- 청킹 ---
    chunk_size: int = 1000
    chunk_overlap: int = 150

    # --- 검색 파이프라인 파라미터 ---
    # 1차 벡터검색으로 넉넉히 뽑고(top_k_retrieve) → 리랭킹으로 추림(top_k_rerank)
    top_k_retrieve: int = 20   # 리랭킹 전 후보 수
    top_k_rerank: int = 5      # 최종 생성에 넣을 근거 수

    # --- 다양성 리랭킹 (MMR) ---
    # 전문 코퍼스에선 한 논문이 여러 청크로 쪼개져 상위 top_k 를 독점 → 다중 홉 질문에서
    # 다른 논문 근거가 밀려남. MMR 로 관련도와 다양성을 함께 최적화하고, 같은 논문 청크는
    # 중복으로 간주해 벌점을 줘 '논문 커버리지'를 높인다.
    rerank_diversity: bool = True   # False 면 순수 cross-encoder 점수 상위만
    mmr_lambda: float = 0.6         # 관련도 가중(1=관련도만, 0=다양성만). 0.5~0.7 권장
    same_source_penalty: float = 0.9  # 같은 논문 청크 간 유사도 하한(↑일수록 같은 논문 억제)
    rerank_max_per_source: int = 2  # 최종 top_k 에 같은 논문 청크 최대 개수(0=무제한).
                                    # MMR의 관련도 격차를 못 이기는 쏠림을 '보장'으로 막는다

    # --- 하이브리드 검색 (BM25 키워드 + 벡터 의미) ---
    # True 면 BM25(정확 매칭)와 벡터(의미)를 RRF 로 융합 → 고유명사·약어에 강해짐
    use_hybrid: bool = True
    rrf_k: int = 60            # RRF 상수(클수록 순위차 영향 완화, 통상 60)

    # --- 질의 변환 ---
    # 기본 multi_query: 확장 평가셋(111문항) 검색 A/B에서 recall 최고(hyde는 최하).
    # hyde→multi_query 재검토 경위는 docs/ADR-001-query-transform-default.md(§재검토) 참고.
    query_transform_mode: str = "multi_query"  # "multi_query" | "hyde" | "decompose" | "none"
    num_query_variations: int = 3              # multi_query 생성 개수

    # --- 검증 ---
    enable_verification: bool = True
    max_regenerations: int = 1  # 검증 실패 시 재생성 최대 횟수

    # --- 품질 하드닝 ---
    enable_llm_cache: bool = True          # 동일 프롬프트 응답 캐싱(SQLite) → 비용·속도 절감
    llm_cache_path: str = "./storage/llm_cache.db"
    llm_max_retries: int = 2               # API 일시 오류 시 자동 재시도 횟수

    def validate(self) -> None:
        """필수 API 키가 설정됐는지 확인."""
        if self.llm_provider == "anthropic" and not os.getenv("ANTHROPIC_API_KEY"):
            raise RuntimeError("ANTHROPIC_API_KEY 가 설정되지 않았습니다. .env 를 확인하세요.")
        if self.llm_provider == "openai" and not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY 가 설정되지 않았습니다. .env 를 확인하세요.")
        if self.llm_provider == "local" and not self.local_base_url:
            raise RuntimeError("LOCAL_BASE_URL 이 설정되지 않았습니다. 예: http://localhost:8000/v1")


# 싱글턴처럼 import 해서 사용: from config import settings
settings = Settings()
