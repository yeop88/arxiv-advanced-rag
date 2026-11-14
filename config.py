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
    # 실행 장치: "auto"(GPU 있으면 GPU) | "cuda" | "cpu". GPU 에선 fp16 으로 올린다.
    # 측정: 질의 임베딩 148→22ms, 문서 20개 14.2→0.3s, 리랭크 20쌍 1.18→0.08s, 검색 결과 사실상 동일.
    embed_device: str = os.getenv("EMBED_DEVICE", "auto")
    rerank_device: str = os.getenv("RERANK_DEVICE", "auto")
    # 리랭커 GPU fp16: VRAM 절반(FreeToken 과 GPU 공유 시 유리). 294문항 측정에서 fp32 대비
    # 1문항만 근소한 점수 뒤집힘(recall@5 0.9711→0.9694). 정확도 최우선이면 0 으로 끄기.
    rerank_fp16: bool = os.getenv("RERANK_FP16", "1") not in ("0", "false", "False")

    @staticmethod
    def resolve_device(value: str) -> str:
        if value != "auto":
            return value
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    # --- 벡터 스토어 ---
    # "chroma"(로컬 파일) | "pinecone"(원격). 전환은 scripts/migrate_to_pinecone.py 로 벡터를 그대로 복사한 뒤.
    vector_backend: str = os.getenv("VECTOR_BACKEND", "chroma")
    chroma_dir: str = os.getenv("CHROMA_DIR", "./storage/chroma")
    collection_name: str = "arxiv_papers"
    # Pinecone: 인덱스는 cosine(정규화 벡터라 Chroma 의 L2 와 순위가 같음), 차원은 임베딩과 동일(bge-m3=1024)
    pinecone_api_key: str = os.getenv("PINECONE_API_KEY", "")
    pinecone_index: str = os.getenv("PINECONE_INDEX", "arxiv-rag")
    pinecone_namespace: str = os.getenv("PINECONE_NAMESPACE", "arxiv")
    pinecone_cloud: str = os.getenv("PINECONE_CLOUD", "aws")
    pinecone_region: str = os.getenv("PINECONE_REGION", "us-east-1")
    # 원문 보관소: Pinecone 사용 시 BM25·본문 조회의 기준(원격 DB 에서 전체 문서를 꺼내기 어렵기 때문)
    docstore_path: str = os.getenv("DOCSTORE_PATH", "./storage/docstore.jsonl")
    # 사내 프록시 CA 묶음(PEM). 비우면 storage/ca_bundle.pem 이 있을 때 자동 사용
    # (scripts/export_ca_bundle.py 로 생성). 인증서 검증을 끄지 않고 프록시를 통과하기 위함.
    ca_bundle: str = os.getenv("CA_BUNDLE", "")

    def resolved_ca_bundle(self) -> str | None:
        if self.ca_bundle:
            return self.ca_bundle
        default = os.path.join("storage", "ca_bundle.pem")
        return default if os.path.exists(default) else None

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
    # BM25 단어 나누기: whitespace(이전) | regex(영문·한글 분리 + 조사 제거, 기본) | kiwi(한글 형태소 분석). 측정 §5.32
    bm25_tokenizer: str = os.getenv("BM25_TOKENIZER", "regex")
    rrf_k: int = 60            # RRF 상수(클수록 순위차 영향 완화, 통상 60)

    # --- 질의 변환 ---
    # 기본 multi_query: 확장 평가셋(111문항) 검색 A/B에서 recall 최고(hyde는 최하).
    # hyde→multi_query 재검토 경위는 docs/ADR-001-query-transform-default.md(§재검토) 참고.
    query_transform_mode: str = "multi_query"  # "multi_query" | "hyde" | "decompose" | "none"
    num_query_variations: int = 3              # multi_query 생성 개수

    # --- 검증 ---
    enable_verification: bool = True
    max_regenerations: int = 1  # 검증 실패 시 재생성 최대 횟수

    # --- 논문 자동 수집 (근거가 부족하면 arXiv 에서 찾아 전문을 DB 에 추가하고 다시 검색) ---
    auto_fetch: bool = os.getenv("AUTO_FETCH", "1") not in ("0", "false", "False")
    # 트리거: 리랭크 최고 점수가 이 값보다 낮으면 "코퍼스에 근거 부족"으로 판단 (평가셋 분포로 보정)
    auto_fetch_threshold: float = float(os.getenv("AUTO_FETCH_THRESHOLD", "0.2"))
    auto_fetch_queries: int = 2           # 만들 arXiv 검색어 수
    # 검색어당 후보 논문 수. arXiv 검색은 글자 일치 위주라 원 논문이 14~19위에 오는 경우가 많아 넉넉히 받는다
    auto_fetch_candidates: int = int(os.getenv("AUTO_FETCH_CANDIDATES", "20"))
    # 후보 논문의 (질문, 제목+초록) 리랭크 점수가 이 값 이상인 것만 추가 — 엉뚱한 논문이 쌓이는 것 방지
    auto_fetch_min_score: float = float(os.getenv("AUTO_FETCH_MIN_SCORE", "0.3"))
    auto_fetch_max_papers: int = int(os.getenv("AUTO_FETCH_MAX_PAPERS", "3"))
    auto_fetch_fulltext: bool = True      # PDF 전문(섹션 청킹). 실패한 논문만 초록으로 대체
    # 인용 수 순 검색을 후보에 더하고, 관련도가 이 값 이상인 후보 중 가장 많이 인용된 논문 1편은 반드시 선정(원 논문 우선)
    auto_fetch_citation_search: bool = os.getenv("AUTO_FETCH_CITATION_SEARCH", "1") not in ("0", "false", "False")
    auto_fetch_anchor_score: float = float(os.getenv("AUTO_FETCH_ANCHOR_SCORE", "0.5"))
    # 1 이면 경량 LLM 이 원 논문 제목을 추정해 제목으로도 찾고, 맞는 후보를 먼저 넣는다(A3, 측정 §5.25: 26%→62%, 확인용 58%→100%)
    auto_fetch_title_guess: bool = os.getenv("AUTO_FETCH_TITLE_GUESS", "1") not in ("0", "false", "False")
    # 1 이면 추정 제목과 맞는 원 논문을 찾았을 때 그 1편만 넣는다(파생 논문이 다른 질문에 끼어드는 것을 줄임, §5.29)
    auto_fetch_title_only: bool = os.getenv("AUTO_FETCH_TITLE_ONLY", "1") not in ("0", "false", "False")
    # 1 이면 근거 논문들이 공통으로 인용한 논문도 후보에 더한다(인용 그래프 필요)
    auto_fetch_common_refs: bool = os.getenv("AUTO_FETCH_COMMON_REFS", "0") not in ("0", "false", "False")
    # 자동 수집 청크 예산(0 = 제한 없음). 넘으면 오래 안 쓰인 논문부터 삭제.
    # 측정(§5.18): 2,000청크에서 MRR −0.0048, 3,000청크부터 −0.0088 로 허용치 0.01 에 근접
    auto_fetch_max_chunks: int = int(os.getenv("AUTO_FETCH_MAX_CHUNKS", "2000"))
    auto_usage_path: str = os.getenv("AUTO_USAGE_PATH", "./storage/auto_usage.json")  # 논문별 근거 사용 기록
    # OpenAlex 사용 여부(2026-11-12 사용자 결정으로 기본 끔). 끄면 자동 수집은 arXiv 만, 인용 그래프는 저장된 것만 쓴다
    openalex_enabled: bool = os.getenv("OPENALEX_ENABLED", "0") not in ("0", "false", "False")
    # OpenAlex API 키(선택). 없으면 같은 네트워크 IP 가 나눠 쓰는 하루 무료 예산(검색 약 100회)을 쓴다
    openalex_api_key: str = os.getenv("OPENALEX_API_KEY", "")
    # 그림 설명 청크도 만들지(기본 끔). 한국어 그림 설명이 무관한 한국어 질문의 근거에 끼어들어 recall −0.005(§5.35)
    visual_figures: bool = os.getenv("VISUAL_FIGURES", "0") not in ("0", "false", "False")
    # 표·그림 근거 청크(scripts/enrich_visuals.py)를 처리한 논문 기록
    visual_done_path: str = os.getenv("VISUAL_DONE_PATH", "./storage/visual_done.json")
    # 관심 분야 알림: 주제 목록·본 논문 기록 / 소식지 폴더 (scripts/watch_papers.py)
    watch_path: str = os.getenv("WATCH_PATH", "./storage/watchlist.json")
    watch_digest_dir: str = os.getenv("WATCH_DIGEST_DIR", "./storage/digests")
    # 인용 그래프(OpenAlex 참고문헌·피인용 관계). scripts/build_citation_graph.py 로 만든다
    citation_graph_path: str = os.getenv("CITATION_GRAPH_PATH", "./storage/citation_graph.json")
    # 수집 논문의 참고문헌·부록은 DB 에 넣지 않음(무관한 질문의 검색에 끼어드는 주된 원인)
    auto_fetch_drop_back_matter: bool = os.getenv("AUTO_FETCH_DROP_BACK_MATTER", "1") not in ("0", "false", "False")
    # 자동 수집 논문 청크의 리랭크 점수에 곱하는 값(1.0 = 그대로). 낮추면 기존 논문이 우선한다
    auto_rerank_weight: float = float(os.getenv("AUTO_RERANK_WEIGHT", "0.7"))
    # 하이브리드 검색에서 본문 후보(top_k_retrieve)와 별도로 더하는 표·그림 후보 수(0 = 표·그림 제외). §5.37
    visual_candidates: int = int(os.getenv("VISUAL_CANDIDATES", "1"))
    # 표·그림 청크가 근거에 들어가는 최소 리랭크 점수(0 = 제한 없음). §5.37
    visual_min_score: float = float(os.getenv("VISUAL_MIN_SCORE", "0.1"))
    # 자동 수집 논문(origin=auto)을 검색에서 숨김. 지우지 않고 '기준 코퍼스'만으로 평가할 때 쓴다
    # (평가 스크립트의 --corpus base 가 켠다. 챗봇은 꺼 둔 채로 자동 수집분까지 검색)
    exclude_auto_papers: bool = os.getenv("EXCLUDE_AUTO_PAPERS", "0") in ("1", "true", "True")

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
        if self.vector_backend == "pinecone" and not self.pinecone_api_key:
            raise RuntimeError("PINECONE_API_KEY 가 설정되지 않았습니다. .env 를 확인하세요.")


# 싱글턴처럼 import 해서 사용: from config import settings
settings = Settings()
