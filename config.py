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


def _postgres_url() -> str:
    """항목별 PostgreSQL 로그인 정보(.env 의 POSTGRES_*) → 접속 URL. 사용자·DB 이름이 없으면 빈 문자열(저장소 끔)."""
    from urllib.parse import quote

    user, db = os.getenv("POSTGRES_USER", ""), os.getenv("POSTGRES_DB", "")
    if not (user and db):
        return ""
    pw = os.getenv("POSTGRES_PASSWORD", "")
    auth = quote(user, safe="") + (":" + quote(pw, safe="") if pw else "")
    host, port = os.getenv("POSTGRES_HOST", "localhost"), os.getenv("POSTGRES_PORT", "5432")
    return f"postgresql://{auth}@{host}:{port}/{quote(db, safe='')}"


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
    # 기본 끔(2026-11-15 사용자 결정): 켬은 결과 이득 없이 8배 느렸다(§5.51). 웹 화면은 요청마다 켬/끔을 보낸다
    local_thinking: str = os.getenv("LOCAL_THINKING", "off")
    local_thinking_light: str = os.getenv("LOCAL_THINKING_LIGHT", "off")

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
    # "chroma"(로컬 파일) | "pinecone"(원격) | "pgvector"(PostgreSQL — DATABASE_URL 사용).
    # 전환은 scripts/migrate_vectors.py --to pinecone|pgvector 로 벡터를 그대로 복사한 뒤.
    vector_backend: str = os.getenv("VECTOR_BACKEND", "chroma")
    chroma_dir: str = os.getenv("CHROMA_DIR", "./storage/chroma")
    collection_name: str = "arxiv_papers"
    pgvector_table: str = os.getenv("PGVECTOR_TABLE", "rag_chunks")  # pgvector 백엔드의 청크 테이블
    # Pinecone: 인덱스는 cosine(정규화 벡터라 Chroma 의 L2 와 순위가 같음), 차원은 임베딩과 동일(bge-m3=1024)
    pinecone_api_key: str = os.getenv("PINECONE_API_KEY", "")
    pinecone_index: str = os.getenv("PINECONE_INDEX", "arxiv-rag")
    pinecone_namespace: str = os.getenv("PINECONE_NAMESPACE", "arxiv")
    pinecone_cloud: str = os.getenv("PINECONE_CLOUD", "aws")
    pinecone_region: str = os.getenv("PINECONE_REGION", "us-east-1")

    # --- 논문 원문 저장소 (PostgreSQL, rag/store/paperstore.py) ---
    # 수집한 논문의 PDF 원문과 추출한 전문 텍스트를 보관(청크만 있으면 재청킹·재처리 때 다시 받아야 해서).
    # 비어 있으면 저장하지 않는다(수집은 그대로 동작). 예: postgresql://user:pw@localhost:5432/arxiv_rag
    # 접속 정보는 항목별(POSTGRES_HOST·PORT·DB·USER·PASSWORD)로 넣거나 DATABASE_URL 한 줄로 넣는다(둘 다 있으면 URL 우선).
    postgres_host: str = os.getenv("POSTGRES_HOST", "localhost")
    postgres_port: str = os.getenv("POSTGRES_PORT", "5432")
    postgres_db: str = os.getenv("POSTGRES_DB", "")
    postgres_user: str = os.getenv("POSTGRES_USER", "")
    postgres_password: str = os.getenv("POSTGRES_PASSWORD", "")
    database_url: str = field(default_factory=lambda: os.getenv("DATABASE_URL", "") or _postgres_url())
    paper_store_pdf: bool = os.getenv("PAPER_STORE_PDF", "1") not in ("0", "false", "False")  # 0 이면 전문 텍스트만
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
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "1000"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "150"))
    chunk_unit: str = os.getenv("CHUNK_UNIT", "chars")   # chars(글자 수) | tokens(임베딩 토크나이저 토큰 수)

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

    # 리랭커 입력에 논문 제목 붙이기(저장 글·LLM 근거는 그대로). 294문항: MRR 0.9545→0.9700, 다중 홉 recall 0.8955→0.9055,
    # recall 0.9711→0.9700, 코퍼스 안 질문의 수집 신호 4.1%→3.4%(§5.42). 0 이면 끔
    rerank_title_prefix: bool = os.getenv("RERANK_TITLE_PREFIX", "1") not in ("0", "false", "False")

    # 보충 검색: 여러 대상을 이름으로 묻는 질문에서 근거에 없는 대상을 따로 다시 찾는다(LLM 없음, §5.45)
    followup_retrieval: bool = os.getenv("FOLLOWUP_RETRIEVAL", "0") not in ("0", "false", "False")
    followup_rounds: int = int(os.getenv("FOLLOWUP_ROUNDS", "1"))

    # --- 하이브리드 검색 (BM25 키워드 + 벡터 의미) ---
    # True 면 BM25(정확 매칭)와 벡터(의미)를 RRF 로 융합 → 고유명사·약어에 강해짐
    use_hybrid: bool = True
    # BM25 단어 나누기: whitespace(이전) | regex(영문·한글 분리 + 조사 제거, 기본) | kiwi(한글 형태소 분석). 측정 §5.32
    bm25_tokenizer: str = os.getenv("BM25_TOKENIZER", "regex")
    rrf_k: int = 60            # RRF 상수(클수록 순위차 영향 완화, 통상 60)

    # --- 질의 변환 ---
    # 기본 multi_query: 확장 평가셋(111문항) 검색 A/B에서 recall 최고(hyde는 최하).
    # hyde→multi_query 재검토 경위는 docs/ADR-001-query-transform-default.md(§재검토) 참고.
    # 다중 홉 분해(§5.48): 질문을 하위 질문으로 나눠(가벼운 LLM 1회) 검색어에 더하고, 하위 질문마다 맞는 논문이
    # 근거 자리에 하나는 들어가게 한다. 하위 질문 최고 점수가 이 값 미만이면 자리를 보장하지 않는다
    # 기본 켬: 다중 홉 기대 논문 재현율 0.925→0.970, 단일 질문 227개는 변화 없음(§5.48)
    multihop_decompose: bool = os.getenv("MULTIHOP_DECOMPOSE", "1") not in ("0", "false", "False")
    multihop_min_score: float = float(os.getenv("MULTIHOP_MIN_SCORE", "0.1"))
    query_transform_mode: str = "multi_query"  # "multi_query" | "hyde" | "decompose" | "none"
    num_query_variations: int = 3              # multi_query 생성 개수

    # --- 생성 ---
    # 생성 시스템 프롬프트 판: v1(이전) | v2(부분 답 허용·추론 표시, rag/core/prompts.py 주석)
    # 기본 v2: 기준 세트 50문항 맞음 36→41(다중 홉 5→9/18), 퇴보 1건(§5.49)
    generation_prompt: str = os.getenv("GENERATION_PROMPT", "v2")

    # --- 검증 ---
    enable_verification: bool = True
    # 검증 프롬프트 판: v1(근거 충실도만) | v2(질문을 주고 빠뜨린 부분까지 확인, rag/core/prompts.py 주석)
    # 기본 v2: v2 답변 판정 일치 κ 0.675(v1 은 전부 '충실'이라 κ 0, 규칙 판정 0.299)(§5.50)
    verify_prompt: str = os.getenv("VERIFY_PROMPT", "v2")
    max_regenerations: int = 1  # 검증 실패 시 재생성 최대 횟수

    # --- 4b 문장 단위 인용 확인(rag/stages/citation_check.py, LLM 없음 — 리랭커 점수) ---
    citation_check: bool = os.getenv("CITATION_CHECK", "1") not in ("0", "false", "False")
    # (문장, 인용한 근거) 리랭커 점수가 이 값 이상이면 '뒷받침됨'. eval/citation_check_eval.py 로 보정
    # 측정(81개 실제 답변): 0.05 에서 맞는 인용 91% 통과, 다른 논문 근거 40%·무관 근거 0% 통과. 걸리는 문장은 대부분 근거를 넘는 종합·추론
    citation_support_threshold: float = float(os.getenv("CITATION_SUPPORT_THRESHOLD", "0.05"))
    # 함의(NLI) 판정 모델로 '같은 주제지만 근거와 반대·다른 말'인 문장도 표시(§5.44). 빈 값이면 기본 모델.
    # 기본 끔: 방향을 뒤집은 문장은 리랭커보다 4배 잘 잡지만(45% vs 11%), 실제 답변의 맞는 한국어 바꿔 쓰기도 모순으로
    # 보는 경우가 많다(원래 문장 7.5% 경보, 상위 16개 대부분 오경보). 사람 판정 일치율을 본 뒤 결정
    citation_nli: bool = os.getenv("CITATION_NLI", "0") not in ("0", "false", "False")
    citation_nli_model: str = os.getenv("CITATION_NLI_MODEL", "")
    citation_nli_contradiction: float = float(os.getenv("CITATION_NLI_CONTRADICTION", "0.7"))

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
    # 첫 추정 제목으로 원 논문을 못 찾으면 경량 LLM 에 제목 후보를 이만큼 더 받아 찾는다(0 = 끔). 측정 §5.43:
    # 원 논문이 제목 검색 후보에 드는 질문 조정용 15/34 → 20/34, 확인용 9/12 → 10/12
    auto_fetch_title_candidates: int = int(os.getenv("AUTO_FETCH_TITLE_CANDIDATES", "3"))
    # 1 이면 질문 주어인 기법의 원 논문 제목(경량 LLM 추정)이 근거에 없을 때도 수집한다(3차 신호, §5.53:
    # 수집 신호 평가 세트 감지 26/34 → 32/34, 코퍼스 안 오작동 3/42 → 4/42)
    auto_fetch_original_signal: bool = os.getenv("AUTO_FETCH_ORIGINAL_SIGNAL", "1") not in ("0", "false", "False")
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
    # 그림 설명·수식(LaTeX) 청크도 만들지. 그림은 한국어 설명일 때 무관한 질문에 끼어들어 껐었다(§5.35).
    # 영어 설명 + 후보 분리 + 제목 붙이기로 294문항 지표 손해 없이 켬(§5.38)
    visual_figures: bool = os.getenv("VISUAL_FIGURES", "1") not in ("0", "false", "False")
    visual_equations: bool = os.getenv("VISUAL_EQUATIONS", "1") not in ("0", "false", "False")
    # 자동 수집한 전문 논문의 표·그림·수식을 챗봇 서버가 한가할 때 백그라운드로 처리(server/enricher.py, 로컬 모델 전용)
    visual_auto_enrich: bool = os.getenv("VISUAL_AUTO_ENRICH", "1") not in ("0", "false", "False")
    visual_auto_idle: float = float(os.getenv("VISUAL_AUTO_IDLE", "60"))   # 마지막 질문 뒤 이만큼(초) 조용하면 시작
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
    # 하이브리드 검색에서 본문 후보(top_k_retrieve)와 별도로 더하는 표·그림·수식 후보 수(0 = 표·그림·수식 제외). §5.37·§5.38
    visual_candidates: int = int(os.getenv("VISUAL_CANDIDATES", "2"))
    # 본문 후보 상위 몇 논문의 표·그림·수식 청크를 모두 리랭크 후보에 넣을지(0 = 끔). §5.38
    visual_from_top_papers: int = int(os.getenv("VISUAL_FROM_TOP_PAPERS", "1"))
    # 표·그림 청크가 근거에 들어가는 최소 리랭크 점수(0 = 제한 없음). §5.37. 리랭커 제목 붙이기와 함께 0.1→0.2(§5.42)
    visual_min_score: float = float(os.getenv("VISUAL_MIN_SCORE", "0.2"))
    # 최종 근거의 최소 리랭크 점수(0 = 끔). 근거 5자리를 채우려고 관련도 0 근처 청크(예: kor_paper 0.0)가 들어가던 것을 뺀다.
    # 평가셋 294문항: 0.003 까지 recall@5 0.9700 그대로·근거 평균 5 → 3.6개, 가장 낮은 정답 근거가 0.0042 라 절반인 0.002(§5.55)
    evidence_min_score: float = float(os.getenv("EVIDENCE_MIN_SCORE", "0.002"))
    # 다중 홉(하위 질문 2개 이상) 질문의 생성에만 로컬 Thinking 켬
    thinking_auto: bool = os.getenv("THINKING_AUTO", "0") not in ("0", "false", "False")
    # 자동 수집 논문(origin=auto)을 검색에서 숨김. 지우지 않고 '기준 코퍼스'만으로 평가할 때 쓴다
    # (평가 스크립트의 --corpus base 가 켠다. 챗봇은 꺼 둔 채로 자동 수집분까지 검색)
    exclude_auto_papers: bool = os.getenv("EXCLUDE_AUTO_PAPERS", "0") in ("1", "true", "True")

    # --- 실제 사용 질문 기록(server/querylog.py) — 질문·답변·근거·검증과 👍/👎 피드백을 로컬 SQLite 에 ---
    query_log: bool = os.getenv("QUERY_LOG", "1") not in ("0", "false", "False")
    query_log_path: str = os.getenv("QUERY_LOG_PATH", "./storage/query_log.db")

    # 대화 저장(server/conversations.py) — 화면의 대화를 로컬 SQLite 에도 남겨 다른 브라우저에서 이어 보기
    conversations: bool = os.getenv("CONVERSATIONS", "1") not in ("0", "false", "False")
    conversations_path: str = os.getenv("CONVERSATIONS_PATH", "./storage/conversations.db")

    # 사람 판정 기준 세트(eval/human_set.py)와 판정 결과(챗봇 '사람 판정' 화면) — 근거 원문이 들어 있어 storage/ 에
    human_set_path: str = os.getenv("HUMAN_SET_PATH", "./storage/human_set.jsonl")
    human_labels_path: str = os.getenv("HUMAN_LABELS_PATH", "./storage/human_labels.json")

    # --- 품질 하드닝 ---
    enable_llm_cache: bool = True          # 동일 프롬프트 응답 캐싱(SQLite) → 비용·속도 절감
    llm_cache_path: str = "./storage/llm_cache.db"
    llm_max_retries: int = 2               # API 일시 오류 시 자동 재시도 횟수
    # 로컬 서버가 응답 없이 연결만 붙잡으면 무한 대기한다(실험 중 3시간 멈춤) → 초 단위 상한 후 재시도
    llm_timeout: float = field(default_factory=lambda: float(os.getenv("LLM_TIMEOUT", "300")))

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
        if self.vector_backend == "pgvector" and not self.database_url:
            raise RuntimeError("VECTOR_BACKEND=pgvector 는 PostgreSQL 로그인 정보(POSTGRES_* 또는 DATABASE_URL)가 필요합니다. .env 를 확인하세요.")


# 싱글턴처럼 import 해서 사용: from config import settings
settings = Settings()
