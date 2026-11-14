"""
rag/schemas.py — 파이프라인을 흐르는 데이터 구조

각 단계의 입출력을 명시적 타입으로 정의하면, 디버깅/시각화(Streamlit)에서
"이 단계가 무엇을 만들었는지"가 한눈에 보입니다. 학습에 특히 중요.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class RetrievedChunk(BaseModel):
    """검색된 하나의 근거 조각."""
    content: str
    id: str | None = None     # 벡터 스토어의 청크 ID — 저장된 임베딩을 다시 꺼내 쓰는 데 사용
    source: str = ""          # arXiv ID
    title: str = ""
    section: str = ""         # 전문 기반일 때 섹션명(Method/Results 등)
    url: str = ""             # 원문 링크 (arXiv abs URL 등)
    origin: str = ""          # "auto" 면 챗봇이 자동 수집한 논문의 청크
    kind: str = ""            # "table"/"figure" 면 표·그림 이해로 추가한 청크(rag/visual.py)
    score: float = 0.0        # 벡터 유사도 또는 리랭크 점수
    rerank_score: float | None = None

    @property
    def link(self) -> str:
        """표시용 링크. url 이 없으면 arXiv ID 로 abs URL 구성."""
        if self.url:
            return self.url
        if self.source:
            return f"https://arxiv.org/abs/{self.source}"
        return ""


class VerificationResult(BaseModel):
    """4단계(검증) 결과 — LLM 이 JSON 으로 채웁니다."""
    grounded: bool = Field(description="답변이 제공된 근거만으로 뒷받침되는가")
    faithfulness: float = Field(ge=0, le=1, description="근거 충실도 0~1")
    unsupported_claims: list[str] = Field(
        default_factory=list, description="근거에 없는 주장 목록"
    )
    reasoning: str = Field(default="", description="판단 근거")


class PipelineResult(BaseModel):
    """4단계 전체 실행 결과 — UI/평가에서 그대로 사용."""
    question: str
    standalone_question: str = ""                                  # 0단계: 문맥화된 질문
    transformed_queries: list[str] = Field(default_factory=list)   # 1단계 산출
    retrieved: list[RetrievedChunk] = Field(default_factory=list)  # 벡터검색 결과
    reranked: list[RetrievedChunk] = Field(default_factory=list)   # 2단계 산출
    answer: str = ""                                               # 3단계 산출
    verification: VerificationResult | None = None                # 4단계 산출
    regenerated: bool = False                                      # 검증 후 재생성 여부
    auto_fetched: list[dict] = Field(default_factory=list)         # 근거 부족으로 자동 추가한 논문

    # 단계별 소요시간(초) — 성능 관찰용
    timings: dict[str, float] = Field(default_factory=dict)
