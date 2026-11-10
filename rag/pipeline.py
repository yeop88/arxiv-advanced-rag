"""
rag/pipeline.py — 4단계를 하나로 엮는 오케스트레이터

   질문
    │
    ▼  [1단계] 질의 변환      stage1_query_transform.transform_query
   여러 검색 질의
    │
    ▼  [2단계] 검색 + 리랭킹   stage2_rerank.retrieve_and_rerank
   엄선된 근거 top-k
    │
    ▼  [3단계] 생성(인용)      stage3_generate.generate
   초안 답변
    │
    ▼  [4단계] 검증            stage4_verify.verify
   grounded?  ──No──▶ 재생성(최대 N회) ──▶ 재검증
    │Yes
    ▼
   최종 답변 + 검증 리포트

각 단계 산출물을 PipelineResult 에 모두 담아 반환하므로,
Streamlit 에서 단계별로 펼쳐 보며 "무슨 일이 일어났는지" 학습할 수 있습니다.
"""
from __future__ import annotations

import time

from config import settings
from rag import stage1_query_transform as s1
from rag import stage2_rerank as s2
from rag import stage3_generate as s3
from rag import stage4_verify as s4
from rag.contextualize import ChatHistory, contextualize
from rag.schemas import PipelineResult
from rag.vectorstore import VectorStoreLike, load_vectorstore


class RAGPipeline:
    """4단계 Advanced RAG 파이프라인.

    사용:
        pipe = RAGPipeline()
        result = pipe.run("트랜스포머의 self-attention 은 어떻게 동작하나요?")
        print(result.answer)
    """

    def __init__(self, vectorstore: VectorStoreLike | None = None):
        self.vectorstore = vectorstore or load_vectorstore()

    def run(
        self, question: str, chat_history: ChatHistory | None = None
    ) -> PipelineResult:
        """4단계 파이프라인 실행.

        Args:
            question: 사용자 질문
            chat_history: 멀티턴 대화 히스토리 [(role, content), ...]. 있으면
                0단계(문맥화)로 후속질문을 독립형으로 재작성 후 진행.
        """
        result = PipelineResult(question=question)

        # --- 0단계(전처리): 멀티턴 문맥화 ---
        t = time.perf_counter()
        standalone = contextualize(question, chat_history)
        result.standalone_question = standalone
        result.timings["0_contextualize"] = time.perf_counter() - t

        # --- 1단계: 질의 변환 (문맥화된 질문 사용) ---
        # 질의변환은 '있으면 좋은' 보조 단계 → 실패해도 원 질문으로 계속 진행(graceful degrade)
        t = time.perf_counter()
        try:
            queries = s1.transform_query(standalone)
        except Exception as e:  # noqa: BLE001
            print(f"[경고] 질의변환 실패, 원 질문으로 진행: {e}")
            queries = [standalone]
        result.transformed_queries = queries
        result.timings["1_query_transform"] = time.perf_counter() - t

        # --- 2단계: 검색 + 리랭킹 ---
        t = time.perf_counter()
        retrieved, reranked = s2.retrieve_and_rerank(self.vectorstore, standalone, queries)
        result.retrieved = retrieved
        result.reranked = reranked
        result.timings["2_retrieve_rerank"] = time.perf_counter() - t

        if not reranked:
            result.answer = "관련 근거를 찾지 못했습니다. 다른 질문을 시도하거나 논문을 더 수집하세요."
            return result

        # --- 3단계: 생성 (문맥화된 질문으로 자기완결적 답변) ---
        # 생성은 핵심 단계 → 실패하면 사용자에게 명확히 알리고 종료(크래시 방지)
        t = time.perf_counter()
        try:
            answer = s3.generate(standalone, reranked)
        except Exception as e:  # noqa: BLE001
            result.answer = f"답변 생성 중 오류가 발생했습니다: {e}"
            result.timings["3_generate"] = time.perf_counter() - t
            return result
        result.timings["3_generate"] = time.perf_counter() - t

        # --- 4단계: 검증 (+ 필요시 자기수정 재생성) ---
        # 검증 실패는 치명적이지 않음 → 답변은 유지하고 검증만 생략(graceful degrade)
        if settings.enable_verification:
            t = time.perf_counter()
            try:
                verification = s4.verify(answer, reranked)

                attempts = 0
                while (
                    not verification.grounded
                    and attempts < settings.max_regenerations
                ):
                    # 근거 없는 주장을 피드백으로 넣어 재생성 → 재검증
                    answer = s3.regenerate(standalone, reranked, verification.unsupported_claims)
                    verification = s4.verify(answer, reranked)
                    attempts += 1
                    result.regenerated = True

                result.verification = verification
            except Exception as e:  # noqa: BLE001
                print(f"[경고] 검증 단계 실패, 답변은 유지: {e}")
            result.timings["4_verify"] = time.perf_counter() - t

        result.answer = answer
        return result
