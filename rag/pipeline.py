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
from typing import Any, Callable

from config import settings
from rag import auto_fetch
from rag import stage1_query_transform as s1
from rag import stage2_rerank as s2
from rag import stage3_generate as s3
from rag import stage4_verify as s4
from rag.contextualize import ChatHistory, contextualize
from rag.schemas import PipelineResult
from rag.vectorstore import VectorStoreLike, load_vectorstore

# 진행 알림: progress(event, **info). 화면(챗봇)에 단계별 진행을 보여주는 데 쓴다.
#   transform_start / transform_done(queries, sec)
#   retrieve_start / retrieve_done(n_candidates, papers, sec)
#   generate_start / token(text) / generate_done(sec, chars)
#   verify_start / verify_done(grounded, faithfulness, unsupported, sec)
#   regenerate_start(attempt, unsupported) / error(stage, message)
Progress = Callable[..., Any]


class RAGPipeline:
    """4단계 Advanced RAG 파이프라인.

    사용:
        pipe = RAGPipeline()
        result = pipe.run("트랜스포머의 self-attention 은 어떻게 동작하나요?")
        print(result.answer)
    """

    def __init__(self, vectorstore: VectorStoreLike | None = None):
        self.vectorstore = vectorstore or load_vectorstore()

    def _fetch_and_retrieve(self, standalone: str, queries: list[str], result: PipelineResult,
                            emit, reason: str, evidence=None, **info):
        """논문 자동 수집 → (추가된 게 있으면) 다시 검색. 새 리랭크 결과를 돌려주고, 추가 없으면 None.
        실패해도 예외를 올리지 않는다(기존 근거로 계속 진행 — graceful degrade)."""
        t = time.perf_counter()
        emit("fetch_start", reason=reason, **info)
        try:
            added = auto_fetch.fetch_and_ingest(standalone, self.vectorstore, emit,
                                                evidence_sources=[c.source for c in (evidence or [])])
        except Exception as e:  # noqa: BLE001
            added = []
            emit("error", stage="fetch", message=str(e))
        result.auto_fetched = added
        result.timings["2b_auto_fetch"] = time.perf_counter() - t
        emit("fetch_done", added=added, sec=result.timings["2b_auto_fetch"])
        if not added:
            return None
        t = time.perf_counter()
        emit("retrieve_start", n_queries=len(queries), again=True)
        retrieved, reranked = s2.retrieve_and_rerank(self.vectorstore, standalone, queries)
        result.retrieved, result.reranked = retrieved, reranked
        result.timings["2c_retrieve_again"] = time.perf_counter() - t
        emit("retrieve_done", n_candidates=len(retrieved),
             papers=[c.title or c.source for c in reranked], sec=result.timings["2c_retrieve_again"])
        return reranked

    def run(
        self, question: str, chat_history: ChatHistory | None = None,
        progress: Progress | None = None,
    ) -> PipelineResult:
        """4단계 파이프라인 실행.

        Args:
            question: 사용자 질문
            chat_history: 멀티턴 대화 히스토리 [(role, content), ...]. 있으면
                0단계(문맥화)로 후속질문을 독립형으로 재작성 후 진행.
            progress: 단계 진행 알림 함수(선택). 주면 답변 생성도 토큰 단위로 스트리밍된다.
        """
        result = PipelineResult(question=question)

        def emit(event: str, **info) -> None:
            if progress is not None:
                progress(event, **info)

        on_token = (lambda text: emit("token", text=text)) if progress is not None else None

        # --- 0단계(전처리): 멀티턴 문맥화 ---
        t = time.perf_counter()
        if chat_history:
            emit("contextualize_start")
        standalone = contextualize(question, chat_history)
        result.standalone_question = standalone
        result.timings["0_contextualize"] = time.perf_counter() - t
        if chat_history:
            emit("contextualize_done", standalone=standalone, changed=standalone != question,
                 sec=result.timings["0_contextualize"])

        # --- 1단계: 질의 변환 (문맥화된 질문 사용) ---
        # 질의변환은 '있으면 좋은' 보조 단계 → 실패해도 원 질문으로 계속 진행(graceful degrade)
        t = time.perf_counter()
        emit("transform_start", mode=settings.query_transform_mode)
        try:
            queries = s1.transform_query(standalone)
        except Exception as e:  # noqa: BLE001
            print(f"[경고] 질의변환 실패, 원 질문으로 진행: {e}")
            queries = [standalone]
        result.transformed_queries = queries
        result.timings["1_query_transform"] = time.perf_counter() - t
        emit("transform_done", queries=queries, sec=result.timings["1_query_transform"])

        # --- 2단계: 검색 + 리랭킹 ---
        t = time.perf_counter()
        emit("retrieve_start", n_queries=len(queries))
        retrieved, reranked = s2.retrieve_and_rerank(self.vectorstore, standalone, queries)
        result.retrieved = retrieved
        result.reranked = reranked
        result.timings["2_retrieve_rerank"] = time.perf_counter() - t
        emit("retrieve_done", n_candidates=len(retrieved),
             papers=[c.title or c.source for c in reranked], sec=result.timings["2_retrieve_rerank"])

        # --- 2b: 근거 부족 시 논문 자동 수집 → 다시 검색 (자기 확장, 1차 신호: 리랭크 점수) ---
        if settings.auto_fetch and auto_fetch.needs_more(reranked):
            reranked = self._fetch_and_retrieve(standalone, queries, result, emit, reason="score", evidence=reranked,
                                                top_score=auto_fetch.top_score(reranked),
                                                threshold=settings.auto_fetch_threshold) or reranked

        if not reranked:
            result.answer = "관련 근거를 찾지 못했습니다. 다른 질문을 시도하거나 논문을 더 수집하세요."
            return result

        # --- 3단계: 생성 (문맥화된 질문으로 자기완결적 답변) ---
        # 생성은 핵심 단계 → 실패하면 사용자에게 명확히 알리고 종료(크래시 방지)
        t = time.perf_counter()
        emit("generate_start", n_chunks=len(reranked))
        try:
            answer = s3.generate(standalone, reranked, on_token=on_token)
        except Exception as e:  # noqa: BLE001
            result.answer = f"답변 생성 중 오류가 발생했습니다: {e}"
            result.timings["3_generate"] = time.perf_counter() - t
            emit("error", stage="generate", message=str(e))
            return result
        result.timings["3_generate"] = time.perf_counter() - t
        emit("generate_done", sec=result.timings["3_generate"], chars=len(answer))

        # --- 3b: 답변이 '근거에서 찾을 수 없다'고 하면 논문을 찾아 추가하고 다시 작성 (2차 신호) ---
        # 점수로는 못 잡는 '비슷한 주제의 다른 논문만 있는' 경우. 한 질문에 수집은 최대 1회.
        if settings.auto_fetch and not result.auto_fetched and "2b_auto_fetch" not in result.timings \
                and auto_fetch.answer_signal(standalone, answer, reranked):
            new_reranked = self._fetch_and_retrieve(standalone, queries, result, emit, reason="answer",
                                                    evidence=reranked)
            if new_reranked:
                reranked = new_reranked
                t = time.perf_counter()
                emit("generate_start", n_chunks=len(reranked), again=True)
                try:
                    answer = s3.generate(standalone, reranked, on_token=on_token)
                except Exception as e:  # noqa: BLE001
                    emit("error", stage="generate", message=str(e))  # 실패하면 첫 답변 유지
                result.timings["3c_generate_again"] = time.perf_counter() - t
                emit("generate_done", sec=result.timings["3c_generate_again"], chars=len(answer))

        if settings.auto_fetch:  # 평가에서는 자동 수집을 끄므로 실제 사용만 기록된다
            try:
                auto_fetch.record_usage(reranked)
            except OSError as e:
                emit("error", stage="usage", message=str(e))

        # --- 4단계: 검증 (+ 필요시 자기수정 재생성) ---
        # 검증 실패는 치명적이지 않음 → 답변은 유지하고 검증만 생략(graceful degrade)
        if settings.enable_verification:
            t = time.perf_counter()
            emit("verify_start")
            try:
                verification = s4.verify(answer, reranked)
                emit("verify_done", grounded=verification.grounded, faithfulness=verification.faithfulness,
                     unsupported=verification.unsupported_claims, sec=time.perf_counter() - t)

                attempts = 0
                while (
                    not verification.grounded
                    and attempts < settings.max_regenerations
                ):
                    # 근거 없는 주장을 피드백으로 넣어 재생성 → 재검증
                    emit("regenerate_start", attempt=attempts + 1, unsupported=verification.unsupported_claims)
                    answer = s3.regenerate(standalone, reranked, verification.unsupported_claims,
                                           on_token=on_token)
                    t2 = time.perf_counter()
                    emit("verify_start", again=True)
                    verification = s4.verify(answer, reranked)
                    emit("verify_done", grounded=verification.grounded, faithfulness=verification.faithfulness,
                         unsupported=verification.unsupported_claims, sec=time.perf_counter() - t2)
                    attempts += 1
                    result.regenerated = True

                result.verification = verification
            except Exception as e:  # noqa: BLE001
                print(f"[경고] 검증 단계 실패, 답변은 유지: {e}")
                emit("error", stage="verify", message=str(e))
            result.timings["4_verify"] = time.perf_counter() - t

        result.answer = answer
        return result
