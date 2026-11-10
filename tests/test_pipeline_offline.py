"""
tests/test_pipeline_offline.py — API 키 없이 4단계 배선 전체를 검증하는 오프라인 테스트

실제 LLM 대신 FakeListChatModel 을 주입해, 네트워크/비용 없이
파이프라인의 '플러밍'(프롬프트 구성 → 체인 실행 → JSON 파싱 → 자기수정 루프)이
올바른지 확인합니다. 로컬 임베딩/리랭커는 진짜로 동작합니다.

실행:
    python tests/test_pipeline_offline.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.language_models.fake_chat_models import FakeListChatModel

from config import settings


def main() -> None:
    # stage 모듈들은 'from rag.llm import get_llm' 로 import 하므로,
    # 각 모듈의 get_llm 심볼을 직접 교체해야 함.
    fake_light = FakeListChatModel(
        responses=[
            "cross-encoder reranking in RAG\nbi-encoder vs cross-encoder precision\nsecond-stage reranking recall",
        ]
    )
    fake_main = FakeListChatModel(
        responses=[
            "리랭킹은 1차 벡터검색의 정밀도 한계를 보완합니다 [1]. "
            "크로스인코더는 질문과 문단을 함께 보고 점수를 매겨 더 정확합니다 [2].",
            '{"grounded": true, "faithfulness": 0.9, "unsupported_claims": [], "reasoning": "근거로 뒷받침됨."}',
        ]
    )

    def fake_get_llm(light: bool = False, raw: bool = False):  # raw: 스트리밍 경로용(가짜 모델은 동일)
        return fake_light if light else fake_main

    # 각 stage 모듈의 get_llm 교체
    import rag.stage1_query_transform as s1
    import rag.stage3_generate as s3
    import rag.stage4_verify as s4

    s1.get_llm = fake_get_llm
    s3.get_llm = fake_get_llm
    s4.get_llm = fake_get_llm

    settings.query_transform_mode = "multi_query"
    settings.enable_verification = True

    from rag.pipeline import RAGPipeline

    pipe = RAGPipeline()
    result = pipe.run("리랭킹은 RAG에서 왜 필요한가요?")

    # --- 검증 ---
    assert result.transformed_queries, "1단계: 질의변환 결과 비어있음"
    assert result.reranked, "2단계: 리랭킹 결과 비어있음"
    assert result.answer, "3단계: 답변 비어있음"
    assert result.verification is not None, "4단계: 검증 결과 없음"

    print("=== 오프라인 전 파이프라인 검증 ===")
    print(f"1단계 질의변환: {len(result.transformed_queries)}개")
    for q in result.transformed_queries:
        print(f"   - {q}")
    print(f"2단계 리랭킹 선택: {len(result.reranked)}개 (최상위: {result.reranked[0].title[:50]})")
    print(f"3단계 답변: {result.answer[:80]}...")
    print(f"4단계 검증: grounded={result.verification.grounded}, "
          f"faithfulness={result.verification.faithfulness}")
    print(f"재생성 여부: {result.regenerated}")

    # --- 진행 알림 + 답변 스트리밍 (챗봇 실시간 표시용) ---
    events: list[tuple[str, dict]] = []
    result2 = pipe.run("리랭킹은 RAG에서 왜 필요한가요?", progress=lambda ev, **kw: events.append((ev, kw)))
    names = [e for e, _ in events if e != "token"]
    expected = ["transform_start", "transform_done", "retrieve_start", "retrieve_done",
                "generate_start", "generate_done", "verify_start", "verify_done"]
    assert names == expected, f"진행 알림 순서가 다름: {names}"
    tokens = [kw["text"] for e, kw in events if e == "token"]
    assert len(tokens) > 1, "답변이 토큰 단위로 스트리밍되지 않음"
    assert tokens[-1].strip() == result2.answer, "스트리밍 마지막 텍스트와 최종 답변이 다름"
    first_tok = next(i for i, (e, _) in enumerate(events) if e == "token")
    assert events[first_tok - 1][0] == "generate_start", "토큰은 generate_start 뒤에 와야 함"
    print(f"진행 알림 {len(names)}개 순서 OK, 답변 스트리밍 토큰 {len(tokens)}회")
    print()
    print("모든 단계 통과 ✅")


if __name__ == "__main__":
    main()
