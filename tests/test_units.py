"""
tests/test_units.py — 순수 함수 단위 테스트 (네트워크/LLM 불필요)

LLM 호출 없이 검증 가능한 핵심 로직들을 빠르게 점검합니다:
  - RRF 융합 순위 계산
  - LLM 출력 파싱 헬퍼(_split_lines, _parse_json)
  - 멀티턴 문맥화의 '히스토리 없음' 단축 경로

실행: python tests/test_units.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.documents import Document


def test_rrf() -> None:
    """RRF: 두 랭킹에서 모두 상위인 문서가 1등이 되어야 한다."""
    from rag.stage2_rerank import _reciprocal_rank_fusion

    A = Document(page_content="doc A")
    B = Document(page_content="doc B")
    C = Document(page_content="doc C")
    ranking1 = [A, B, C]  # A가 1위
    ranking2 = [A, C, B]  # A가 1위
    fused = _reciprocal_rank_fusion([ranking1, ranking2], k=60)
    top_doc = fused[0][0]
    assert top_doc.page_content == "doc A", "두 랭킹 모두 1위인 A가 종합 1위여야 함"
    # 점수는 내림차순 정렬
    scores = [s for _, s in fused]
    assert scores == sorted(scores, reverse=True), "RRF 점수가 내림차순이어야 함"
    print("  test_rrf ✅")


def test_split_lines() -> None:
    """_split_lines: 번호/불릿 접두 제거, 빈 줄 제거."""
    from rag.stage1_query_transform import _split_lines

    out = _split_lines("1. first\n- second\n\n  3) third  ")
    assert out == ["first", "second", "third"], out
    print("  test_split_lines ✅")


def test_parse_json() -> None:
    """_parse_json: 코드펜스/주변 텍스트가 있어도 JSON 추출."""
    from rag.stage4_verify import _parse_json

    assert _parse_json('{"grounded": true}')["grounded"] is True
    fenced = '```json\n{"faithfulness": 0.8}\n```'
    assert _parse_json(fenced)["faithfulness"] == 0.8
    noisy = '설명 텍스트 {"a": 1} 뒤에도 말이 있음'
    assert _parse_json(noisy)["a"] == 1
    print("  test_parse_json ✅")


def test_contextualize_no_history() -> None:
    """히스토리가 없으면 LLM 호출 없이 원문 그대로 반환해야 한다."""
    from rag.contextualize import contextualize

    q = "리랭킹은 왜 필요한가요?"
    assert contextualize(q, None) == q
    assert contextualize(q, []) == q
    print("  test_contextualize_no_history ✅")


class _FakeEmbeddings:
    """테스트용 가짜 임베딩: 각 텍스트를 서로 직교하는 one-hot 벡터로(유사도 0)."""

    def embed_documents(self, texts):
        n = len(texts)
        return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


def test_mmr_per_source_cap() -> None:
    """MMR: 논문당 상한으로 한 논문의 쏠림을 막고 커버리지를 넓혀야 한다."""
    import rag.llm as llm
    from rag.schemas import RetrievedChunk
    from rag.stage2_rerank import _mmr_select

    orig = llm.get_embeddings
    llm.get_embeddings = lambda: _FakeEmbeddings()
    try:
        # P1이 관련도 상위를 독점하는 상황 (점수 내림차순)
        srcs = ["P1", "P1", "P1", "P2", "P3"]
        cands = [
            RetrievedChunk(content=f"c{i}", source=s, title=s, rerank_score=5 - i)
            for i, s in enumerate(srcs)
        ]
        picked = _mmr_select(cands, k=4, lambda_=0.6, same_source_penalty=0.9, max_per_source=2)
        got = [c.source for c in picked]
        assert len(picked) == 4, got
        assert got.count("P1") <= 2, f"P1 상한 2 초과: {got}"
        assert {"P2", "P3"} <= set(got), f"다른 논문 커버 실패: {got}"

        # 후보 논문이 부족하면 상한을 완화해 슬롯을 채운다
        only_p1 = [RetrievedChunk(content=f"c{i}", source="P1", title="P1", rerank_score=3 - i)
                   for i in range(3)]
        picked2 = _mmr_select(only_p1, k=3, lambda_=0.6, same_source_penalty=0.9, max_per_source=2)
        assert len(picked2) == 3, "후보가 한 논문뿐이면 상한을 완화해 k를 채워야 함"
    finally:
        llm.get_embeddings = orig
    print("  test_mmr_per_source_cap ✅")


def main() -> None:
    print("단위 테스트 실행:")
    test_rrf()
    test_split_lines()
    test_parse_json()
    test_contextualize_no_history()
    test_mmr_per_source_cap()
    print("모든 단위 테스트 통과 ✅")


if __name__ == "__main__":
    main()
