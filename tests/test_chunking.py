"""
tests/test_chunking.py — 섹션 인식 청킹 검증 (네트워크/LLM 불필요)

arXiv PDF 다운로드가 막힌 환경에서도, 합성 논문 텍스트로 섹션 분리·메타데이터
부여가 올바른지 확인합니다.

실행: python tests/test_chunking.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.chunking import section_aware_split, split_into_sections

# 논문 전문을 흉내 낸 합성 텍스트 (번호형 + 이름형 섹션 혼용)
SAMPLE_PAPER = """A Study of Retrieve-then-Rerank Pipelines
John Doe, Jane Smith

Abstract
We study two-stage retrieval where a fast retriever selects candidates and a
cross-encoder reranks them. We find reranking improves precision substantially.

1 Introduction
Dense retrieval with bi-encoders scales to millions of documents but trades off
precision. This motivates a second-stage reranker.

2 Method
We use a bi-encoder for first-stage retrieval over a vector index, then a
cross-encoder that jointly encodes the query and each candidate passage.
The cross-encoder outputs a relevance score used to reorder candidates.

3 Experiments
We evaluate on a passage ranking benchmark with 20 candidates per query.

4 Results
Reranking improves nDCG by a large margin over bi-encoder-only retrieval.

5 Conclusion
Two-stage retrieve-then-rerank is an effective and practical design.

References
[1] Some reference.
"""


def main() -> None:
    sections = split_into_sections(SAMPLE_PAPER)
    names = [n for n, _ in sections]
    print("감지된 섹션:", names)

    # 핵심 섹션들이 잡혔는지
    joined = " ".join(names).lower()
    for expected in ["abstract", "method", "results", "conclusion"]:
        assert expected in joined, f"섹션 '{expected}' 미감지"

    base_meta = {"source": "0000.00000", "title": "A Study of Retrieve-then-Rerank"}
    docs = section_aware_split(SAMPLE_PAPER, base_meta)
    print(f"\n청크 수: {len(docs)}")
    for d in docs:
        print(f"  [{d.metadata.get('section'):<20}] {d.page_content[:50].strip()}...")

    # 모든 청크가 section 메타를 갖는지 + base_meta 유지되는지
    for d in docs:
        assert d.metadata.get("section"), "section 메타 누락"
        assert d.metadata.get("source") == "0000.00000", "base_meta 유실"

    # 섹션 못 찾는 텍스트(초록만) → body 폴백
    fallback = split_into_sections("Just a short abstract with no headings.")
    assert fallback == [("body", "Just a short abstract with no headings.")], fallback

    print("\n섹션 인식 청킹 검증 통과 ✅")


if __name__ == "__main__":
    main()
