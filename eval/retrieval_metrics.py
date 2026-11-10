"""
eval/retrieval_metrics.py — 검색 단계 지표 (LLM 불필요 = 무료·결정적)

RAGAS 스타일 지표(faithfulness 등)는 LLM 심판을 호출해 질문당 비용·지연이 든다.
반면 "정답 논문이 상위 k 안에 검색됐는가"는 **expected_sources 와 검색 결과만으로**
계산되므로 LLM이 전혀 필요 없다. 그래서 평가셋을 수백 문항으로 키워도 비용이 0이다.

이것이 "더 큰 평가셋 재검토"를 싸게 만드는 핵심 레버:
  - 1차 스크리닝은 recall@k / MRR(무료)로 수백 문항을 돌려 검색 품질 회귀를 잡고,
  - 비싼 LLM-judge(faithfulness 등)는 대표 표본에만 쓴다.

지표:
  - recall@k : 기대 논문 중 상위 k 검색결과에 포함된 비율
  - hit@k    : 기대 논문이 하나라도 상위 k에 있으면 1
  - MRR      : 첫 번째 정답 논문의 역순위(1/rank) 평균 기반
"""
from __future__ import annotations


def _ranked_sources(chunks) -> list[str]:
    """검색/리랭킹 결과(RetrievedChunk 리스트)에서 중복 제거된 source 순위 목록."""
    seen: list[str] = []
    for c in chunks:
        s = getattr(c, "source", None)
        if s and s not in seen:
            seen.append(s)
    return seen


def recall_at_k(chunks, expected_sources: list[str], k: int) -> float:
    """기대 논문 중 상위 k 안에 검색된 비율. expected 가 없으면 NaN 대신 None 취급은 호출부에서."""
    if not expected_sources:
        return 0.0
    topk = set(_ranked_sources(chunks)[:k])
    hits = sum(1 for s in expected_sources if s in topk)
    return hits / len(expected_sources)


def hit_at_k(chunks, expected_sources: list[str], k: int) -> float:
    """기대 논문이 하나라도 상위 k 안에 있으면 1.0, 아니면 0.0."""
    if not expected_sources:
        return 0.0
    topk = set(_ranked_sources(chunks)[:k])
    return 1.0 if any(s in topk for s in expected_sources) else 0.0


def reciprocal_rank(chunks, expected_sources: list[str]) -> float:
    """첫 번째 정답 논문의 역순위(1/rank). 없으면 0."""
    if not expected_sources:
        return 0.0
    exp = set(expected_sources)
    for i, s in enumerate(_ranked_sources(chunks), start=1):
        if s in exp:
            return 1.0 / i
    return 0.0
