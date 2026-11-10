"""
eval/dataset.py — 평가셋 단일 소스(JSONL) 로더

평가 질문을 코드(compare.py/evaluate.py)에 하드코딩하지 않고 `eval/eval_set.jsonl`
한 곳에서 관리합니다. 평가셋을 키우거나(수백 문항) 난이도·카테고리로 필터링해도
코드를 고칠 필요가 없습니다 — 이것이 "더 큰 평가셋 재검토"의 기반입니다.

각 레코드(JSON 한 줄) 필드:
  - id              : 고유 ID (예: "mh01")
  - question        : 질문 텍스트
  - category        : 주제 분류 (리포트 집계용)
  - difficulty      : "easy" | "medium" | "hard"
  - multi_hop       : 여러 논문을 종합해야 하는 질문인가
  - expected_sources: 이 질문에 '검색돼야 하는' 논문 source(arXiv ID) 목록
                      → recall@k / MRR 같은 LLM-무관(무료) 검색 지표 계산에 사용
"""
from __future__ import annotations

import json
from pathlib import Path

EVAL_SET_PATH = Path(__file__).resolve().parent / "eval_set.jsonl"


def load_eval_set(
    path: Path | str = EVAL_SET_PATH,
    difficulty: str | None = None,
    category: str | None = None,
    limit: int | None = None,
) -> list[dict]:
    """평가셋을 로드. difficulty/category 로 필터, limit 로 개수 제한 가능."""
    records: list[dict] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        rec = json.loads(line)
        if difficulty and rec.get("difficulty") != difficulty:
            continue
        if category and rec.get("category") != category:
            continue
        records.append(rec)
    return records[:limit] if limit else records


def questions(**kwargs) -> list[str]:
    """질문 텍스트만 리스트로 (기존 EVAL_QUESTIONS 대체용)."""
    return [r["question"] for r in load_eval_set(**kwargs)]
