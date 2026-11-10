"""
eval/corpus.py — 평가에 쓸 코퍼스 선택 (자동 수집 논문 포함/제외)

챗봇은 근거가 부족하면 논문을 자동 수집해(origin=auto) 코퍼스를 키운다. 그대로 평가하면
이전 수치와 코퍼스가 달라 비교가 어긋나므로, 지우지 않고 평가할 때만 골라 쓴다.

  base : 자동 수집 논문을 숨긴 기준 코퍼스 → 이전 수치와 그대로 비교 가능 (평가 기본값)
  all  : 자동 수집분까지 포함한 전체 코퍼스 → 실제 챗봇 상태의 성능
  both : 두 코퍼스를 차례로 측정해 나란히 보고 (compare.py)
"""
from __future__ import annotations

from collections import Counter

from config import settings

LABELS = {
    "base": "기준 코퍼스(자동 수집 제외)",
    "all": "전체 코퍼스(자동 수집 포함)",
}


def add_corpus_arg(ap, allow_both: bool = False) -> None:
    choices = ["base", "all"] + (["both"] if allow_both else [])
    ap.add_argument("--corpus", choices=choices, default="base",
                    help="base=자동 수집 논문 제외(기본, 이전 수치와 비교용) | all=포함(실제 챗봇 상태)"
                         + (" | both=둘 다 측정해 나란히" if allow_both else ""))


def corpora(arg: str) -> list[str]:
    return ["base", "all"] if arg == "both" else [arg]


def set_corpus(name: str) -> None:
    """이후 load_vectorstore() 가 돌려줄 코퍼스를 정한다."""
    settings.exclude_auto_papers = name == "base"


def auto_stats(vs) -> tuple[int, int]:
    """(자동 수집 논문 수, 청크 수). 숨긴 보기여도 실제 스토어 기준으로 센다."""
    raw = getattr(vs, "inner", vs).get()
    srcs = Counter((m or {}).get("source") for m in raw.get("metadatas", []) if (m or {}).get("origin") == "auto")
    return len(srcs), sum(srcs.values())


def describe(name: str, vs) -> str:
    papers, chunks = auto_stats(vs)
    return f"{LABELS[name]} — 저장소의 자동 수집 논문 {papers}편/{chunks}청크 " + (
        "숨김" if name == "base" else "포함")
