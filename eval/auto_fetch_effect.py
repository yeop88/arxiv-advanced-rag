"""
eval/auto_fetch_effect.py — 논문 자동 수집이 실제로 도움이 됐는지(수집 전 vs 후) 측정

자동 수집 논문에는 수집을 일으킨 질문(fetch_query)이 저장돼 있다. 그 질문들을
  - 수집 전: 자동 수집 논문을 숨긴 기준 코퍼스(base)
  - 수집 후: 전체 코퍼스(all)
에서 똑같이 검색·리랭크해 비교한다. 질의변환 없이 원 질문으로 검색 → LLM 0회(무료).

지표(질문별):
  - 최고 리랭크 점수 / 수집 트리거 여부(점수 < AUTO_FETCH_THRESHOLD)
  - 최종 근거 k개 중 자동 수집 논문 청크 수
  - 수집한 논문 recall@k (그 질문으로 수집한 논문들을 정답으로 간주)

한계: 정답을 '시스템이 고른 논문'으로 두므로, 고른 논문이 맞는지는 이 스크립트가 판단하지 않는다.
      "수집한 근거가 검색에 실제로 올라와 쓰이는가 / 점수가 정상 범위로 올라가는가"를 본다.
      논문이 맞는지는 수집 로그의 제목이나 답변을 직접 확인한다.

사용:
    python eval/auto_fetch_effect.py
"""
from __future__ import annotations

import sys
import warnings
from collections import defaultdict
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import settings  # noqa: E402

settings.auto_fetch = False
settings.query_transform_mode = "none"

from eval.corpus import set_corpus  # noqa: E402
from eval.retrieval_metrics import recall_at_k  # noqa: E402
from rag.stage2_rerank import retrieve_and_rerank  # noqa: E402
from rag.vectorstore import load_vectorstore  # noqa: E402


def fetch_groups(vs) -> dict[str, dict]:
    """수집을 일으킨 질문 → {sources: 그 질문으로 수집한 논문 ID들, titles}."""
    raw = getattr(vs, "inner", vs).get()
    groups: dict[str, dict] = defaultdict(lambda: {"sources": set(), "titles": {}})
    for m in raw.get("metadatas", []):
        m = m or {}
        if m.get("origin") == "auto" and m.get("fetch_query"):
            g = groups[m["fetch_query"]]
            g["sources"].add(m.get("source"))
            g["titles"][m.get("source")] = m.get("title", "")
    return dict(groups)


def measure(vs, q: str, expected: list[str], auto_srcs: set[str]) -> dict:
    _, rr = retrieve_and_rerank(vs, q, [q])
    k = settings.top_k_rerank
    top = max((c.rerank_score or 0.0 for c in rr), default=0.0)
    return {
        "top": top, "trigger": top < settings.auto_fetch_threshold,
        "auto_in_topk": sum(c.source in auto_srcs for c in rr),
        "recall": recall_at_k(rr, expected, k),
        "papers": [c.source for c in rr],
    }


def main() -> None:
    set_corpus("all")
    vs_all = load_vectorstore()
    groups = fetch_groups(vs_all)
    if not groups:
        print("자동 수집 논문이 없습니다(origin=auto). 챗봇에서 코퍼스 밖 질문을 해 보세요.")
        return
    auto_srcs = set().union(*(g["sources"] for g in groups.values()))
    set_corpus("base")
    vs_base = load_vectorstore()

    k = settings.top_k_rerank
    thr = settings.auto_fetch_threshold
    print(f"자동 수집 질문 {len(groups)}개, 논문 {len(auto_srcs)}편 | 수집 트리거 기준: 최고 점수 < {thr}\n")
    rows = []
    for q, g in groups.items():
        exp = sorted(g["sources"])
        b = measure(vs_base, q, exp, auto_srcs)
        a = measure(vs_all, q, exp, auto_srcs)
        rows.append((q, g, b, a))
        print(f"■ {q[:70]}")
        for s in exp:
            print(f"    수집: [{s}] {g['titles'][s][:70]}")

    print(f"\n| 질문 | 최고 점수 전→후 | 트리거 전→후 | top-{k} 중 수집 논문 청크 | 수집 논문 recall@{k} 전→후 |")
    print("|---|---|---|---|---|")
    for q, _, b, a in rows:
        trig = lambda r: "걸림" if r["trigger"] else "안 걸림"  # noqa: E731
        print(f"| {q[:40]} | {b['top']:.3f} → {a['top']:.3f} | {trig(b)} → {trig(a)} "
              f"| {b['auto_in_topk']} → {a['auto_in_topk']} | {b['recall']:.2f} → {a['recall']:.2f} |")

    n = len(rows)
    avg = lambda key, side: sum(r[side][key] for r in rows) / n  # noqa: E731
    print(f"\n평균 최고 점수 {avg('top', 2):.3f} → {avg('top', 3):.3f} | "
          f"트리거 {sum(r[2]['trigger'] for r in rows)}/{n} → {sum(r[3]['trigger'] for r in rows)}/{n} | "
          f"수집 논문 recall@{k} {avg('recall', 2):.2f} → {avg('recall', 3):.2f}")
    print("※ '후'에도 트리거가 걸리면 같은 질문을 다시 해도 또 수집을 시도한다는 뜻 → 수집 논문이 질문과 안 맞을 가능성")


if __name__ == "__main__":
    main()
