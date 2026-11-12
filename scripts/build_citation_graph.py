"""
scripts/build_citation_graph.py — DB 논문의 인용 그래프를 OpenAlex 에서 받아 저장(LLM 없음)

    python scripts/build_citation_graph.py            # 그래프에 없는 논문만 추가
    python scripts/build_citation_graph.py --stats    # 저장된 그래프 요약만 출력

결과: CITATION_GRAPH_PATH(기본 storage/citation_graph.json). 자동 수집 논문은 수집 때 자동으로 추가된다.
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import settings  # noqa: E402
from rag import citations  # noqa: E402


def db_papers() -> dict[str, str]:
    """DB 논문의 arXiv ID → 제목."""
    from rag.vectorstore import load_vectorstore

    settings.exclude_auto_papers = False
    metas = load_vectorstore().get(include=["metadatas"])["metadatas"]
    return {m["source"]: m.get("title", "") for m in metas if m and m.get("source")}


def stats(g: dict, sources: list[str] | None = None) -> None:
    papers = g["papers"]
    if sources is not None:
        arx = [s for s in sources if citations.is_arxiv_id(s)]
        print(f"DB 논문 {len(sources)}편 중 arXiv {len(arx)}편 → 그래프 {sum(s in papers for s in arx)}편, "
              f"OpenAlex 에서 못 찾음 {sum(s in g['missing'] for s in arx)}편")
    edges = sum(len(citations.related_in_corpus(a, g)["prior"]) for a in papers)
    no_refs = sum(not p["refs"] for p in papers.values())
    print(f"그래프 논문 {len(papers)}편, 코퍼스 안 인용 관계 {edges}개, 참고문헌 정보 없음 {no_refs}편")
    top = sorted(papers, key=lambda a: -len(citations.related_in_corpus(a, g)["later"]))[:5]
    for a in top:
        print(f"  코퍼스 안에서 가장 많이 인용됨: {papers[a]['title'][:60]} ({len(citations.related_in_corpus(a, g)['later'])}편)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats", action="store_true")
    args = ap.parse_args()
    if args.stats:
        stats(citations.load())
        return
    papers = db_papers()
    sources = sorted(papers)
    g = citations.build(sources, titles=papers, progress=lambda i, n: print(f"  {i}/{n}", flush=True))
    stats(g, sources)


if __name__ == "__main__":
    main()
