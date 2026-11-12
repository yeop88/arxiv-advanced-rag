"""
eval/original_paper_eval.py — 원 논문 수집률 측정(A3). DB 에 아무것도 넣지 않는다.

B1 기록(eval/results/fetch_signal_record.jsonl)의 '수집 필요' 질문마다 후보 출처를 한 번씩 모은 뒤,
조합별로 관련도 채점·선정만 해서 원 논문이 뽑히는지 본다.

    기존            검색어 검색 + 인용 순 검색 (A3 이전)
    +제목 추정      + 경량 LLM 이 추정한 원 논문 제목으로 검색, 맞는 후보를 먼저
    +공통 참고문헌  + 근거 논문들이 공통으로 인용한 논문
    둘 다

LLM: 경량 모델(검색어 생성·제목 추정, 질문당 2회, 캐시됨). 나머지는 arXiv·OpenAlex 검색과 로컬 리랭커.
    python eval/original_paper_eval.py            # 측정(이어서 실행 가능)
    python eval/original_paper_eval.py --report   # 요약만
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import settings  # noqa: E402

RECORD = Path(__file__).resolve().parent / "results" / "fetch_signal_record.jsonl"
OUT = Path(__file__).resolve().parent / "results" / "original_paper_eval.jsonl"
CONFIGS = {"기존": (False, False), "+제목 추정": (True, False), "+공통 참고문헌": (False, True), "둘 다": (True, True)}


def target(row: dict) -> str:
    """원 논문: 비교·멀티턴은 DB 에 없는 쪽(XLNet), 나머지는 기대 논문 하나."""
    missing = [k for k in row["keys"] if k not in row.get("_in_db", set())]
    return (missing or row["keys"])[-1]


def pools(question: str, evidence: list[str], existing: set[str]) -> dict:
    from rag import auto_fetch as af
    from rag.arxiv_source import search_openalex, search_papers

    queries = af.make_search_queries(question)
    base = []
    for q in queries:
        try:
            base += search_papers(q, settings.auto_fetch_candidates)
        except Exception as e:  # noqa: BLE001
            print(f"    검색 실패: {e}")
        try:
            base += search_openalex(q, 50, by_citations=True)
        except Exception as e:  # noqa: BLE001
            print(f"    인용 순 검색 실패: {e}")
    guessed = af.guess_original_title(question)
    titled = af.title_candidates(guessed) if guessed else []
    try:
        common = af.common_ref_candidates(evidence)
    except Exception as e:  # noqa: BLE001
        print(f"    공통 참고문헌 실패: {e}")
        common = []
    keep = lambda ps: [p for p in ps if p.pid not in existing]  # noqa: E731
    return {"queries": queries, "guessed": guessed, "base": keep(base), "title": keep(titled), "common": keep(common)}


def choose(question: str, pool: dict, use_title: bool, use_common: bool) -> list[tuple]:
    from rag import auto_fetch as af
    from rag.llm import get_reranker

    cands: dict = {}
    for src, on in (("base", True), ("title", use_title), ("common", use_common)):
        if not on:
            continue
        for p in pool[src]:
            p = dataclasses.replace(p, title_match=p.title_match and use_title)
            if p.pid in cands:
                c = cands[p.pid]
                c.cited_by, c.title_match = max(c.cited_by, p.cited_by), c.title_match or p.title_match
            else:
                cands[p.pid] = p
    if not cands:
        return []
    items = list(cands.values())
    scores = get_reranker().predict([(question, f"{r.title}\n{r.summary}") for r in items])
    ranked = sorted(zip(items, (float(s) for s in scores)), key=lambda x: x[1], reverse=True)
    return af.select_papers(ranked)


def run() -> None:
    from rag.vectorstore import load_vectorstore

    if settings.llm_provider != "local":
        raise SystemExit("로컬 LLM(LLM_PROVIDER=local)에서만 실행합니다.")
    settings.local_thinking_light = "off"
    settings.exclude_auto_papers = False
    existing = {(m or {}).get("source") for m in load_vectorstore().get(include=["metadatas"])["metadatas"]}
    rows = [json.loads(l) for l in RECORD.read_text(encoding="utf-8").splitlines() if l.strip()]
    need = [r for r in rows if r["should_fetch"]]
    done = set()
    if OUT.exists():
        done = {json.loads(l)["id"] for l in OUT.read_text(encoding="utf-8").splitlines() if l.strip()}
    print(f"수집 필요 질문 {len(need)}개 (기록됨 {len(done)}개 건너뜀)")
    for r in need:
        if r["id"] in done:
            continue
        r["_in_db"] = existing
        key = target(r)
        q = r["standalone"]
        pool = pools(q, [e["source"] for e in r["evidence"]], existing)
        out = {"id": r["id"], "question": q, "key": key, "guessed": pool["guessed"], "queries": pool["queries"],
               "key_in_pool": {k: key in {p.pid for p in pool[k]} for k in ("base", "title", "common")}, "chosen": {}}
        for name, (t, c) in CONFIGS.items():
            ch = choose(q, pool, t, c)
            out["chosen"][name] = [[p.pid, p.title[:80], round(s, 3)] for p, s in ch]
        with OUT.open("a", encoding="utf-8") as f:
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
        marks = " ".join("O" if key in [x[0] for x in out["chosen"][n]] else "-" for n in CONFIGS)
        print(f"  {r['id']:<12} {marks}  추정={str(pool['guessed'])[:50]}", flush=True)
    report()


def report() -> None:
    rows = [json.loads(l) for l in OUT.read_text(encoding="utf-8").splitlines() if l.strip()]
    print(f"\n원 논문 선정 ({len(rows)}개 질문)")
    for name in CONFIGS:
        hit = sum(r["key"] in [x[0] for x in r["chosen"][name]] for r in rows)
        empty = sum(not r["chosen"][name] for r in rows)
        print(f"  {name:<12} {hit:>2}/{len(rows)} ({hit / len(rows):.0%})  선정 없음 {empty}")
    print(f"  후보에 원 논문 포함: 기존 {sum(r['key_in_pool']['base'] for r in rows)}, "
          f"제목 추정 {sum(r['key_in_pool']['title'] for r in rows)}, 공통 참고문헌 {sum(r['key_in_pool']['common'] for r in rows)}")
    wrong = [r for r in rows if r["guessed"] and not r["key_in_pool"]["title"]]
    print(f"  제목 추정 {sum(bool(r['guessed']) for r in rows)}개 중 원 논문을 못 찾은 추정 {len(wrong)}개")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    report() if args.report else run()


if __name__ == "__main__":
    main()
