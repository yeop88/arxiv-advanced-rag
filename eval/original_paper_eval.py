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
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import settings  # noqa: E402

RECORD = Path(__file__).resolve().parent / "results" / "fetch_signal_record.jsonl"
OUT = Path(__file__).resolve().parent / "results" / "original_paper_eval.jsonl"
# 확인용 세트: 개선(프롬프트 예시·기준 조정)에 쓰지 않은 새 주제, DB 에 없는 원 논문. 근거 기록이 없어 공통 참고문헌은 해당 없음
HELDOUT = [
    ("U-Net은 의료 영상 분할에서 어떤 구조를 쓰나요?", "1505.04597"),
    ("DQN은 아타리 게임을 어떻게 학습했나요?", "1312.5602"),
    ("WaveNet은 원시 오디오 파형을 어떻게 생성하나요?", "1609.03499"),
    ("BLIP-2는 고정된 이미지 인코더와 LLM을 어떻게 연결하나요?", "2301.12597"),
    ("ControlNet은 확산 모델에 조건을 어떻게 추가하나요?", "2302.05543"),
    ("Swin Transformer의 이동 윈도우 방식은 무엇인가요?", "2103.14030"),
    ("배치 정규화는 내부 공변량 변화를 어떻게 줄이나요?", "1502.03167"),
    ("EfficientNet의 복합 스케일링은 어떻게 동작하나요?", "1905.11946"),
    ("Faster R-CNN의 영역 제안 네트워크는 어떤 역할을 하나요?", "1506.01497"),
    ("DETR은 객체 탐지를 집합 예측으로 어떻게 푸나요?", "2005.12872"),
    ("BART는 어떤 노이즈 방식으로 사전학습하나요?", "1910.13461"),
    ("FlashAttention-2는 이전 버전보다 병렬화를 어떻게 개선했나요?", "2307.08691"),
]
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
        if settings.auto_fetch_citation_search and settings.openalex_enabled:
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


def choose(question: str, pool: dict, use_title: bool, use_common: bool, key: str | None = None) -> tuple:
    from rag import auto_fetch as af
    from rag.llm import get_reranker

    cands: dict = {}
    for src, on in (("base", True), ("title", use_title), ("common", use_common)):
        if not on:
            continue
        for p in pool[src]:
            p = dataclasses.replace(p, title_match=p.title_match if use_title else 0.0)
            if p.pid in cands:
                c = cands[p.pid]
                c.cited_by, c.title_match = max(c.cited_by, p.cited_by), max(c.title_match, p.title_match)
            else:
                cands[p.pid] = p
    if not cands:
        return [], None
    items = list(cands.values())
    scores = get_reranker().predict([(question, f"{r.title}\n{r.summary}") for r in items])
    ranked = sorted(zip(items, (float(s) for s in scores)), key=lambda x: x[1], reverse=True)
    key_score = next((round(s, 3) for p, s in ranked if p.pid == key), None)
    return af.select_papers(ranked), key_score


def run(heldout: bool = False) -> None:
    global OUT
    from rag.vectorstore import load_vectorstore

    if settings.llm_provider != "local":
        raise SystemExit("로컬 LLM(LLM_PROVIDER=local)에서만 실행합니다.")
    settings.local_thinking_light = "off"
    settings.exclude_auto_papers = False
    existing = {(m or {}).get("source") for m in load_vectorstore().get(include=["metadatas"])["metadatas"]}
    if heldout:
        OUT = OUT.with_name("original_paper_eval_heldout.jsonl")
        need = [{"id": f"held{i:02d}", "standalone": q, "keys": [k], "evidence": []}
                for i, (q, k) in enumerate(HELDOUT, 1)]
    else:
        rows = [json.loads(l) for l in RECORD.read_text(encoding="utf-8").splitlines() if l.strip()]
        need = [r for r in rows if r["should_fetch"]]
    done = set()
    if OUT.exists():
        done = {json.loads(l)["id"] for l in OUT.read_text(encoding="utf-8").splitlines() if l.strip()}
    print(f"수집 필요 질문 {len(need)}개 (기록됨 {len(done)}개 건너뜀), OpenAlex {'켜짐' if settings.openalex_enabled else '꺼짐'}")
    for r in need:
        if r["id"] in done:
            continue
        r["_in_db"] = existing
        key = target(r)
        q = r["standalone"]
        pool = pools(q, [e["source"] for e in r["evidence"]], existing)
        out = {"id": r["id"], "question": q, "key": key, "guessed": pool["guessed"], "queries": pool["queries"],
               "key_in_pool": {k: key in {p.pid for p in pool[k]} for k in ("base", "title", "common")},
               "n_pool": {k: len(pool[k]) for k in ("base", "title", "common")}, "chosen": {}, "key_score": {}}
        if not pool["base"]:  # 검색이 막혀(429 등) 후보가 없으면 기록하지 않고 중단 — 같은 명령으로 이어서 실행
            raise SystemExit(f"[{r['id']}] 검색 후보 0개(요청 제한 의심) — 기록하지 않고 중단합니다.")
        for name, (t, c) in CONFIGS.items():
            ch, ks = choose(q, pool, t, c, key)
            out["chosen"][name] = [[p.pid, p.title[:80], round(s, 3)] for p, s in ch]
            out["key_score"][name] = ks
        with OUT.open("a", encoding="utf-8") as f:
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
        marks = " ".join("O" if key in [x[0] for x in out["chosen"][n]] else "-" for n in CONFIGS)
        print(f"  {r['id']:<12} {marks}  추정={str(pool['guessed'])[:50]}", flush=True)
        time.sleep(10)  # 회사망 공용 IP 에서 arXiv 요청 제한을 피하려고 질문 사이를 넉넉히
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
    ap.add_argument("--heldout", action="store_true", help="개선에 쓰지 않은 확인용 12문항")
    args = ap.parse_args()
    if args.report:
        global OUT
        if args.heldout:
            OUT = OUT.with_name("original_paper_eval_heldout.jsonl")
        report()
    else:
        run(args.heldout)


if __name__ == "__main__":
    main()
