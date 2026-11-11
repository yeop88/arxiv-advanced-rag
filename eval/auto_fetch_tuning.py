"""
eval/auto_fetch_tuning.py — 자동 수집 논문이 다른 질문을 흐리는 문제(이슈 13)의 해결 방법 비교

비교하는 방법:
  1) 관련도 기준 상향 (AUTO_FETCH_MIN_SCORE)
  2) 질문당 편수 축소 (AUTO_FETCH_MAX_PAPERS)
  3) 수집 논문의 리랭크 가중치 낮추기 (AUTO_RERANK_WEIGHT)

1·2 는 '어떤 논문을 남기느냐'의 차이이므로, 현재 설정(0.3·3편)으로 한 번 수집한 뒤 조건에 맞지 않는
논문만 숨겨 재현한다(설정마다 다시 수집하지 않음). 모든 측정은 질의변환 없이 하이브리드 검색으로 → LLM 0회.

양쪽을 함께 본다:
  - 피해: 평가셋(코퍼스 안 질문)의 recall@k·MRR, 기준 코퍼스와 결과가 달라진 문항 수
  - 효과: 코퍼스 밖 질문에서 원 논문이 남았는지, 근거 상위 k 에 올라오는지, 최고 점수

사용:
    python eval/auto_fetch_tuning.py --collect   # 코퍼스 밖 질문으로 실제 수집(경량 LLM 사용)
    python eval/auto_fetch_tuning.py --select    # 후보 수를 넓혀 다시 선정, 새 논문만 추가(경량 LLM 사용)
    python eval/auto_fetch_tuning.py             # 설정별 측정(이어서 실행 가능)
    python eval/auto_fetch_tuning.py --fresh     # 기록을 지우고 처음부터
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import settings  # noqa: E402

settings.auto_fetch = False
settings.query_transform_mode = "none"
settings.use_hybrid = True
settings.exclude_auto_papers = False

from eval.dataset import load_eval_set  # noqa: E402
from eval.retrieval_metrics import recall_at_k, reciprocal_rank  # noqa: E402
from rag.auto_fetch import is_back_matter  # noqa: E402
from rag.stage2_rerank import retrieve_and_rerank  # noqa: E402
from rag.vectorstore import ExcludeOriginStore, load_vectorstore  # noqa: E402

OUT = Path(__file__).resolve().parent / "results" / "auto_fetch_tuning.json"

# (코퍼스 밖 질문, 원 논문 arXiv ID). AlphaFold2 는 arXiv 에 없어 원 논문 지표에서 뺀다.
OOC = [
    ("Kolmogorov-Arnold Network(KAN)은 기존 MLP와 구조가 어떻게 다른가요?", "2404.19756"),
    ("FlashAttention은 GPU 메모리 접근을 줄여 어텐션 계산을 어떻게 빠르게 하나요?", "2205.14135"),
    ("DPO(Direct Preference Optimization)는 보상 모델 없이 어떻게 선호 학습을 하나요?", "2305.18290"),
    ("AlphaFold는 아미노산 서열에서 단백질 구조를 어떻게 예측하나요?", None),
    ("NeRF는 여러 장의 사진으로 새로운 시점의 이미지를 어떻게 합성하나요?", "2003.08934"),
    ("Diffusion Policy는 로봇 행동을 확산 모델로 어떻게 생성하나요?", "2303.04137"),
    ("GraphCast는 그래프 신경망으로 중기 날씨 예보를 어떻게 하나요?", "2212.12794"),
    ("Mamba의 선택적 상태 공간 모델은 트랜스포머와 어떻게 다른가요?", "2312.00752"),
]

VARIANTS = [
    ("기준(자동 수집 제외)", {"keep": "none"}),
    ("현재(0.3·3편)", {"keep": "all"}),
    ("방법1 기준 0.5", {"min_score": 0.5}),
    ("방법1 기준 0.7", {"min_score": 0.7}),
    ("방법2 1편", {"max_papers": 1}),
    ("방법2 2편", {"max_papers": 2}),
    ("방법3 가중치 0.7", {"keep": "all", "weight": 0.7}),
    ("방법3 가중치 0.5", {"keep": "all", "weight": 0.5}),
    # 아래는 후보 수를 20개로 넓혀 다시 고른 논문 기준(--select 로 만든 선정 결과 사용)
    ("후보20", {"selection": True}),
    ("후보20+참고문헌 제외", {"selection": True, "no_refs": True}),
    ("후보20+참고문헌 제외+30청크", {"selection": True, "no_refs": True, "cap": 30}),
    ("후보20+참고문헌 제외+가중치0.7", {"selection": True, "no_refs": True, "weight": 0.7}),
]

SELECTION = OUT.with_name("auto_fetch_selection.json")


def auto_papers(vs) -> dict[str, dict]:
    """자동 수집 논문 → {query, title, score}. 점수 기록이 없는 옛 수집분은 None."""
    papers: dict[str, dict] = {}
    for m in vs.get().get("metadatas", []):
        m = m or {}
        if m.get("origin") == "auto":
            papers[m["source"]] = {"query": m.get("fetch_query", ""), "title": m.get("title", ""),
                                   "score": m.get("fetch_score")}
    return papers


def backfill_scores(papers: dict[str, dict]) -> None:
    """점수 없는 논문: 초록을 다시 찾아 수집 때와 같은 방식 (질문, 제목+초록) 으로 채점."""
    from rag.arxiv_source import fetch_arxiv_by_ids
    from rag.llm import get_reranker

    missing = [pid for pid, p in papers.items() if p["score"] is None]
    if not missing:
        return
    found = fetch_arxiv_by_ids(missing)
    for pid in missing:
        if pid not in found:
            raise SystemExit(f"[{pid}] 초록을 찾지 못해 채점할 수 없습니다: {papers[pid]['title']}")
        p, f = papers[pid], found[pid]
        p["score"] = float(get_reranker().predict([(p["query"], f"{f.title}\n{f.summary}")])[0])


def kept_sources(papers: dict[str, dict], spec: dict) -> set[str]:
    if spec.get("keep") == "none":
        return set()
    if spec.get("keep") == "all":
        return set(papers)
    if "min_score" in spec:
        return {pid for pid, p in papers.items() if p["score"] >= spec["min_score"]}
    by_q: dict[str, list[str]] = defaultdict(list)
    for pid, p in papers.items():
        by_q[p["query"]].append(pid)
    out: set[str] = set()
    for pids in by_q.values():
        out |= set(sorted(pids, key=lambda i: papers[i]["score"], reverse=True)[: spec["max_papers"]])
    return out


def kept_ids(vs, sources: set[str], spec: dict) -> set[str]:
    """남길 논문의 청크 ID. no_refs 면 참고문헌·부록을 빼고, cap 이면 논문당 앞에서부터 그 개수까지."""
    raw = vs.get()
    count: dict[str, int] = defaultdict(int)
    out: set[str] = set()
    for i, m in zip(raw["ids"], raw["metadatas"]):
        m = m or {}
        if m.get("origin") != "auto" or m.get("source") not in sources:
            continue
        if spec.get("no_refs") and is_back_matter(m.get("section")):
            continue
        count[m["source"]] += 1
        if not spec.get("cap") or count[m["source"]] <= spec["cap"]:
            out.add(i)
    return out


def select_wider(vs, n: int) -> None:
    """후보를 n 개로 넓혀 수집 때와 같은 기준으로 다시 고른다. 기존 자동 수집분을 지우지 않고 새 논문만 추가."""
    from rag.arxiv_source import http_session, result_to_chunks, search_papers
    from rag.auto_fetch import make_search_queries
    from rag.llm import get_reranker

    settings.local_thinking_light = "off"
    metas = [m or {} for m in vs.get()["metadatas"]]
    manual = {m.get("source") for m in metas if m.get("origin") != "auto"}
    auto = {m.get("source") for m in metas if m.get("origin") == "auto"}
    selection: dict[str, list[dict]] = {}
    for q, key in OOC:
        cands: dict = {}
        for s in make_search_queries(q):
            time.sleep(4)  # arXiv API 예의(요청 간격)
            for p in search_papers(s, n, warn=print):
                if p.pid not in manual:
                    cands.setdefault(p.pid, p)
        items = list(cands.values())
        scores = get_reranker().predict([(q, f"{p.title}\n{p.summary}") for p in items])
        ranked = sorted(zip(items, map(float, scores)), key=lambda x: x[1], reverse=True)
        chosen = [x for x in ranked if x[1] >= settings.auto_fetch_min_score][: settings.auto_fetch_max_papers]
        print(f"■ {q} (후보 {len(items)}편, 원 논문 {key}, 후보 포함 {key in cands})")
        for p, sc in chosen:
            new = p.pid not in auto
            if new:
                chunks, _ = result_to_chunks(p, http_session(), True, extra_meta={
                    "origin": "auto", "fetched_at": time.strftime("%Y-%m-%d"), "fetch_query": q[:200],
                    "fetch_score": round(sc, 4)})
                vs.add_documents(chunks)
                auto.add(p.pid)
            print(f"    [{p.pid}] {sc:.3f} {p.title[:64]}{' (새로 추가)' if new else ''}")
        selection[q] = [{"pid": p.pid, "title": p.title, "score": sc} for p, sc in chosen]
    SELECTION.write_text(json.dumps(selection, ensure_ascii=False, indent=1), encoding="utf-8")


def measure(vs, records: list[dict]) -> dict:
    k = settings.top_k_rerank
    rows = {}
    for r in records:
        _, rr = retrieve_and_rerank(vs, r["question"], [r["question"]])
        exp = r.get("expected_sources") or []
        rows[r["id"]] = {"recall": recall_at_k(rr, exp, k) if exp else None,
                         "mrr": reciprocal_rank(rr, exp) if exp else None,
                         "final": [c.id for c in rr], "auto": sum(c.origin == "auto" for c in rr)}
    ooc = []
    for q, key in OOC:
        _, rr = retrieve_and_rerank(vs, q, [q])
        ooc.append({"question": q, "key": key, "top": max((c.rerank_score or 0.0 for c in rr), default=0.0),
                    "auto": sum(c.origin == "auto" for c in rr),
                    "key_in_topk": (key in [c.source for c in rr]) if key else None})
    return {"rows": rows, "ooc": ooc}


def collect(vs) -> None:
    from rag.auto_fetch import fetch_and_ingest

    settings.local_thinking_light = "off"
    print(f"수집 설정: 기준 {settings.auto_fetch_min_score}, 최대 {settings.auto_fetch_max_papers}편")
    done = {p["query"] for p in auto_papers(vs).values()}
    for q, key in OOC:
        if q[:200] in done:
            print(f"- 이미 수집됨: {q}")
            continue
        t0 = time.perf_counter()
        added = fetch_and_ingest(q, vs, lambda ev, **kw: print(f"    {kw['message']}") if ev == "fetch_warn" else None)
        print(f"■ {q} ({time.perf_counter() - t0:.0f}초, {len(added)}편, 원 논문 {key})")
        for a in added:
            print(f"    [{a['source']}] {a['score']:.3f} {a['title'][:70]} | {a['chunks']}청크 전문={a['fulltext']}")


def report(results: dict, n_records: int) -> None:
    k = settings.top_k_rerank
    base = results[VARIANTS[0][0]]

    def avg(xs):
        xs = [x for x in xs if x is not None]
        return sum(xs) / len(xs) if xs else 0.0

    print(f"\n| 설정 | 남긴 논문 | recall@{k} | MRR | 결과 바뀐 문항(/{n_records}) | 수집 논문이 낀 문항 "
          f"| 원 논문 유지 | 원 논문 top-{k} | 코퍼스 밖 평균 최고 점수 |")
    print("|---|---|---|---|---|---|---|---|---|")
    for name, _ in VARIANTS:
        if name not in results:
            continue
        r = results[name]
        rows = r["rows"]
        changed = sum(rows[i]["final"] != base["rows"][i]["final"] for i in rows)
        with_auto = sum(rows[i]["auto"] > 0 for i in rows)
        keys = [o for o in r["ooc"] if o["key"]]
        kept = sum(o["key"] in r["kept"] for o in keys)
        print(f"| {name} | {len(r['kept'])} | {avg([v['recall'] for v in rows.values()]):.4f} "
              f"| {avg([v['mrr'] for v in rows.values()]):.4f} | {changed} | {with_auto} "
              f"| {kept}/{len(keys)} | {sum(bool(o['key_in_topk']) for o in keys)}/{len(keys)} "
              f"| {avg([o['top'] for o in r['ooc']]):.3f} |")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--collect", action="store_true", help="코퍼스 밖 질문으로 실제 수집")
    ap.add_argument("--select", action="store_true", help="후보를 넓혀 다시 선정하고 새 논문만 추가")
    ap.add_argument("--fresh", action="store_true", help="기록을 지우고 처음부터")
    args = ap.parse_args()

    vs = load_vectorstore()
    if args.collect:
        collect(vs)
        return
    if args.select:
        select_wider(vs, settings.auto_fetch_candidates)
        return

    papers = auto_papers(vs)
    backfill_scores(papers)
    print(f"자동 수집 논문 {len(papers)}편")
    for pid, p in sorted(papers.items(), key=lambda kv: (kv[1]["query"], -kv[1]["score"])):
        print(f"  [{pid}] {p['score']:.3f} {p['title'][:60]} | {p['query'][:24]}")

    records = load_eval_set()
    results = {} if args.fresh or not OUT.exists() else json.loads(OUT.read_text(encoding="utf-8"))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    for name, spec in VARIANTS:
        if name in results:
            print(f"- 기록 있음, 건너뜀: {name}")
            continue
        if spec.get("selection"):
            sel = json.loads(SELECTION.read_text(encoding="utf-8"))
            keep = {x["pid"] for rows in sel.values() for x in rows}
        else:
            keep = kept_sources(papers, spec)
        view = ExcludeOriginStore(vs, keep_ids=kept_ids(vs, keep, spec))
        settings.auto_rerank_weight = spec.get("weight", 1.0)
        t0 = time.perf_counter()
        res = measure(view, records)
        res["kept"] = sorted(keep)
        results[name] = res
        OUT.write_text(json.dumps(results, ensure_ascii=False), encoding="utf-8")
        print(f"▶ {name}: 남긴 논문 {len(keep)}편, {time.perf_counter() - t0:.0f}초", flush=True)
    report(results, len(records))


if __name__ == "__main__":
    main()
