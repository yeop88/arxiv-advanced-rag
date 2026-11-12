"""
scripts/enrich_visuals.py — 전문이 있는 논문의 표·그림을 근거 청크로 추가(로컬 멀티모달 모델)

    python scripts/enrich_visuals.py --list                 # 대상(전문이 있는 논문)과 처리 여부
    python scripts/enrich_visuals.py --sources 2004.04906   # 지정 논문만
    python scripts/enrich_visuals.py --auto [--limit 5]     # 자동 수집 논문 중 아직 처리 안 한 것

PDF 는 arxiv.org 에서 받는다(검색 API 아님, 요청 사이 3초). 캡션 페이지당 로컬 모델 1회(약 30~50초), 논문당 2~8분.
추가 청크: section="Table N"/"Figure N", kind=table|figure, 원 논문의 origin·제목을 그대로 따른다.
처리한 논문은 VISUAL_DONE_PATH 에 기록해 다시 하지 않는다. 챗봇은 재시작하면 새 청크를 BM25 에도 반영한다.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import settings  # noqa: E402


def _done() -> dict:
    p = Path(settings.visual_done_path)
    try:
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    except ValueError:
        return {}


def _save_done(d: dict) -> None:
    p = Path(settings.visual_done_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")


def papers(vs) -> dict[str, dict]:
    """전문이 있는 논문(청크가 여럿이고 섹션 정보가 있는 것): ID → 대표 메타데이터·청크 수."""
    out: dict[str, dict] = {}
    for m in vs.get(include=["metadatas"])["metadatas"]:
        m = m or {}
        src = m.get("source")
        if not src or m.get("kind") in ("table", "figure"):
            continue
        p = out.setdefault(src, {"meta": m, "chunks": 0, "sections": set()})
        p["chunks"] += 1
        if m.get("section"):
            p["sections"].add(m["section"])
    return {k: v for k, v in out.items() if v["chunks"] >= 8 and len(v["sections"]) >= 3}


def enrich(src: str, meta: dict, vs) -> list[dict]:
    from langchain_core.documents import Document

    from rag import visual
    from rag.arxiv_source import http_session

    resp = http_session().get(f"https://arxiv.org/pdf/{src}", timeout=90)
    resp.raise_for_status()
    t0 = time.perf_counter()
    items = visual.extract(resp.content, progress=lambda ev, **kw: print(
        f"    page {kw['page']} ({kw['i']}/{kw['n']}) {', '.join(kw['labels'])}", flush=True))
    keep = {k: meta[k] for k in ("source", "title", "url", "published", "origin", "fetched_at", "fetch_query") if k in meta}
    docs = [Document(page_content=it.text, metadata={**keep, "section": it.label, "kind": it.kind, "page": it.page})
            for it in items]
    if docs:
        vs.add_documents(docs)
    print(f"    → 표 {sum(i.kind == 'table' for i in items)}개·그림 {sum(i.kind == 'figure' for i in items)}개 추가 "
          f"({time.perf_counter() - t0:.0f}초)", flush=True)
    return [{"label": i.label, "kind": i.kind, "page": i.page, "number_match": i.number_match} for i in items]


def main() -> None:
    from rag.vectorstore import load_vectorstore

    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--sources", nargs="+")
    ap.add_argument("--auto", action="store_true", help="자동 수집 논문 중 아직 처리 안 한 것")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    settings.exclude_auto_papers = False
    vs = load_vectorstore()
    full = papers(vs)
    done = _done()
    if args.list or not (args.sources or args.auto):
        for src, p in sorted(full.items(), key=lambda x: x[1]["meta"].get("origin", "")):
            mark = f"처리함({len(done[src])}개)" if src in done else "-"
            print(f"  {src:<14} {p['meta'].get('origin') or 'manual':<6} 청크 {p['chunks']:>4} {mark:<12} {p['meta'].get('title', '')[:50]}")
        print(f"전문 논문 {len(full)}편, 처리함 {sum(s in done for s in full)}편")
        return
    if settings.llm_provider != "local":
        raise SystemExit("로컬 LLM(LLM_PROVIDER=local)에서만 실행합니다.")
    todo = args.sources or [s for s, p in full.items() if p["meta"].get("origin") == "auto" and s not in done]
    if args.limit:
        todo = todo[: args.limit]
    for i, src in enumerate(todo, 1):
        if src not in full:
            print(f"[{i}/{len(todo)}] {src}: 전문이 있는 논문이 아니라 건너뜀")
            continue
        print(f"[{i}/{len(todo)}] {src} {full[src]['meta'].get('title', '')[:60]}", flush=True)
        try:
            done[src] = enrich(src, full[src]["meta"], vs)
            _save_done(done)
        except Exception as e:  # noqa: BLE001 — 한 편 실패가 나머지를 막지 않게
            print(f"    실패: {type(e).__name__}: {str(e)[:150]}")
        time.sleep(3)


if __name__ == "__main__":
    main()
