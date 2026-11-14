"""
scripts/enrich_visuals.py — 전문이 있는 논문의 표·그림·수식을 근거 청크로 추가(로컬 멀티모달 모델)

    python scripts/enrich_visuals.py --list                 # 대상(전문이 있는 논문)과 처리한 종류
    python scripts/enrich_visuals.py --sources 2004.04906   # 지정 논문만
    python scripts/enrich_visuals.py --auto [--limit 5]     # 자동 수집 논문 중 아직 처리 안 한 것
    python scripts/enrich_visuals.py --all --kinds figure equation   # 전문 논문 전체에 그림·수식만 추가

--kinds 를 빼면 표 + 설정(VISUAL_FIGURES·VISUAL_EQUATIONS)으로 켠 종류. 논문마다 처리한 종류를 기록해, 이미 한 종류는 다시 하지 않는다.

PDF 는 arxiv.org 에서 받는다(검색 API 아님, 요청 사이 3초). 캡션 페이지당 로컬 모델 1회(약 30~50초), 논문당 2~8분.
추가 청크: section="Table N"/"Figure N"/"Equation N", kind=table|figure|equation, 원 논문의 origin·제목을 그대로 따른다.
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
from rag.schemas import VISUAL_KINDS  # noqa: E402


def _done() -> dict:
    p = Path(settings.visual_done_path)
    try:
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    except ValueError:
        return {}


def done_kinds(entry) -> set[str]:
    """처리 기록에서 처리한 종류. 예전 기록(항목 목록)은 표만 처리한 것."""
    if isinstance(entry, dict):
        return set(entry.get("kinds", []))
    return {"table"} if entry is not None else set()


def _items(entry) -> list[dict]:
    return entry.get("items", []) if isinstance(entry, dict) else (entry or [])


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
        if not src or m.get("kind") in VISUAL_KINDS:
            continue
        p = out.setdefault(src, {"meta": m, "chunks": 0, "sections": set()})
        p["chunks"] += 1
        if m.get("section"):
            p["sections"].add(m["section"])
    return {k: v for k, v in out.items() if v["chunks"] >= 8 and len(v["sections"]) >= 3}


def enrich(src: str, meta: dict, vs, kinds: tuple[str, ...]) -> list[dict]:
    from langchain_core.documents import Document

    from rag import visual
    from rag.arxiv_source import http_session

    resp = http_session().get(f"https://arxiv.org/pdf/{src}", timeout=90)
    resp.raise_for_status()
    t0 = time.perf_counter()
    items = visual.extract(resp.content, kinds=kinds, progress=lambda ev, **kw: print(
        f"    page {kw['page']} ({kw['i']}/{kw['n']}) {', '.join(kw['labels'])}", flush=True))
    keep = {k: meta[k] for k in ("source", "title", "url", "published", "origin", "fetched_at", "fetch_query") if k in meta}
    docs = [Document(page_content=it.text, metadata={**keep, "section": it.label, "kind": it.kind, "page": it.page})
            for it in items]
    if docs:
        vs.add_documents(docs)
    print(f"    → 표 {sum(i.kind == 'table' for i in items)}개·그림 {sum(i.kind == 'figure' for i in items)}개·"
          f"수식 {sum(i.kind == 'equation' for i in items)}개 추가 "
          f"({time.perf_counter() - t0:.0f}초)", flush=True)
    return [{"label": i.label, "kind": i.kind, "page": i.page, "number_match": i.number_match} for i in items]


def main() -> None:
    from rag import visual
    from rag.vectorstore import load_vectorstore

    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--sources", nargs="+")
    ap.add_argument("--auto", action="store_true", help="자동 수집 논문 중 아직 처리 안 한 것")
    ap.add_argument("--all", action="store_true", help="전문이 있는 논문 전체(이미 처리한 종류는 건너뜀)")
    ap.add_argument("--kinds", nargs="+", choices=visual.KINDS, help="처리할 종류(기본: 표 + 설정으로 켠 종류)")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    kinds = tuple(args.kinds or visual.default_kinds())
    settings.exclude_auto_papers = False
    vs = load_vectorstore()
    full = papers(vs)
    done = _done()
    if args.list or not (args.sources or args.auto or args.all):
        for src, p in sorted(full.items(), key=lambda x: x[1]["meta"].get("origin", "")):
            mark = ",".join(sorted(done_kinds(done.get(src)))) or "-"
            print(f"  {src:<14} {p['meta'].get('origin') or 'manual':<6} 청크 {p['chunks']:>4} {mark:<22} {p['meta'].get('title', '')[:50]}")
        print(f"전문 논문 {len(full)}편, 처리함 {sum(s in done for s in full)}편")
        return
    if settings.llm_provider != "local":
        raise SystemExit("로컬 LLM(LLM_PROVIDER=local)에서만 실행합니다.")
    if args.sources:
        todo = args.sources
    else:
        todo = [s for s, p in full.items() if (args.all or p["meta"].get("origin") == "auto")
                and set(kinds) - done_kinds(done.get(s))]
    if args.limit:
        todo = todo[: args.limit]
    for i, src in enumerate(todo, 1):
        if src not in full:
            print(f"[{i}/{len(todo)}] {src}: 전문이 있는 논문이 아니라 건너뜀")
            continue
        need = tuple(k for k in kinds if k not in done_kinds(done.get(src)))
        if not need:
            print(f"[{i}/{len(todo)}] {src}: {','.join(kinds)} 이미 처리함")
            continue
        print(f"[{i}/{len(todo)}] {src} [{','.join(need)}] {full[src]['meta'].get('title', '')[:55]}", flush=True)
        try:
            new_items = enrich(src, full[src]["meta"], vs, need)
            done[src] = {"kinds": sorted(done_kinds(done.get(src)) | set(need)),
                         "items": _items(done.get(src)) + new_items}
            _save_done(done)
        except Exception as e:  # noqa: BLE001 — 한 편 실패가 나머지를 막지 않게
            print(f"    실패: {type(e).__name__}: {str(e)[:150]}")
        time.sleep(3)


if __name__ == "__main__":
    main()
