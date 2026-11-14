"""
scripts/enrich_visuals.py — 전문이 있는 논문의 표·그림·수식을 근거 청크로 추가(로컬 멀티모달 모델)

    python scripts/enrich_visuals.py --list                 # 대상(전문이 있는 논문)과 처리한 종류
    python scripts/enrich_visuals.py --sources 2004.04906   # 지정 논문만
    python scripts/enrich_visuals.py --auto [--limit 5]     # 자동 수집 논문 중 아직 처리 안 한 것
    python scripts/enrich_visuals.py --all --kinds figure equation   # 전문 논문 전체에 그림·수식만 추가
    python scripts/enrich_visuals.py --retitle               # 기존 청크 앞에 논문 제목 붙이기(LLM 없음)

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
from rag.core.schemas import VISUAL_KINDS  # noqa: E402


from rag.ingest.enrich import done_items as _items  # noqa: E402
from rag.ingest.enrich import done_kinds, enrich, fulltext_papers as papers, titled  # noqa: E402,F401
from rag.ingest.enrich import load_done as _done  # noqa: E402
from rag.ingest.enrich import save_done as _save_done  # noqa: E402


def retitle(vs) -> int:
    """이미 넣은 표·그림·수식 청크에 제목을 붙여 다시 넣는다(LLM 없음, 임베딩만 다시)."""
    from langchain_core.documents import Document

    d = vs.get(include=["documents", "metadatas"])
    ids, docs = [], []
    for i, text, m in zip(d["ids"], d["documents"], d["metadatas"]):
        m = m or {}
        if m.get("kind") in VISUAL_KINDS and titled(m.get("title", ""), text) != text:
            ids.append(i)
            docs.append(Document(page_content=titled(m["title"], text), metadata=m))
    for k in range(0, len(ids), 100):
        vs.delete(ids[k:k + 100])
        vs.add_documents(docs[k:k + 100])
    return len(ids)


def main() -> None:
    from rag.ingest import visual
    from rag.store.vectorstore import load_vectorstore

    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--sources", nargs="+")
    ap.add_argument("--auto", action="store_true", help="자동 수집 논문 중 아직 처리 안 한 것")
    ap.add_argument("--all", action="store_true", help="전문이 있는 논문 전체(이미 처리한 종류는 건너뜀)")
    ap.add_argument("--kinds", nargs="+", choices=visual.KINDS, help="처리할 종류(기본: 표 + 설정으로 켠 종류)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--retitle", action="store_true", help="기존 표·그림·수식 청크 앞에 논문 제목 붙이기(LLM 없음)")
    ap.add_argument("--no-health", action="store_true", help="끝난 뒤 코퍼스 점검(eval/corpus_health.py) 생략")
    args = ap.parse_args()
    kinds = tuple(args.kinds or visual.default_kinds())
    settings.exclude_auto_papers = False
    vs = load_vectorstore()
    if args.retitle:
        print(f"제목을 붙인 청크 {retitle(vs)}개")
        return
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
            new_items = enrich(src, full[src]["meta"], vs, need, progress=lambda ev, **kw: print(
                f"    page {kw['page']} ({kw['i']}/{kw['n']}) {', '.join(kw['labels'])}", flush=True))
            done[src] = {"kinds": sorted(done_kinds(done.get(src)) | set(need)),
                         "items": _items(done.get(src)) + new_items}
            _save_done(done)
        except Exception as e:  # noqa: BLE001 — 한 편 실패가 나머지를 막지 않게
            print(f"    실패: {type(e).__name__}: {str(e)[:150]}")
        time.sleep(3)
    if todo and not args.no_health:   # 청크가 늘었으니 검색 품질이 유지되는지 확인
        from eval.corpus import run_health_check
        run_health_check(f"표·그림·수식 {len(todo)}편 처리 뒤")


if __name__ == "__main__":
    main()
