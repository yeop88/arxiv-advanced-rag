"""
rag/ingest/enrich.py — 전문 논문의 표·그림·수식을 근거 청크로 추가하는 공용 부분

scripts/enrich_visuals.py(배치)와 챗봇 서버의 백그라운드 처리(server/enricher.py)가 같이 쓴다.
  - 처리 기록(VISUAL_DONE_PATH): 논문마다 처리한 종류·항목 — 이미 한 종류는 다시 하지 않는다
  - enrich(): PDF 를 받아 rag/ingest/visual.extract 로 항목을 뽑고 '[논문 제목] …' 청크로 DB 에 추가
  - guard: 페이지마다 로컬 멀티모달 모델을 부르기 직전에 들어가는 context manager(서버는 LLM 작업 잠금)
"""
from __future__ import annotations

import json
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Callable, ContextManager

from config import settings
from rag.core.schemas import VISUAL_KINDS

Guard = Callable[[], ContextManager]


def load_done() -> dict:
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


def done_items(entry) -> list[dict]:
    return entry.get("items", []) if isinstance(entry, dict) else (entry or [])


def save_done(d: dict) -> None:
    p = Path(settings.visual_done_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(p)


def mark_done(src: str, kinds: tuple[str, ...], items: list[dict]) -> None:
    d = load_done()
    d[src] = {"kinds": sorted(done_kinds(d.get(src)) | set(kinds)), "items": done_items(d.get(src)) + items}
    save_done(d)


def titled(title: str, text: str) -> str:
    """청크 앞에 논문 제목을 붙인다 — 짧은 수식·표 청크만으로는 어느 논문 것인지 몰라 "Mamba 논문에서 …" 같은 질문과
    맞지 않았다(Mamba 이산화 수식 리랭크 0.02, §5.38)."""
    prefix = f"[{title}] "
    return text if not title or text.startswith(prefix) else prefix + text


def fulltext_papers(vs) -> dict[str, dict]:
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


def enrich(src: str, meta: dict, vs, kinds: tuple[str, ...], progress=None, guard: Guard | None = None,
           add: Callable[[list], None] | None = None) -> list[dict]:
    """논문 하나의 표·그림·수식을 청크로 추가하고 처리 항목 요약을 돌려준다(기록은 호출한 쪽이 mark_done).
    add 를 주면 DB 추가를 그 함수로(서버는 잠금 안에서 추가하고 BM25 를 다시 만든다)."""
    from langchain_core.documents import Document

    from rag.ingest import visual
    from rag.ingest.arxiv_source import http_session

    from rag.store import paperstore

    pdf = paperstore.get_pdf(src)   # 원문 저장소에 있으면 다시 받지 않는다
    if pdf is None:
        resp = http_session().get(f"https://arxiv.org/pdf/{src}", timeout=90)
        resp.raise_for_status()
        pdf = resp.content
    t0 = time.perf_counter()
    items = visual.extract(pdf, kinds=kinds, progress=progress, guard=guard or nullcontext)
    keep = {k: meta[k] for k in ("source", "title", "url", "published", "origin", "fetched_at", "fetch_query") if k in meta}
    docs = [Document(page_content=titled(meta.get("title", ""), it.text),
                     metadata={**keep, "section": it.label, "kind": it.kind, "page": it.page}) for it in items]
    if docs:
        (add or vs.add_documents)(docs)
    print(f"    → {src}: 표 {sum(i.kind == 'table' for i in items)}개·그림 {sum(i.kind == 'figure' for i in items)}개·"
          f"수식 {sum(i.kind == 'equation' for i in items)}개 추가 ({time.perf_counter() - t0:.0f}초)", flush=True)
    return [{"label": i.label, "kind": i.kind, "page": i.page, "number_match": i.number_match} for i in items]
