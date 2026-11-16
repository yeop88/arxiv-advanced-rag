"""
scripts/backfill_paper_store.py — 이미 벡터 DB 에 있는 논문의 PDF 원문·전문을 원문 저장소(PostgreSQL)에 채운다

원문 저장소가 생기기 전에 수집한 논문은 청크만 있다. 이 스크립트가 arXiv 에서 PDF 를 다시 받아
PDF·전문 텍스트를 저장한다. 이미 PDF 가 저장된 논문은 건너뛰므로 중간에 멈춰도 다시 실행하면 이어서 한다.
LLM 은 쓰지 않는다. 직접 올린 파일(origin=local)은 --local-dir 로 원본 폴더를 알려 줘야 채운다.

실행:
  python scripts/backfill_paper_store.py              # 전체(이미 있는 것은 건너뜀)
  python scripts/backfill_paper_store.py --limit 5    # 시험 삼아 5편
  python scripts/backfill_paper_store.py --stats      # 저장 현황만
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.store import paperstore  # noqa: E402


def print_stats() -> None:
    s = paperstore.stats()
    print(f"논문 {s['papers']}편 · 전문 {s['fulltext']}편(초록만 {s['texts'] - s['fulltext']}편) · "
          f"전문 {s['text_chars'] / 1e6:.1f}M자 · PDF {s['pdfs']}편 {s['pdf_bytes'] / 2**20:.0f}MB · 출처별 {s['by_origin']}")


def corpus_papers() -> dict[str, dict]:
    """벡터 DB 의 논문별 메타데이터(첫 본문 청크 기준)."""
    from rag.store.vectorstore import load_vectorstore

    papers: dict[str, dict] = {}
    for m in load_vectorstore().get()["metadatas"]:
        src = (m or {}).get("source")
        if src and (src not in papers or m.get("kind", "body") == "body" and papers[src].get("kind", "body") != "body"):
            papers[src] = m
    return papers


def main() -> None:
    ap = argparse.ArgumentParser(description="벡터 DB 논문의 PDF·전문을 원문 저장소에 채우기")
    ap.add_argument("--limit", type=int, default=0, help="최대 처리 편수(0 = 전부)")
    ap.add_argument("--sleep", type=float, default=1.0, help="arXiv 요청 사이 대기(초)")
    ap.add_argument("--force", action="store_true", help="이미 저장된 논문도 다시 받기")
    ap.add_argument("--local-dir", help="직접 올린 파일(origin=local)의 원본 폴더")
    ap.add_argument("--stats", action="store_true", help="저장 현황만 출력")
    args = ap.parse_args()

    if not paperstore.enabled():
        sys.exit("DATABASE_URL 이 설정되지 않았습니다(.env). README 의 '원문 저장소' 절을 참고하세요.")
    if args.stats:
        print_stats()
        return

    from rag.ingest.arxiv_source import http_session, pdf_to_text

    papers = corpus_papers()
    done = set() if args.force else paperstore.stored_sources(with_pdf=True) | paperstore.stored_sources()
    todo = [(s, m) for s, m in sorted(papers.items()) if s not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"벡터 DB 논문 {len(papers)}편 · 이미 저장 {sum(s in done for s in papers)}편 · 이번에 {len(todo)}편")

    session = http_session()
    ok = fail = skipped = 0
    t0 = time.perf_counter()
    for i, (src, meta) in enumerate(todo, 1):
        try:
            if meta.get("origin") == "local":
                path = Path(args.local_dir) / src if args.local_dir else None
                if not path or not path.exists():
                    skipped += 1
                    print(f"[{i}/{len(todo)}] {src}: 직접 올린 파일 — 원본 폴더(--local-dir)가 없어 건너뜀")
                    continue
                is_pdf = path.suffix.lower() == ".pdf"
                pdf = path.read_bytes() if is_pdf else None
                text = pdf_to_text(pdf) if is_pdf else path.read_text(encoding="utf-8", errors="ignore")
                saved = paperstore.save(meta, text, True, pdf=pdf, extractor="pypdf" if is_pdf else "text")
            else:
                url = f"https://arxiv.org/pdf/{src}"
                resp = session.get(url, timeout=90)
                resp.raise_for_status()
                text = pdf_to_text(resp.content)
                saved = paperstore.save(meta, text, bool(text.strip()), pdf=resp.content, pdf_url=url)
                time.sleep(args.sleep)
            if not saved:
                raise RuntimeError("저장 실패(DB 연결 확인)")
            ok += 1
            print(f"[{i}/{len(todo)}] {src}: 전문 {len(text):,}자 저장 — {meta.get('title', '')[:60]}", flush=True)
        except Exception as e:  # noqa: BLE001 — 한 편 실패가 나머지를 막지 않게
            fail += 1
            print(f"[{i}/{len(todo)}] {src}: 실패 {type(e).__name__}: {str(e)[:120]}", flush=True)
    print(f"\n완료: 저장 {ok} · 실패 {fail} · 건너뜀 {skipped} ({time.perf_counter() - t0:.0f}초)")
    print_stats()


if __name__ == "__main__":
    main()
