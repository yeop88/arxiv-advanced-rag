"""
scripts/rebuild_chunks.py — 원문 저장소(PostgreSQL)의 전문으로 청크를 다시 만들어 pgvector 테이블에 넣는다 (LLM 없음)

청크 크기·분할 방식·PDF 추출기를 바꿔 보는 실험(§5.56)용. arXiv 에서 다시 받지 않는다.
  - 본문: 논문마다 저장된 전문(또는 --text-dir 의 다른 추출기 결과 <source>.md/.txt)을 섹션 인식 분할 → bge-m3 임베딩
  - 표·그림·수식 청크와 전문이 없는 논문(직접 올린 파일 등)의 청크: 지금 벡터 DB 에서 벡터째 그대로 복사
  - 자동 수집 논문은 수집 때처럼 참고문헌·부록을 뺀다(AUTO_FETCH_DROP_BACK_MATTER)
  - 메타데이터(제목·저자·origin·수집일 등)는 그 논문의 기존 본문 청크에서 가져온다

  python scripts/rebuild_chunks.py --table rag_chunks_t512 --unit tokens --size 512 --overlap 64
  python scripts/rebuild_chunks.py --table rag_chunks_docling --text-dir storage/extract_docling
비교: python eval/retrieval_variants.py --tag t512 --set vector_backend=pgvector --set pgvector_table=rag_chunks_t512
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import settings  # noqa: E402

KEEP_META = ("source", "title", "authors", "published", "url", "origin", "fetched_at", "fetch_query", "fetch_score")


def main() -> None:
    ap = argparse.ArgumentParser(description="원문 저장소 전문으로 청크 다시 만들기 → pgvector 테이블")
    ap.add_argument("--table", required=True, help="만들 pgvector 테이블(지금 서비스 테이블과 다르게)")
    ap.add_argument("--from-backend", default=None, help="기존 청크를 읽을 벡터 DB(기본: 지금 설정)")
    ap.add_argument("--unit", choices=["chars", "tokens"], default="chars")
    ap.add_argument("--size", type=int, default=1000)
    ap.add_argument("--overlap", type=int, default=150)
    ap.add_argument("--text-dir", help="다른 추출기 결과 폴더(<source>.md 또는 .txt). 없는 논문은 저장 전문을 쓴다")
    ap.add_argument("--drop-back-matter", action="store_true",
                    help="자동 수집 논문뿐 아니라 모든 논문에서 참고문헌·부록 청크를 뺀다")
    a = ap.parse_args()
    if a.table == settings.pgvector_table and settings.vector_backend == "pgvector":
        sys.exit("지금 서비스 중인 테이블에는 쓰지 않습니다. 다른 --table 을 주세요.")

    from langchain_core.documents import Document

    from rag.core.llm import get_embeddings
    from rag.core.schemas import VISUAL_KINDS
    from rag.ingest.auto_fetch import is_back_matter
    from rag.ingest.chunking import section_aware_split
    from rag.store import paperstore
    from rag.store.vectorstore import PgVectorStore, get_vectors, load_vectorstore

    settings.chunk_unit, settings.chunk_size, settings.chunk_overlap = a.unit, a.size, a.overlap
    src_vs = load_vectorstore(a.from_backend)
    src_vs = getattr(src_vs, "inner", src_vs)
    raw = src_vs.get()
    ids, docs, metas = raw["ids"], raw["documents"], raw["metadatas"]

    meta_of: dict[str, dict] = {}
    for m in metas:
        if m.get("kind", "body") not in VISUAL_KINDS and m.get("source") not in meta_of:
            meta_of[m["source"]] = {k: m[k] for k in KEEP_META if k in m}
    copy_ids, new_docs, stats = [], [], {"rechunked": 0, "copied_papers": 0, "external_text": 0}
    for src, base in sorted(meta_of.items()):
        text = None
        if a.text_dir:
            for ext in (".md", ".txt"):
                f = Path(a.text_dir) / f"{src}{ext}"
                if f.exists():
                    text = f.read_text(encoding="utf-8", errors="ignore").replace("\x00", "")
                    stats["external_text"] += 1
                    break
        text = text or paperstore.get_text(src, full_only=True)
        if not text:
            copy_ids += [i for i, m in zip(ids, metas) if m.get("source") == src and m.get("kind", "body") not in VISUAL_KINDS]
            stats["copied_papers"] += 1
            continue
        chunks = section_aware_split(text, dict(base))
        if a.drop_back_matter or (base.get("origin") == "auto" and settings.auto_fetch_drop_back_matter):
            chunks = [c for c in chunks if not is_back_matter(c.metadata.get("section"))]
        new_docs += chunks
        stats["rechunked"] += 1
    copy_ids += [i for i, m in zip(ids, metas) if m.get("kind", "body") in VISUAL_KINDS]   # 표·그림·수식은 그대로

    dst = PgVectorStore(settings.database_url, a.table, get_embeddings())
    dst.delete_all()
    t0 = time.perf_counter()
    text_of = dict(zip(ids, docs))
    meta_by = dict(zip(ids, metas))
    for s in range(0, len(copy_ids), 500):
        part = copy_ids[s:s + 500]
        vec = get_vectors(src_vs, part)
        keep = [i for i in part if i in vec]
        dst.upsert_vectors(keep, [vec[i] for i in keep], [text_of[i] for i in keep], [meta_by[i] for i in keep])
    print(f"복사: {len(copy_ids)}개(표·그림·수식 + 전문 없는 논문 {stats['copied_papers']}편)", flush=True)
    for s in range(0, len(new_docs), 256):
        dst.add_documents([Document(page_content=d.page_content.replace("\x00", ""), metadata=d.metadata)
                           for d in new_docs[s:s + 256]])
        print(f"  임베딩 {min(s + 256, len(new_docs))}/{len(new_docs)}", flush=True)
    print(f"완료: {a.table} — 다시 자른 논문 {stats['rechunked']}편(외부 추출 {stats['external_text']}), "
          f"본문 청크 {len(new_docs)}개, 전체 {dst.count()}개 ({time.perf_counter() - t0:.0f}초)")


if __name__ == "__main__":
    main()
