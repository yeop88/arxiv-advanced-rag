"""
rag/store/paperstore.py — 논문 원문(PDF)·전문 텍스트 저장소 (PostgreSQL)

왜 필요한가?
  벡터 DB(Chroma)와 docstore.jsonl 에는 1,000자로 자른 청크만 남는다. 청킹 방식을 바꾸거나
  표·그림·수식을 다시 만들려면 논문을 arXiv 에서 다시 받아야 했다. 수집할 때 PDF 와 전문을 여기에 남겨 두면
  네트워크 없이 다시 처리할 수 있다.

테이블 (source = 청크 메타데이터의 source: arXiv ID 또는 로컬 파일 이름)
  papers      : 논문 정보(제목·저자·날짜·url·origin)
  paper_texts : 추출한 전문 텍스트, 전문 여부(실패 시 초록), 섹션 목록(이름·글자 수)
  paper_pdfs  : PDF 원본(bytea), 크기, sha256 — 무거워서 따로 둔다(논문 목록 조회가 PDF 를 읽지 않게)

DATABASE_URL 이 비어 있거나 DB 에 연결하지 못하면 저장을 건너뛰고 경고만 한다(수집은 계속).
벡터 DB 에서 논문을 지워도(자동 수집 예산 정리) 원문은 남긴다 — 다시 넣을 때 받지 않아도 되게.
"""
from __future__ import annotations

import hashlib
import json

from config import settings

SCHEMA = """
CREATE TABLE IF NOT EXISTS papers (
    source      TEXT PRIMARY KEY,
    title       TEXT NOT NULL DEFAULT '',
    authors     TEXT NOT NULL DEFAULT '',
    published   TEXT NOT NULL DEFAULT '',
    url         TEXT NOT NULL DEFAULT '',
    origin      TEXT NOT NULL DEFAULT 'base',
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS paper_texts (
    source      TEXT PRIMARY KEY REFERENCES papers(source) ON DELETE CASCADE,
    fulltext    TEXT NOT NULL,
    is_fulltext BOOLEAN NOT NULL,
    char_count  INTEGER NOT NULL,
    sections    JSONB NOT NULL DEFAULT '[]'::jsonb,
    extractor   TEXT NOT NULL DEFAULT 'pypdf',
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS paper_pdfs (
    source      TEXT PRIMARY KEY REFERENCES papers(source) ON DELETE CASCADE,
    pdf         BYTEA NOT NULL,
    size_bytes  INTEGER NOT NULL,
    sha256      TEXT NOT NULL,
    pdf_url     TEXT NOT NULL DEFAULT '',
    fetched_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
"""

_SCHEMA_READY: set[str] = set()   # 스키마를 만든 DATABASE_URL (프로세스당 1번)
_WARNED = False


def enabled() -> bool:
    return bool(settings.database_url)


def connect():
    """DB 연결(처음 한 번은 테이블도 만든다). with 문으로 쓰면 끝날 때 커밋·닫기."""
    import psycopg

    conn = psycopg.connect(settings.database_url, connect_timeout=5)
    if settings.database_url not in _SCHEMA_READY:
        with conn.transaction():
            conn.execute(SCHEMA)
        _SCHEMA_READY.add(settings.database_url)
    return conn


def _warn(e: Exception) -> None:
    global _WARNED
    if not _WARNED:   # 서버가 계속 같은 경고를 찍지 않게 한 번만
        print(f"    (원문 저장소 사용 불가 — 저장을 건너뜁니다: {type(e).__name__}: {str(e)[:150]})")
        _WARNED = True


def section_summary(text: str) -> list[dict]:
    """섹션 목록(이름·글자 수). 청킹과 같은 섹션 나누기를 쓴다."""
    from rag.ingest.chunking import split_into_sections

    return [{"name": n, "chars": len(b)} for n, b in split_into_sections(text)]


def save(meta: dict, text: str, is_fulltext: bool, pdf: bytes | None = None, pdf_url: str = "",
         extractor: str = "pypdf") -> bool:
    """논문 1편의 정보·전문·(있으면) PDF 를 저장(같은 source 면 갱신). 저장했으면 True.

    이미 전문이 있는데 이번엔 초록뿐이면(PDF 받기 실패) 전문을 덮어쓰지 않는다.
    """
    source = meta.get("source")
    if not enabled() or not source:
        return False
    text = (text or "").replace("\x00", "")
    row = {k: str(meta.get(k) or "") for k in ("title", "authors", "published", "url")}
    try:
        with connect() as conn, conn.transaction():
            conn.execute(
                """INSERT INTO papers (source, title, authors, published, url, origin)
                   VALUES (%s, %s, %s, %s, %s, %s)
                   ON CONFLICT (source) DO UPDATE SET title = EXCLUDED.title, authors = EXCLUDED.authors,
                       published = EXCLUDED.published, url = EXCLUDED.url, origin = EXCLUDED.origin, updated_at = now()""",
                (source, row["title"], row["authors"], row["published"], row["url"], meta.get("origin") or "base"),
            )
            if text:
                conn.execute(
                    """INSERT INTO paper_texts (source, fulltext, is_fulltext, char_count, sections, extractor)
                       VALUES (%s, %s, %s, %s, %s::jsonb, %s)
                       ON CONFLICT (source) DO UPDATE SET fulltext = EXCLUDED.fulltext, is_fulltext = EXCLUDED.is_fulltext,
                           char_count = EXCLUDED.char_count, sections = EXCLUDED.sections,
                           extractor = EXCLUDED.extractor, updated_at = now()
                       WHERE EXCLUDED.is_fulltext OR NOT paper_texts.is_fulltext""",
                    (source, text, is_fulltext, len(text), json.dumps(section_summary(text), ensure_ascii=False), extractor),
                )
            if pdf and settings.paper_store_pdf:
                conn.execute(
                    """INSERT INTO paper_pdfs (source, pdf, size_bytes, sha256, pdf_url) VALUES (%s, %s, %s, %s, %s)
                       ON CONFLICT (source) DO UPDATE SET pdf = EXCLUDED.pdf, size_bytes = EXCLUDED.size_bytes,
                           sha256 = EXCLUDED.sha256, pdf_url = EXCLUDED.pdf_url, fetched_at = now()""",
                    (source, pdf, len(pdf), hashlib.sha256(pdf).hexdigest(), pdf_url),
                )
        return True
    except Exception as e:  # noqa: BLE001 — 저장소 문제로 수집이 멈추면 안 된다
        _warn(e)
        return False


def _one(sql: str, params: tuple):
    if not enabled():
        return None
    try:
        with connect() as conn:
            r = conn.execute(sql, params).fetchone()
        return r[0] if r else None
    except Exception as e:  # noqa: BLE001
        _warn(e)
        return None


def get_text(source: str) -> str | None:
    return _one("SELECT fulltext FROM paper_texts WHERE source = %s", (source,))


def get_pdf(source: str) -> bytes | None:
    pdf = _one("SELECT pdf FROM paper_pdfs WHERE source = %s", (source,))
    return bytes(pdf) if pdf is not None else None


def stored_sources(with_pdf: bool = False) -> set[str]:
    """저장된 논문 source 목록(with_pdf 면 PDF 까지 있는 것만)."""
    if not enabled():
        return set()
    sql = "SELECT source FROM paper_pdfs" if with_pdf else "SELECT source FROM paper_texts WHERE is_fulltext"
    with connect() as conn:
        return {r[0] for r in conn.execute(sql).fetchall()}


def stats() -> dict:
    with connect() as conn:
        papers, = conn.execute("SELECT count(*) FROM papers").fetchone()
        texts, full, chars = conn.execute(
            "SELECT count(*), count(*) FILTER (WHERE is_fulltext), coalesce(sum(char_count), 0) FROM paper_texts").fetchone()
        pdfs, pdf_bytes = conn.execute("SELECT count(*), coalesce(sum(size_bytes), 0) FROM paper_pdfs").fetchone()
        by_origin = dict(conn.execute("SELECT origin, count(*) FROM papers GROUP BY origin ORDER BY 2 DESC").fetchall())
    return {"papers": papers, "texts": texts, "fulltext": full, "text_chars": int(chars),
            "pdfs": pdfs, "pdf_bytes": int(pdf_bytes), "by_origin": by_origin}
