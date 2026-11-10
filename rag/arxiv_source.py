"""
rag/arxiv_source.py — arXiv 검색·PDF 전문 추출·청킹 공용 모듈

수집 스크립트(scripts/ingest_arxiv.py)와 챗봇의 논문 자동 수집(rag/auto_fetch.py)이 함께 쓴다.
사내 프록시 환경이면 config.resolved_ca_bundle() 의 CA 묶음으로 인증서 검증을 유지한 채 접속한다.
"""
from __future__ import annotations

import io
import re

import arxiv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import settings
from rag.chunking import section_aware_split


def norm_id(short_id: str) -> str:
    """'2310.11511v1' → '2310.11511' (버전 접미사 제거). 초록/전문 간 source 일치용."""
    return re.sub(r"v\d+$", "", short_id)


def make_client(ca_bundle: str | None = None, insecure: bool = False, interactive: bool = False) -> arxiv.Client:
    """arXiv 클라이언트. 사내 프록시(MITM SSL) 환경 대응.

    ca_bundle: 사내 CA 묶음(.pem) — 주지 않으면 config 의 CA 묶음을 자동 사용(검증 유지).
    insecure: SSL 검증을 끈다(보안 약화 — 신뢰된 사내망 최후수단).
    interactive: 챗봇처럼 사용자가 기다리는 경우 — 요청 간 지연·재시도를 줄인다(검색 1~2회뿐).
    """
    if interactive:
        client = arxiv.Client(page_size=20, delay_seconds=3.0, num_retries=2)
    else:  # 대량 수집: arXiv API 예의(rate limit) 준수 — 요청 간 지연 + 재시도로 429 완화
        client = arxiv.Client(page_size=50, delay_seconds=5.0, num_retries=5)
    ca_bundle = ca_bundle or settings.resolved_ca_bundle()
    if ca_bundle:
        client._session.verify = ca_bundle  # 사내 CA 로 검증 (안전)
    elif insecure:
        import urllib3

        urllib3.disable_warnings()
        client._session.verify = False  # ⚠️ 검증 비활성 (신뢰된 사내망 한정)
        print("  ⚠️ SSL 검증을 비활성화했습니다. 신뢰된 네트워크에서만 사용하세요.")
    return client


def paper_meta(result: arxiv.Result) -> dict:
    return {
        "source": norm_id(result.get_short_id()),
        "title": result.title.strip(),
        "authors": ", ".join(a.name for a in result.authors[:5]),
        "published": result.published.strftime("%Y-%m-%d"),
        "url": result.entry_id,
    }


def download_pdf_text(result: arxiv.Result, session, timeout: int = 60) -> str:
    """논문 PDF 를 받아 텍스트 추출. 실패하면 빈 문자열.

    arxiv 4.x 에는 Result.download_pdf 가 없어 pdf_url 을 직접 받는다.
    client 의 세션(session)을 재사용해 프록시 SSL(CA 묶음) 설정을 그대로 쓴다.
    """
    from pypdf import PdfReader

    url = getattr(result, "pdf_url", None)
    if not url:
        return ""
    try:
        resp = session.get(url, timeout=timeout)
        resp.raise_for_status()
        reader = PdfReader(io.BytesIO(resp.content))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception as e:  # noqa: BLE001
        print(f"    (PDF 실패, 초록으로 대체: {e})")
        return ""


# ---------------------------------------------------------------------------
# 논문 검색 (챗봇 자동 수집용): arXiv API → 막히면 OpenAlex
#   회사망처럼 공용 출구 IP 를 쓰면 arXiv 검색 API(export.arxiv.org)가 429 로 막히는 일이 잦다.
#   PDF(arxiv.org)는 받을 수 있으므로, 검색만 OpenAlex(무료·키 불필요 학술 검색)로 대신하고
#   결과 중 arXiv 에 있는 논문만 쓴다. 어느 쪽이든 아래 Paper 형태로 맞춰 이후 처리는 같다.
# ---------------------------------------------------------------------------
import time as _time
from dataclasses import dataclass, field
from datetime import date
from types import SimpleNamespace

_ARXIV_COOLDOWN_UNTIL = 0.0      # arXiv 검색 API 가 429 를 준 뒤 다시 시도하기까지 기다릴 시각
ARXIV_COOLDOWN_SEC = 30 * 60


class SearchRateLimited(Exception):
    pass


@dataclass
class Paper:
    """검색 출처와 무관한 논문 정보. arxiv.Result 와 같은 속성 이름이라 result_to_chunks 에 그대로 쓴다."""
    pid: str
    title: str
    summary: str = ""
    author_names: list[str] = field(default_factory=list)
    published: date = field(default_factory=date.today)
    found_by: str = ""

    def get_short_id(self) -> str:
        return self.pid

    @property
    def entry_id(self) -> str:
        return f"https://arxiv.org/abs/{self.pid}"

    @property
    def pdf_url(self) -> str:
        return f"https://arxiv.org/pdf/{self.pid}"

    @property
    def authors(self):
        return [SimpleNamespace(name=n) for n in self.author_names]


def http_session():
    """PDF·OpenAlex 요청용 세션(사내 CA 묶음으로 인증서 검증 유지)."""
    import requests

    s = requests.Session()
    s.verify = settings.resolved_ca_bundle() or True
    s.headers["User-Agent"] = "arxiv-advanced-rag/0.1 (research prototype)"
    return s


def search_arxiv(query: str, n: int, timeout: float = 8.0) -> list[Paper]:
    """arXiv 검색 API 를 직접 1회 호출(재시도 없음, 시간 제한). 429·시간 초과면 SearchRateLimited.

    arxiv 라이브러리를 거치면 막혀 있을 때 실패를 알기까지 15~60초가 걸려(실측),
    사용자가 기다리는 챗봇에서는 짧은 시간 제한으로 직접 부르고 Atom 응답을 표준 XML 파서로 읽는다.
    """
    import xml.etree.ElementTree as ET

    import requests

    try:
        r = http_session().get("https://export.arxiv.org/api/query", timeout=timeout, params={
            "search_query": f"all:{query}", "max_results": n, "sortBy": "relevance"})
    except requests.Timeout as e:
        raise SearchRateLimited(f"시간 초과({timeout:.0f}초)") from e
    if r.status_code == 429:
        raise SearchRateLimited("HTTP 429")
    r.raise_for_status()
    ns = {"a": "http://www.w3.org/2005/Atom"}
    text = lambda el, tag: " ".join((el.findtext(tag, default="", namespaces=ns) or "").split())  # noqa: E731
    out = []
    for e in ET.fromstring(r.content).findall("a:entry", ns):
        try:
            pub = date.fromisoformat(text(e, "a:published")[:10])
        except ValueError:
            pub = date.today()
        out.append(Paper(
            pid=norm_id(text(e, "a:id").split("/abs/")[-1]), title=text(e, "a:title"),
            summary=text(e, "a:summary"),
            author_names=[" ".join((a.findtext("a:name", default="", namespaces=ns)).split())
                          for a in e.findall("a:author", ns)[:5]],
            published=pub, found_by="arXiv"))
    return out


def _abstract_from_inverted(inv: dict | None) -> str:
    if not inv:
        return ""
    words = sorted((pos, w) for w, positions in inv.items() for pos in positions)
    return " ".join(w for _, w in words)


def search_openalex(query: str, n: int, session=None) -> list[Paper]:
    """OpenAlex 검색 → arXiv 에 올라온 논문만(arXiv ID 추출). 초록은 역색인으로 복원."""
    session = session or http_session()
    r = session.get("https://api.openalex.org/works", timeout=30, params={
        "search": query, "per-page": 25,
        "select": "display_name,locations,abstract_inverted_index,publication_date,authorships",
    })
    r.raise_for_status()
    out: list[Paper] = []
    for w in r.json().get("results", []):
        url = next((loc.get("landing_page_url") for loc in (w.get("locations") or [])
                    if "arxiv.org/abs/" in (loc.get("landing_page_url") or "")), None)
        if not url:
            continue
        pid = norm_id(url.rstrip("/").split("arxiv.org/abs/")[-1])
        try:
            pub = date.fromisoformat(w.get("publication_date") or "")
        except ValueError:
            pub = date.today()
        out.append(Paper(
            pid=pid, title=(w.get("display_name") or "").strip(),
            summary=_abstract_from_inverted(w.get("abstract_inverted_index")),
            author_names=[(a.get("author") or {}).get("display_name", "") for a in (w.get("authorships") or [])[:5]],
            published=pub, found_by="OpenAlex",
        ))
        if len(out) >= n:
            break
    return out


def search_papers(query: str, n: int, warn=None) -> list[Paper]:
    """arXiv 검색 → 429 면 30분간 arXiv 를 건너뛰고 OpenAlex 로. warn(message) 로 상황을 알린다."""
    global _ARXIV_COOLDOWN_UNTIL
    warn = warn or (lambda m: None)
    if _time.time() >= _ARXIV_COOLDOWN_UNTIL:
        try:
            return search_arxiv(query, n)
        except SearchRateLimited:
            _ARXIV_COOLDOWN_UNTIL = _time.time() + ARXIV_COOLDOWN_SEC
            warn("arXiv 검색 API 가 요청 제한(429) 중 → 30분간 OpenAlex 로 검색합니다")
        except Exception as e:  # noqa: BLE001
            warn(f"arXiv 검색 실패({type(e).__name__}) → OpenAlex 로 검색합니다")
    return search_openalex(query, n)


def flat_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap)


def result_to_chunks(result: arxiv.Result, session, fulltext: bool,
                     extra_meta: dict | None = None) -> tuple[list[Document], bool]:
    """논문 1편 → 청크 목록. (청크, 전문 사용 여부) 반환.

    fulltext: PDF 전문을 섹션 인식 청킹(각 청크에 section 메타). 실패하면 초록으로 대체.
    """
    meta = {**paper_meta(result), **(extra_meta or {})}
    abstract = f"{result.title}\n\n{result.summary}"
    if fulltext:
        text = download_pdf_text(result, session)
        if text:
            return section_aware_split(text, meta), True
        return section_aware_split(abstract, meta), False
    return flat_splitter().split_documents([Document(page_content=abstract, metadata=meta)]), False
