"""
scripts/ingest_arxiv.py — arXiv 논문 수집 → 청킹 → 벡터DB 구축

사용 예:
    python scripts/ingest_arxiv.py --query "retrieval augmented generation" --max 20
    python scripts/ingest_arxiv.py --query "large language model agents" --max 30

arXiv API(무료)로 논문 메타데이터+초록을 가져와 Chroma 에 저장합니다.
초록(abstract) 기반이라 빠르고 합법적입니다. 전문(full text)이 필요하면
--fulltext 옵션으로 PDF 를 받아 파싱할 수 있습니다(느림).
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# 프로젝트 루트를 import 경로에 추가 (scripts/ 하위에서 실행해도 되게)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import arxiv  # noqa: E402
from langchain_core.documents import Document  # noqa: E402
from langchain_text_splitters import RecursiveCharacterTextSplitter  # noqa: E402

from config import settings  # noqa: E402
from rag.chunking import section_aware_split  # noqa: E402
from rag.vectorstore import build_vectorstore, delete_sources  # noqa: E402


def _norm_id(short_id: str) -> str:
    """'2310.11511v1' → '2310.11511' (버전 접미사 제거). 초록/전문 간 source 일치용."""
    return re.sub(r"v\d+$", "", short_id)


def _existing_sources() -> set[str]:
    """이미 벡터DB에 저장된 논문 source(arXiv ID) 집합. 없으면 빈 집합.

    재수집 시 같은 논문이 중복 청크로 쌓이는 것을 막기 위해 사용한다.
    """
    try:
        from rag.vectorstore import load_vectorstore  # 지역 import: DB 없을 때 대비

        raw = load_vectorstore().get()
        return {
            (m or {}).get("source")
            for m in raw.get("metadatas", [])
            if (m or {}).get("source")
        }
    except Exception:  # noqa: BLE001 — DB 미존재 등: 중복제거 없이 진행
        return set()


def _make_client(ca_bundle: str | None, insecure: bool) -> "arxiv.Client":
    """arXiv 클라이언트 생성. 사내 프록시(MITM SSL) 환경 대응.

    권장: --ca-bundle 로 사내 루트 CA 인증서(.pem)를 지정하면 '검증을 유지한 채' 통과.
    최후수단: --insecure 는 SSL 검증을 끈다(보안 약화 — 신뢰된 사내망에서만).
    """
    # arXiv API 예의(rate limit) 준수: 요청 간 지연 + 재시도로 429 완화.
    client = arxiv.Client(page_size=50, delay_seconds=5.0, num_retries=5)
    if ca_bundle:
        client._session.verify = ca_bundle  # 사내 CA 로 검증 (안전)
    elif insecure:
        import urllib3

        urllib3.disable_warnings()
        client._session.verify = False  # ⚠️ 검증 비활성 (신뢰된 사내망 한정)
        print("  ⚠️ SSL 검증을 비활성화했습니다. 신뢰된 네트워크에서만 사용하세요.")
    return client


def fetch_papers(
    query: str,
    max_results: int,
    fulltext: bool,
    ca_bundle: str | None = None,
    insecure: bool = False,
    skip: set[str] | None = None,
    id_list: list[str] | None = None,
) -> list[Document]:
    """arXiv 검색 결과를 LangChain Document 리스트로 변환.

    id_list 가 주어지면 검색어 대신 해당 논문 ID들을 정확히 가져온다.
    skip 에 든 arXiv ID(이미 수집된 논문)는 건너뛴다(중복 방지).
    """
    skip = skip or set()
    client = _make_client(ca_bundle, insecure)
    if id_list:
        search = arxiv.Search(id_list=id_list)
    else:
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance,
        )

    chunks: list[Document] = []
    for result in client.results(search):
        arxiv_id = _norm_id(result.get_short_id())
        if arxiv_id in skip:
            print(f"  건너뜀(중복): [{arxiv_id}] {result.title[:50]}")
            continue
        skip.add(arxiv_id)  # 같은 실행 내 중복도 방지
        meta = {
            "source": arxiv_id,
            "title": result.title.strip(),
            "authors": ", ".join(a.name for a in result.authors[:5]),
            "published": result.published.strftime("%Y-%m-%d"),
            "url": result.entry_id,
        }

        if fulltext:
            text = _download_and_read_pdf(result, client._session)
            if text:
                # 전문 → 섹션 인식 청킹 (각 청크에 section 메타 부여)
                paper_chunks = section_aware_split(text, meta)
            else:
                # PDF 실패 → 초록으로 폴백
                paper_chunks = section_aware_split(
                    f"{result.title}\n\n{result.summary}", meta
                )
        else:
            # 제목 + 초록만 (빠르고 충분한 경우 많음) → 플랫 청킹
            paper_chunks = _flat_splitter().split_documents(
                [Document(page_content=f"{result.title}\n\n{result.summary}", metadata=meta)]
            )

        chunks.extend(paper_chunks)
        n_sec = len({c.metadata.get("section") for c in paper_chunks})
        print(f"  수집: [{arxiv_id}] {meta['title'][:60]} "
              f"({len(paper_chunks)}청크/{n_sec}섹션)")

    return chunks


def _flat_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )


def _download_and_read_pdf(result: "arxiv.Result", session) -> str:
    """논문 PDF 를 받아 텍스트 추출 (fulltext 모드).

    arxiv 4.x 에는 Result.download_pdf 가 없어 pdf_url 을 직접 받는다.
    client 의 세션(session)을 재사용해 프록시 SSL(CA bundle) 설정을 그대로 쓴다.
    """
    import io

    from pypdf import PdfReader

    url = getattr(result, "pdf_url", None)
    if not url:
        return ""
    try:
        resp = session.get(url, timeout=60)
        resp.raise_for_status()
        reader = PdfReader(io.BytesIO(resp.content))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception as e:  # noqa: BLE001
        print(f"    (PDF 실패, 초록으로 대체: {e})")
        return ""


def main() -> None:
    parser = argparse.ArgumentParser(description="arXiv 논문 수집 → 벡터DB")
    parser.add_argument("--query", nargs="+", default=None,
                        help="arXiv 검색어(여러 개 가능: --query \"rag\" \"reranking\")")
    parser.add_argument("--ids", nargs="+", default=None,
                        help="논문 ID 직접 지정(예: --ids 2310.11511 2007.01282). --fulltext와 함께 전문 수집에 유용")
    parser.add_argument("--max", type=int, default=20, help="검색어당 수집할 논문 수")
    parser.add_argument("--fulltext", action="store_true", help="PDF 전문 파싱(느림)")
    parser.add_argument("--replace", action="store_true",
                        help="이미 있는 논문도 다시 수집(기존 청크 삭제 후 교체). 초록→전문 업그레이드에 사용")
    parser.add_argument("--ca-bundle", default=None,
                        help="사내 루트 CA 인증서(.pem) 경로 — 프록시 SSL 환경에서 검증 유지(권장)")
    parser.add_argument("--insecure", action="store_true",
                        help="⚠️ SSL 검증 비활성화 (신뢰된 사내망 최후수단)")
    args = parser.parse_args()
    if not args.query and not args.ids:
        parser.error("--query 또는 --ids 중 하나는 필요합니다.")

    # 임베딩은 로컬이라 LLM 키 없이도 수집 가능 → validate 생략
    mode = "전문(섹션 청킹)" if args.fulltext else "초록"
    # --replace 면 중복 건너뜀을 끄고(같은 논문을 다시 받음), 아니면 기존 논문 스킵.
    seen: set[str] = set() if args.replace else _existing_sources()
    if seen:
        print(f"  (기존 코퍼스 {len(seen)}편 감지 → 중복 건너뜀)")
    if args.replace:
        print("  (--replace: 기존 논문도 다시 수집, 옛 청크는 저장 전 삭제)")

    def _safe_fetch(**kw) -> list[Document]:
        """한 쿼리 실패(429 등)가 전체 수집을 날리지 않게 격리."""
        try:
            return fetch_papers(**kw)
        except Exception as e:  # noqa: BLE001
            print(f"  ⚠️ 이 쿼리 실패(건너뜀): {type(e).__name__}: {str(e)[:80]}")
            return []

    chunks: list[Document] = []
    if args.ids:
        print(f"[1/2] arXiv ID 지정 수집·청킹: {args.ids} ({mode})")
        chunks.extend(_safe_fetch(query="", max_results=args.max, fulltext=args.fulltext,
                                  ca_bundle=args.ca_bundle, insecure=args.insecure,
                                  skip=seen, id_list=args.ids))
    for q in (args.query or []):
        print(f"[1/2] arXiv 검색·청킹: '{q}' (최대 {args.max}편, {mode})")
        chunks.extend(_safe_fetch(query=q, max_results=args.max, fulltext=args.fulltext,
                                  ca_bundle=args.ca_bundle, insecure=args.insecure,
                                  skip=seen))
    if not chunks:
        print("수집된 논문이 없습니다(모두 중복이거나 결과 없음). 검색어/ID를 바꿔보세요.")
        return

    # --replace: 이번에 받은 논문들의 옛 청크(초록 등)를 지우고 새로 넣는다.
    if args.replace:
        new_sources = {c.metadata.get("source") for c in chunks if c.metadata.get("source")}
        removed = delete_sources(new_sources)
        if removed:
            print(f"  (교체: 기존 청크 {removed}개 삭제 후 재적재)")

    print(f"[2/2] 임베딩 + Chroma 저장 → {settings.chroma_dir}")
    build_vectorstore(chunks)
    print(f"완료! 총 {len(chunks)}개 청크 저장. 이제 `streamlit run app.py` 로 실행하세요.")


if __name__ == "__main__":
    main()
