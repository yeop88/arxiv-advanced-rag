"""
rag/auto_fetch.py — 코퍼스에 근거가 부족할 때 arXiv 에서 논문을 찾아 DB 에 추가 (자기 확장 RAG)

흐름:
  1) 판단: 2단계 리랭크 최고 점수 < auto_fetch_threshold → "코퍼스에 맞는 근거가 없다"
  2) 검색어: 질문(한국어 가능)을 짧은 영어 arXiv 검색어 N개로 변환(경량 LLM 1회)
  3) 후보: arXiv 검색(막히면 OpenAlex) → 이미 DB 에 있는 논문 제외
  4) 선별: (질문, 제목+초록)을 리랭커로 채점해 auto_fetch_min_score 이상만, 상위 max_papers 편
     → 관련 없는 논문이 DB 에 쌓여 다른 질문의 검색을 흐리는 것을 막는다
  5) 추가: PDF 전문을 섹션 청킹해 벡터 DB 에 추가(실패한 논문만 초록). origin=auto 로 표시
  6) BM25 색인 초기화 → 다음 검색부터 새 논문 포함

진행 상황은 emit(event, **info) 로 알린다(챗봇 실시간 표시용).
"""
from __future__ import annotations

import datetime as dt
import re
from typing import Callable

from config import settings

Emit = Callable[..., None]


_BACK_MATTER = re.compile(r"references|bibliograph|appendix", re.I)


def is_back_matter(section: str | None) -> bool:
    """참고문헌·부록 섹션인가. 온갖 논문 제목이 나열돼 있어 무관한 질문의 검색에 잘 걸린다
    (측정: 상위 5개에 끼어든 수집 청크의 68%)."""
    return bool(_BACK_MATTER.search(section or ""))


def top_score(reranked) -> float:
    return max((c.rerank_score or 0.0) for c in reranked) if reranked else 0.0


def needs_more(reranked) -> bool:
    """근거가 부족한가(1차 신호): 리랭크 최고 점수가 기준 미만이면 True.

    보정(2026-11-10, 평가셋 294문항 vs 코퍼스 밖 질문 8개): 기준 0.2 에서 코퍼스 안 질문의 4.1% 만
    잘못 걸리고, 코퍼스 밖 질문은 8개 중 6개를 잡는다. 주제는 비슷하지만 그 논문은 없는 경우
    (예: FlashAttention 0.45)는 점수가 높아 못 잡는다 → answer_lacks_evidence 가 2차 신호.
    """
    return top_score(reranked) < settings.auto_fetch_threshold


# 생성 프롬프트(GENERATION_SYSTEM 규칙 2)가 근거 부족 시 쓰도록 한 표현과 그 변형
_NO_EVIDENCE_MARKERS = ("찾을 수 없습니다", "찾을 수 없음", "근거 없음", "확인할 수 없습니다")


def answer_lacks_evidence(answer: str) -> bool:
    """근거가 부족한가(2차 신호): 답변이 '근거에서 찾을 수 없다'고 밝혔으면 True.
    점수로는 못 잡는 '비슷한 주제의 다른 논문만 있는' 경우를 잡는다."""
    return any(m in answer for m in _NO_EVIDENCE_MARKERS)


def make_search_queries(question: str) -> list[str]:
    """질문 → 짧은 영어 arXiv 검색어 목록(경량 LLM). 실패하면 질문 그대로 1개."""
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate

    from rag import prompts
    from rag.llm import get_llm
    from rag.stage1_query_transform import _split_lines

    try:
        chain = ChatPromptTemplate.from_template(prompts.ARXIV_SEARCH_PROMPT) | get_llm(light=True) | StrOutputParser()
        raw = chain.invoke({"question": question, "n": settings.auto_fetch_queries})
        queries = [q.strip().strip('"') for q in _split_lines(raw) if q.strip()]
        return queries[: settings.auto_fetch_queries] or [question]
    except Exception:  # noqa: BLE001
        return [question]


def _existing_sources(vectorstore) -> set[str]:
    raw = getattr(vectorstore, "inner", vectorstore).get()  # 숨긴 자동 수집분까지 포함해 중복 판단
    return {(m or {}).get("source") for m in raw.get("metadatas", []) if (m or {}).get("source")}


def fetch_and_ingest(question: str, vectorstore, emit: Emit | None = None) -> list[dict]:
    """근거 보강: 관련 논문을 찾아 DB 에 추가하고, 추가한 논문 정보를 돌려준다(없으면 빈 목록)."""
    from rag import stage2_rerank as s2
    from rag.arxiv_source import Paper, http_session, result_to_chunks, search_papers
    from rag.llm import get_reranker

    emit = emit or (lambda *a, **k: None)

    # 1) 검색어
    queries = make_search_queries(question)
    emit("fetch_queries", queries=queries)

    # 2) 후보 (arXiv 검색 → 막히면 OpenAlex). DB 에 이미 있는 논문 제외
    existing = _existing_sources(vectorstore)
    candidates: dict[str, Paper] = {}
    n_skipped = 0
    sources_used: set[str] = set()
    for q in queries:
        try:
            for p in search_papers(q, settings.auto_fetch_candidates,
                                   warn=lambda m: emit("fetch_warn", message=m)):
                sources_used.add(p.found_by)
                if p.pid in existing:
                    n_skipped += 1
                elif p.pid not in candidates:
                    candidates[p.pid] = p
        except Exception as e:  # noqa: BLE001 — 한 검색어 실패가 전체를 막지 않게
            emit("fetch_warn", message=f"검색 실패({q}): {str(e)[:120]}")
    emit("fetch_candidates", n=len(candidates), n_existing_skipped=n_skipped, sources=sorted(sources_used))
    if not candidates:
        return []
    session = http_session()  # PDF 는 arxiv.org 에서(사내 CA 로 검증 유지)

    # 3) 선별: (질문, 제목+초록) 리랭크 점수
    items = list(candidates.values())
    scores = get_reranker().predict([(question, f"{r.title}\n{r.summary}") for r in items])
    ranked = sorted(zip(items, (float(s) for s in scores)), key=lambda x: x[1], reverse=True)
    chosen = [(r, s) for r, s in ranked if s >= settings.auto_fetch_min_score][: settings.auto_fetch_max_papers]
    emit("fetch_selected", papers=[(r.title.strip(), s) for r, s in chosen],
         best_rejected=(ranked[len(chosen)][1] if len(ranked) > len(chosen) else None))
    if not chosen:
        return []

    # 4) 전문 수집 → DB 추가
    added: list[dict] = []
    today = dt.date.today().isoformat()
    for i, (r, s) in enumerate(chosen, 1):
        title = r.title.strip()
        emit("fetch_paper_start", i=i, n=len(chosen), title=title)
        try:  # 논문 한 편의 실패가 나머지 수집을 막지 않게
            chunks, used_full = result_to_chunks(
                r, session, settings.auto_fetch_fulltext,
                extra_meta={"origin": "auto", "fetched_at": today, "fetch_query": question[:200],
                            "fetch_score": round(s, 4)},
            )
            if settings.auto_fetch_drop_back_matter:
                chunks = [c for c in chunks if not is_back_matter(c.metadata.get("section"))]
            if not chunks:
                continue
            vectorstore.add_documents(chunks)
        except Exception as e:  # noqa: BLE001
            emit("fetch_warn", message=f"논문 추가 실패({r.pid}): {type(e).__name__}: {str(e)[:120]}")
            continue
        info = {"source": r.pid, "title": title, "score": s, "found_by": r.found_by,
                "chunks": len(chunks), "fulltext": used_full}
        added.append(info)
        emit("fetch_paper_done", i=i, n=len(chosen), **info)

    # 5) 새 원문을 BM25 에 반영
    if added:
        s2.reset_bm25()
    return added
