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
import json
import os
import re
from pathlib import Path
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


_NAME = re.compile(r"[A-Za-z][A-Za-z0-9\-]+")
_GENERIC_NAMES = {"LLM", "LLMS", "RAG", "GPU", "CPU", "AI", "NLP", "MLP", "CNN", "RNN", "API", "QA"}


def named_terms(question: str) -> list[str]:
    """질문의 고유 명칭(대문자 2개 이상 또는 숫자 포함: SimCLR, BM25). 일반 약어는 뺀다."""
    names = [w for w in _NAME.findall(question)
             if (sum(c.isupper() for c in w) >= 2 or any(c.isdigit() for c in w)) and w.upper() not in _GENERIC_NAMES]
    return list(dict.fromkeys(names))


# 비교 질문에서 명칭이 근거 청크 몇 개 이상에 나와야 '근거에 있다'고 볼지.
# 1이면 다른 논문이 한 번 언급만 해도 수집하지 않아 비교 질문을 놓쳤다(이슈 18).
_MIN_MENTIONS = 2
_SMALL_WORDS = {"a", "an", "the", "of", "for", "and", "with", "in", "on", "to", "via", "by", "from"}


def _norm(text: str) -> str:
    """띄어쓰기·하이픈·밑줄을 지운 소문자("Graph RAG" == "GraphRAG")."""
    return re.sub(r"[\s\-_]+", "", text.lower())


def _is_acronym(name: str) -> bool:
    """대문자 3~6자로만 된 약어(VQA, NER). 2자는 우연히 맞는 경우가 많아 뺀다."""
    return 3 <= len(name) <= 6 and name.isalpha() and name.isupper()


def _initials(title: str) -> str:
    """제목 단어의 머리글자(관사·전치사 제외): "Visual Question Answering" → "vqa"."""
    return "".join(w[0] for w in re.findall(r"[A-Za-z]+", title) if w.lower() not in _SMALL_WORDS).lower()


def _in_title(name: str, title: str) -> bool:
    """제목에 명칭이 있는가: 띄어쓰기 무시 일치, 또는 약어가 제목 머리글자와 일치."""
    if _norm(name) in _norm(title):
        return True
    return _is_acronym(name) and name.lower() in _initials(title)


def answer_signal(question: str, answer: str, reranked) -> bool:
    """2차 신호: 답변이 근거 부족을 밝혔고, 질문이 묻는 대상이 근거에 정말 없을 때만 수집한다.

    다중 홉 질문은 일부만 못 찾아도 '찾을 수 없다'고 쓰므로 표현만으로 판단하면 과하게 켜진다(측정 47%).
    - 명칭이 하나: 그 이름이 제목에 들어간 근거 논문이 없으면 수집(다른 논문이 언급만 한 경우 포함)
    - 명칭이 여럿: 어느 하나라도 제목에 없고 근거 청크 _MIN_MENTIONS 개 미만에서만 언급되면 수집
      (예: "RoBERTa와 XLNet 차이" — XLNet 이 다른 논문 한 곳에서 언급만 된 경우)
    제목 비교는 띄어쓰기를 무시하고 약어는 머리글자로도 맞춘다("GraphRAG"↔"Graph RAG", "VQA"↔"Visual Question Answering").
    """
    if not answer_lacks_evidence(answer):
        return False
    names = named_terms(question)
    if not names:
        return False

    def in_titles(name: str) -> bool:
        return any(_in_title(name, c.title) for c in reranked)

    if len(names) == 1:
        return not in_titles(names[0])

    def mentions(name: str) -> int:
        return sum(_norm(name) in _norm(f"{c.title} {c.content}") for c in reranked)

    return any(not in_titles(n) and mentions(n) < _MIN_MENTIONS for n in names)


def select_papers(ranked: list) -> list:
    """(논문, 점수) 점수순 목록에서 추가할 논문을 고른다.

    파생 논문이 원 논문보다 점수가 높은 경우가 많아, 관련도가 충분한 후보 중 가장 많이 인용된 논문을 먼저 넣는다.
    """
    eligible = [x for x in ranked if x[1] >= settings.auto_fetch_min_score]
    anchors = [x for x in eligible if x[1] >= settings.auto_fetch_anchor_score and x[0].cited_by > 0]
    if anchors:
        anchor = max(anchors, key=lambda x: x[0].cited_by)
        eligible = [anchor] + [x for x in eligible if x is not anchor]
    return eligible[: settings.auto_fetch_max_papers]


def load_usage() -> dict[str, dict]:
    """논문별 근거 사용 기록 {arXiv ID: {count, last_used}}. 없거나 깨졌으면 빈 기록."""
    try:
        return json.loads(Path(settings.auto_usage_path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _save_usage(usage: dict) -> None:
    path = Path(settings.auto_usage_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(usage, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def record_usage(reranked) -> None:
    """최종 근거로 쓰인 자동 수집 논문을 기록한다. 예산을 넘을 때 안 쓰인 논문부터 지우는 기준이 된다."""
    sources = {c.source for c in reranked if c.origin == "auto"}
    if not sources:
        return
    usage, today = load_usage(), dt.date.today().isoformat()
    for s in sources:
        row = usage.setdefault(s, {"count": 0})
        row["count"] += 1
        row["last_used"] = today
    _save_usage(usage)


def evict_over_budget(vectorstore, protect: set[str] | None = None) -> list[dict]:
    """자동 수집 청크가 예산을 넘으면 오래 안 쓰인 논문부터 삭제하고, 지운 논문 목록을 돌려준다.

    순서: 마지막 사용일(안 쓰였으면 수집일)이 오래된 순, 같으면 사용 횟수가 적은 순.
    protect(방금 수집한 논문)와 수동 수집 논문은 지우지 않는다.
    """
    budget = settings.auto_fetch_max_chunks
    if budget <= 0:
        return []
    protect = protect or set()
    store = getattr(vectorstore, "inner", vectorstore)
    raw = store.get()
    papers: dict[str, dict] = {}
    for i, m in zip(raw["ids"], raw["metadatas"]):
        m = m or {}
        if m.get("origin") == "auto":
            p = papers.setdefault(m["source"], {"ids": [], "title": m.get("title", ""), "fetched_at": m.get("fetched_at", "")})
            p["ids"].append(i)
    total = sum(len(p["ids"]) for p in papers.values())
    if total <= budget:
        return []
    usage = load_usage()

    def order(item):
        pid, p = item
        row = usage.get(pid, {})
        return row.get("last_used") or p["fetched_at"], row.get("count", 0)

    evicted = []
    for pid, p in sorted(((k, v) for k, v in papers.items() if k not in protect), key=order):
        if total <= budget:
            break
        store.delete(ids=p["ids"])
        total -= len(p["ids"])
        usage.pop(pid, None)
        evicted.append({"source": pid, "title": p["title"], "chunks": len(p["ids"])})
    if evicted:
        _save_usage(usage)
    return evicted


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
    from rag.arxiv_source import Paper, http_session, result_to_chunks, search_openalex, search_papers
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
    if settings.auto_fetch_citation_search:  # 원 논문은 글자 일치 순서에서 밀리므로 인용 수 순으로도 찾는다
        for q in queries:
            try:
                for p in search_openalex(q, 50, by_citations=True):
                    sources_used.add(p.found_by)
                    if p.pid in candidates:
                        candidates[p.pid].cited_by = max(candidates[p.pid].cited_by, p.cited_by)
                    elif p.pid not in existing:
                        candidates[p.pid] = p
            except Exception as e:  # noqa: BLE001
                emit("fetch_warn", message=f"인용 순 검색 실패({q}): {str(e)[:120]}")
    emit("fetch_candidates", n=len(candidates), n_existing_skipped=n_skipped, sources=sorted(sources_used))
    if not candidates:
        return []
    session = http_session()  # PDF 는 arxiv.org 에서(사내 CA 로 검증 유지)

    # 3) 선별: (질문, 제목+초록) 리랭크 점수
    items = list(candidates.values())
    scores = get_reranker().predict([(question, f"{r.title}\n{r.summary}") for r in items])
    ranked = sorted(zip(items, (float(s) for s in scores)), key=lambda x: x[1], reverse=True)
    chosen = select_papers(ranked)
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

    # 5) 예산을 넘었으면 오래 안 쓰인 논문 정리 → 새 원문을 BM25 에 반영
    if added:
        try:  # 인용 그래프에도 추가(실패해도 수집 결과에는 영향 없음)
            from rag import citations
            citations.build([a["source"] for a in added], titles={a["source"]: a["title"] for a in added}, session=session)
        except Exception as e:  # noqa: BLE001
            emit("fetch_warn", message=f"인용 그래프 갱신 실패: {type(e).__name__}")
        evicted = evict_over_budget(vectorstore, protect={a["source"] for a in added})
        if evicted:
            emit("fetch_evicted", papers=evicted)
        s2.reset_bm25()
    return added
