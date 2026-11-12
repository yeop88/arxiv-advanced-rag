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


_DEFINE_MENTIONS = 2  # 한 청크에 이름이 단어로 이만큼 나오면 그 논문이 이름의 주인으로 본다


def _word_count(name: str, text: str) -> int:
    """단어 경계를 지킨 등장 횟수("RETRO" 가 "retrocausal" 에, "FiD" 가 "fidelity" 에 걸리지 않게)."""
    return len(re.findall(r"(?<![A-Za-z0-9])" + re.escape(name) + r"(?![A-Za-z0-9])", text or "", flags=re.I))


def answer_signal(question: str, answer: str, reranked) -> bool:
    """2차 신호: 답변이 근거 부족을 밝혔고, 질문이 묻는 대상이 근거에 정말 없을 때만 수집한다.

    다중 홉 질문은 일부만 못 찾아도 '찾을 수 없다'고 쓰므로 표현만으로 판단하면 과하게 켜진다(측정 47%).
    - 명칭이 하나: 그 이름이 제목에 들어간 근거 논문도, 이름을 여러 번 쓴 근거 청크도 없으면 수집(다른 논문이 언급만 한 경우 포함)
    - 명칭이 여럿: 어느 하나라도 제목에 없고 근거 청크 _MIN_MENTIONS 개 미만에서만 언급되면 수집
      (예: "RoBERTa와 XLNet 차이" — XLNet 이 다른 논문 한 곳에서 언급만 된 경우)
    제목 비교는 띄어쓰기를 무시하고 약어는 머리글자로도 맞춘다("GraphRAG"↔"Graph RAG", "VQA"↔"Visual Question Answering").
    """
    if not answer_lacks_evidence(answer):
        return False
    names = named_terms(question)
    if not names:
        return False

    def covered(name: str) -> bool:
        # 제목에 있거나, 한 근거 청크 안에 단어로 여러 번 나오면(그 논문이 이름을 정의한 원 논문) 근거 있음.
        # 제목에 이름이 없는 원 논문(HyDE: "Precise Zero-Shot Dense Retrieval…")이 근거에 있는데도 켜지던 경우(§5.31)
        return any(_in_title(name, c.title) or _word_count(name, c.content) >= _DEFINE_MENTIONS for c in reranked)

    if len(names) == 1:
        return not covered(names[0])

    def mentions(name: str) -> int:
        return sum(_norm(name) in _norm(f"{c.title} {c.content}") for c in reranked)

    return any(not covered(n) and mentions(n) < _MIN_MENTIONS for n in names)


def select_papers(ranked: list) -> list:
    """(논문, 점수) 점수순 목록에서 추가할 논문을 고른다.

    파생 논문이 원 논문보다 점수가 높은 경우가 많아, 관련도가 충분한 후보 중 가장 많이 인용된 논문을 먼저 넣는다.
    """
    eligible = [x for x in ranked if x[1] >= settings.auto_fetch_min_score
                or float(getattr(x[0], "title_match", 0) or 0) >= _TITLE_EXACT]
    anchors = [x for x in eligible if x[1] >= settings.auto_fetch_anchor_score and x[0].cited_by > 0]
    if anchors:
        anchor = max(anchors, key=lambda x: x[0].cited_by)
        eligible = [anchor] + [x for x in eligible if x is not anchor]
    titled = [x for x in eligible if getattr(x[0], "title_match", False)]  # 추정한 원 논문 제목과 맞으면 맨 앞(A3)
    if titled:
        best = max(titled, key=lambda x: (float(x[0].title_match), x[1]))  # 제목이 더 정확히 맞는 쪽, 같으면 관련도
        # 원 논문을 찾았으면 그것만 — 함께 넣던 파생 논문이 다른 질문 근거에 끼어들었다(§5.28).
        # 단, 관련도 기준 미달인데 제목만 맞아 뽑힌 경우는 추정이 틀렸을 수 있어(MuZero → AlphaZero, §5.29) 다른 후보도 함께 넣는다
        if settings.auto_fetch_title_only and best[1] >= settings.auto_fetch_min_score:
            return [best]
        eligible = [best] + [x for x in eligible if x is not best]
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


def guess_original_title(question: str) -> str | None:
    """경량 LLM 이 질문의 원 논문 제목을 추정(모르면 None)."""
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate

    from rag import prompts
    from rag.llm import get_llm

    try:
        chain = ChatPromptTemplate.from_template(prompts.ORIGINAL_TITLE_PROMPT) | get_llm(light=True) | StrOutputParser()
        lines = chain.invoke({"question": question}).strip().splitlines()
    except Exception:  # noqa: BLE001
        return None
    title = (lines[0] if lines else "").strip().strip("\"'*").strip()
    if len(title) < 8 or title.upper().startswith("NONE"):
        return None
    return title


def _title_words(t: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", (t or "").lower()) if len(w) > 2}


def title_overlap(a: str, b: str) -> float:
    """두 제목의 3자 이상 단어 겹침 비율(긴 쪽 기준, 0~1)."""
    wa, wb = _title_words(a), _title_words(b)
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / max(len(wa), len(wb))


_TITLE_MATCH = 0.6     # 같은 논문으로 볼 제목 겹침 비율(긴 쪽 기준). "Nets"↔"Networks" 같은 차이를 허용
_TITLE_EXACT = 0.9     # 이 이상이면 관련도 기준 없이 선정(한국어 질문과 영어 초록의 관련도가 낮게 나오는 원 논문 대비)


def title_similar(a: str, b: str, min_overlap: float = _TITLE_MATCH) -> bool:
    """두 제목이 같은 논문인가: 겹침 비율(긴 쪽 기준)이 기준 이상.
    긴 쪽 기준이라 "Vision Transformer" 같은 짧은 추정이 "Vision Transformer with ..." 파생 논문과 맞지 않는다."""
    return title_overlap(a, b) >= min_overlap


def title_candidates(title: str, warn=None) -> list:
    """추정 제목으로 arXiv 제목 검색(실패 시 OpenAlex) → 제목이 비슷한 논문만(title_match 표시)."""
    from rag.arxiv_source import search_arxiv, search_openalex

    warn = warn or (lambda m: None)
    q = re.sub(r"[^A-Za-z0-9 \-]", " ", title).strip()
    found = []
    try:  # arXiv 제목 검색 먼저, 실패할 때만 OpenAlex(하루 무료 예산이 작다)
        found = search_arxiv(f'"{q}"', 5, field="ti")
    except Exception as e:  # noqa: BLE001
        warn(f"arXiv 제목 검색 실패: {type(e).__name__}")
        try:
            found = search_openalex(title, 10)
        except Exception as e2:  # noqa: BLE001
            warn(f"OpenAlex 제목 검색 실패: {type(e2).__name__}")
    out = {}
    for p in found:
        if p.pid not in out and title_similar(p.title, title):
            p.title_match = round(title_overlap(p.title, title), 3)  # 참(0보다 큼)이면서 얼마나 정확히 맞는지
            out[p.pid] = p
    return list(out.values())


def common_ref_candidates(evidence_sources: list[str], n: int = 3) -> list:
    """근거 논문들이 공통으로 인용한 논문 상위 n편(arXiv 에 있는 것만). 인용 그래프가 없으면 빈 목록."""
    from rag import citations
    from rag.arxiv_source import fetch_arxiv_by_ids, http_session

    g = citations.load()
    common = citations.common_references(evidence_sources, g)[: n * 2]
    if not common:
        return []
    if settings.openalex_enabled:  # 끈 상태면 저장해 둔 정보만 쓴다
        citations._resolve([w for w, _ in common], g, http_session())
        citations.save(g)
    works = [g["works"].get(w) or {} for w, _ in common]
    ids = list(dict.fromkeys(w["arxiv"] for w in works if w.get("arxiv")))[:n]
    if not ids:
        return []
    cited = {w["arxiv"]: w.get("cited_by", 0) for w in works if w.get("arxiv")}
    papers = fetch_arxiv_by_ids(ids)
    for pid, p in papers.items():
        p.cited_by, p.found_by = cited.get(pid, 0), "인용 그래프"
    return list(papers.values())


def find_candidates(question: str, existing: set[str], emit: Emit | None = None,
                    evidence_sources: list[str] | None = None, title_guess: bool | None = None,
                    common_refs: bool | None = None) -> dict:
    """수집 후보를 모아 관련도로 채점하고 넣을 논문을 고른다(DB 는 바꾸지 않음 — 평가에서도 그대로 쓴다).

    후보 출처: ① 검색어(경량 LLM)로 arXiv(막히면 OpenAlex) ② 같은 검색어로 OpenAlex 인용 순
    ③ 원 논문 제목 추정 → 제목 검색(A3) ④ 근거 논문들의 공통 참고문헌(A3, 인용 그래프).
    """
    from rag.arxiv_source import Paper, search_openalex, search_papers
    from rag.llm import get_reranker

    emit = emit or (lambda *a, **k: None)
    title_guess = settings.auto_fetch_title_guess if title_guess is None else title_guess
    common_refs = settings.auto_fetch_common_refs if common_refs is None else common_refs

    def warn(m: str) -> None:
        emit("fetch_warn", message=m)

    # 1) 검색어
    queries = make_search_queries(question)
    emit("fetch_queries", queries=queries)

    # 2) 후보. DB 에 이미 있는 논문 제외
    candidates: dict[str, Paper] = {}
    n_skipped = 0
    sources_used: set[str] = set()

    def add(p: Paper) -> None:
        nonlocal n_skipped
        sources_used.add(p.found_by)
        if p.pid in existing:
            n_skipped += 1
        elif p.pid in candidates:
            old = candidates[p.pid]
            old.cited_by = max(old.cited_by, p.cited_by)
            old.title_match = max(old.title_match, p.title_match)
        else:
            candidates[p.pid] = p

    for q in queries:
        try:
            for p in search_papers(q, settings.auto_fetch_candidates, warn=warn):
                add(p)
        except Exception as e:  # noqa: BLE001 — 한 검색어 실패가 전체를 막지 않게
            warn(f"검색 실패({q}): {str(e)[:120]}")
    if settings.auto_fetch_citation_search and settings.openalex_enabled:  # 원 논문은 글자 일치 순서에서 밀려 인용 수 순으로도 찾는다
        for q in queries:
            try:
                for p in search_openalex(q, 50, by_citations=True):
                    add(p)
            except Exception as e:  # noqa: BLE001
                warn(f"인용 순 검색 실패({q}): {str(e)[:120]}")
    guessed = guess_original_title(question) if title_guess else None
    if guessed:
        emit("fetch_title_guess", title=guessed)
        for p in title_candidates(guessed, warn=warn):
            add(p)
    if common_refs and evidence_sources:
        try:
            for p in common_ref_candidates(evidence_sources):
                add(p)
        except Exception as e:  # noqa: BLE001
            warn(f"공통 참고문헌 후보 실패: {type(e).__name__}")
    emit("fetch_candidates", n=len(candidates), n_existing_skipped=n_skipped, sources=sorted(sources_used))
    result = {"queries": queries, "guessed": guessed, "ranked": [], "chosen": []}
    if not candidates:
        return result

    # 3) 선별: (질문, 제목+초록) 리랭크 점수
    items = list(candidates.values())
    scores = get_reranker().predict([(question, f"{r.title}\n{r.summary}") for r in items])
    ranked = sorted(zip(items, (float(s) for s in scores)), key=lambda x: x[1], reverse=True)
    chosen = select_papers(ranked)
    emit("fetch_selected", papers=[(r.title.strip(), s) for r, s in chosen],
         best_rejected=(ranked[len(chosen)][1] if len(ranked) > len(chosen) else None))
    result.update(ranked=ranked, chosen=chosen)
    return result


def fetch_and_ingest(question: str, vectorstore, emit: Emit | None = None,
                     evidence_sources: list[str] | None = None) -> list[dict]:
    """근거 보강: 관련 논문을 찾아 DB 에 추가하고, 추가한 논문 정보를 돌려준다(없으면 빈 목록).
    evidence_sources: 지금 근거 논문들(공통 참고문헌 후보에 쓴다)."""
    from rag import stage2_rerank as s2
    from rag.arxiv_source import http_session, result_to_chunks

    emit = emit or (lambda *a, **k: None)
    chosen = find_candidates(question, _existing_sources(vectorstore), emit, evidence_sources)["chosen"]
    if not chosen:
        return []
    session = http_session()  # PDF 는 arxiv.org 에서(사내 CA 로 검증 유지)

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
    if added and settings.openalex_enabled:
        try:  # 인용 그래프에도 추가(실패해도 수집 결과에는 영향 없음)
            from rag import citations
            citations.build([a["source"] for a in added], titles={a["source"]: a["title"] for a in added}, session=session)
        except Exception as e:  # noqa: BLE001
            emit("fetch_warn", message=f"인용 그래프 갱신 실패: {type(e).__name__}")
    if added:
        evicted = evict_over_budget(vectorstore, protect={a["source"] for a in added})
        if evicted:
            emit("fetch_evicted", papers=evicted)
        s2.reset_bm25()
    return added
