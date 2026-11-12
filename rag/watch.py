"""
rag/watch.py — 관심 분야 알림

등록한 주제마다 arXiv 를 최신순으로 검색해 새 논문을 찾고, 주제와의 관련도(로컬 리랭커)로 거른 뒤
경량 LLM 이 초록을 한국어 2~3문장으로 요약해 소식지(마크다운)를 만든다.

- 주제 목록·이미 본 논문: WATCH_PATH(JSON)    - 소식지: WATCH_DIGEST_DIR/YYYY-MM-DD.md
- 새 논문 = 아직 소식지에 실린 적 없고 DB 에도 없는 논문. 처음 실행은 최근 lookback_days 일만 본다.
- DB 에는 넣지 않는다(코퍼스 예산·품질 관리와 섞이지 않게). 필요하면 챗봇에서 질문해 자동 수집으로 넣는다.
"""
from __future__ import annotations

import datetime as dt
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from config import settings

MIN_SCORE = 0.3          # 자동 수집 후보와 같은 관련도 기준
CANDIDATES = 30          # 주제당 최신순으로 볼 논문 수
MAX_ITEMS = 5            # 주제당 소식지에 싣는 논문 수


@dataclass
class WatchItem:
    pid: str
    title: str
    published: str
    score: float
    summary: str


@dataclass
class TopicDigest:
    topic: str
    query: str
    items: list[WatchItem] = field(default_factory=list)
    checked: int = 0
    error: str = ""


def load() -> dict:
    p = Path(settings.watch_path)
    if not p.exists():
        return {"topics": []}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"topics": []}


def save(state: dict) -> None:
    p = Path(settings.watch_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, p)


def add_topic(topic: str, query: str | None = None) -> dict:
    """주제 등록. query(영어 arXiv 검색어)를 안 주면 자동 수집과 같은 방식으로 경량 LLM 이 만든다."""
    from rag.auto_fetch import make_search_queries

    state = load()
    if any(t["topic"] == topic for t in state["topics"]):
        return state
    if not query:  # 단어를 모두 포함해야 하므로(AND) 짧은 검색어를 고른다: 가장 짧은 것, 최대 4단어
        query = " ".join(min(make_search_queries(topic), key=lambda q: len(q.split())).split()[:4])
    state["topics"].append({"topic": topic, "query": query, "last_checked": None, "seen": []})
    save(state)
    return state


def remove_topic(topic: str) -> dict:
    state = load()
    state["topics"] = [t for t in state["topics"] if t["topic"] != topic]
    save(state)
    return state


def _summarize(topic: str, title: str, abstract: str) -> str:
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate

    from rag import prompts
    from rag.llm import get_llm

    chain = ChatPromptTemplate.from_template(prompts.WATCH_SUMMARY_PROMPT) | get_llm(light=True) | StrOutputParser()
    try:
        return chain.invoke({"topic": topic, "title": title, "abstract": abstract[:2500]}).strip()
    except Exception as e:  # noqa: BLE001
        return f"(요약 실패: {type(e).__name__})"


def check_topic(t: dict, existing: set[str], lookback_days: int, today: dt.date) -> TopicDigest:
    """한 주제의 새 논문을 찾는다. t["seen"]·t["last_checked"] 를 갱신한다."""
    from rag.arxiv_source import search_arxiv
    from rag.llm import get_reranker

    d = TopicDigest(t["topic"], t["query"])
    since = (dt.date.fromisoformat(t["last_checked"]) if t.get("last_checked")
             else today - dt.timedelta(days=lookback_days))
    try:
        # 최신순 정렬에서는 관련도 순위가 없어서, 단어를 모두 포함한 논문만 받는다(AND). 그냥 쓰면 무관한 최신 논문이 섞였다
        papers = search_arxiv(" AND all:".join(t["query"].split()), CANDIDATES, timeout=20, sort="submittedDate")
    except Exception as e:  # noqa: BLE001
        d.error = f"검색 실패: {type(e).__name__}: {str(e)[:80]}"
        return d
    seen = set(t.get("seen", []))
    new = [p for p in papers if p.published >= since and p.pid not in seen and p.pid not in existing]
    d.checked = len(new)
    if new:
        scores = get_reranker().predict([(t["topic"], f"{p.title}\n{p.summary}") for p in new])
        ranked = sorted(zip(new, (float(s) for s in scores)), key=lambda x: -x[1])
        for p, s in [x for x in ranked if x[1] >= MIN_SCORE][:MAX_ITEMS]:
            d.items.append(WatchItem(p.pid, p.title.strip(), p.published.isoformat(), s,
                                     _summarize(t["topic"], p.title, p.summary)))
    t["seen"] = list(seen | {p.pid for p in new})[-2000:]
    t["last_checked"] = today.isoformat()
    return d


def digest_markdown(digests: list[TopicDigest], today: dt.date) -> str:
    lines = [f"# 관심 분야 소식 — {today.isoformat()}", ""]
    for d in digests:
        lines.append(f"## {d.topic}")
        lines.append(f"<small>검색어: `{d.query}` · 새로 확인한 논문 {d.checked}편 · 실은 논문 {len(d.items)}편</small>")
        lines.append("")
        if d.error:
            lines.append(f"- ⚠ {d.error}")
        elif not d.items:
            lines.append("- 새로 실을 만한 논문이 없습니다.")
        for it in d.items:
            lines.append(f"- **[{it.title}](https://arxiv.org/abs/{it.pid})** ({it.published}, 관련도 {it.score:.2f})  ")
            lines.append(f"  {it.summary}")
        lines.append("")
    lines.append("> 요약은 로컬 모델이 초록만 보고 쓴 것입니다. 자세한 내용은 원문을 확인하세요.")
    return "\n".join(lines)


def run(vectorstore=None, lookback_days: int = 7, today: dt.date | None = None, progress=None) -> tuple[str, Path | None]:
    """등록된 모든 주제를 확인해 소식지를 만들고 저장한다. (마크다운, 저장 경로)."""
    from rag.auto_fetch import _existing_sources

    progress = progress or (lambda *a, **k: None)
    today = today or dt.date.today()
    state = load()
    if not state["topics"]:
        return "등록된 관심 주제가 없습니다.", None
    existing = _existing_sources(vectorstore) if vectorstore is not None else set()
    digests = []
    for i, t in enumerate(state["topics"], 1):
        progress("watch_topic", i=i, n=len(state["topics"]), topic=t["topic"])
        digests.append(check_topic(t, existing, lookback_days, today))
    save(state)
    md = digest_markdown(digests, today)
    out = Path(settings.watch_digest_dir) / f"{today.isoformat()}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md, encoding="utf-8")
    return md, out


def latest_digest() -> tuple[str, str] | None:
    """가장 최근 소식지 (날짜, 마크다운)."""
    p = Path(settings.watch_digest_dir)
    files = sorted(p.glob("*.md")) if p.exists() else []
    if not files:
        return None
    return files[-1].stem, files[-1].read_text(encoding="utf-8")
