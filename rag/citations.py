"""
rag/citations.py — 인용 그래프(OpenAlex). LLM 을 쓰지 않는다.

DB 논문마다 OpenAlex 의 피인용 수와 참고문헌 목록을 받아 `CITATION_GRAPH_PATH`(로컬 JSON)에 저장한다.
- 코퍼스 안 관계: 이 논문이 인용한 DB 논문(선행 연구) / 이 논문을 인용한 DB 논문(후속 연구) — 저장된 그래프만으로 계산
- 코퍼스 밖 관계: 많이 인용된 선행·후속 논문을 필요할 때 조회하고 결과를 그래프 파일에 캐시
- `common_references`: 여러 논문이 공통으로 인용한 논문(파생 논문들의 원 논문 찾기, A3)

OpenAlex 는 키 없이 쓰고, 연락처(mailto)는 보내지 않는다. 요청 사이에 짧게 쉬고 429 는 몇 번 재시도한다.
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
from collections import Counter
from pathlib import Path

from config import settings

API = "https://api.openalex.org/works"
_BATCH = 50
_LOCK = threading.Lock()
_ARXIV_ID = re.compile(r"^(\d{4}\.\d{4,5}|[a-z\-]+(\.[A-Z]{2})?/\d{7})$")


def is_arxiv_id(source: str) -> bool:
    return bool(_ARXIV_ID.match(source or ""))


def _empty() -> dict:
    return {"papers": {}, "works": {}, "later": {}, "missing": []}


def load() -> dict:
    p = Path(settings.citation_graph_path)
    if not p.exists():
        return _empty()
    try:
        g = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return _empty()
    for k, v in _empty().items():
        g.setdefault(k, v)
    return g


def save(g: dict) -> None:
    p = Path(settings.citation_graph_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(g, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, p)


def _get(session, params: dict, retries: int = 3) -> list[dict]:
    for attempt in range(retries + 1):
        from rag.arxiv_source import openalex_params

        r = session.get(API, params=openalex_params(params), timeout=30)
        if r.status_code == 429 and attempt < retries:
            time.sleep(2 * (attempt + 1))
            continue
        r.raise_for_status()
        time.sleep(0.2)  # 익명 요청 예절
        return r.json().get("results", [])
    return []


def _arxiv_of(work: dict) -> str | None:
    """OpenAlex 작업에서 arXiv ID 를 찾는다(DOI 10.48550/arxiv.* 또는 arxiv.org 위치)."""
    doi = (work.get("doi") or "").lower()
    if "10.48550/arxiv." in doi:
        return doi.split("10.48550/arxiv.")[-1]
    for loc in work.get("locations") or []:
        url = loc.get("landing_page_url") or ""
        if "arxiv.org/abs/" in url:
            return re.sub(r"v\d+$", "", url.rstrip("/").split("arxiv.org/abs/")[-1])
    return None


def _brief(work: dict) -> dict:
    return {"title": (work.get("display_name") or work.get("title") or "").strip(),
            "cited_by": work.get("cited_by_count") or 0, "year": work.get("publication_year"),
            "arxiv": _arxiv_of(work)}


def _short(oa_id: str) -> str:
    return (oa_id or "").rsplit("/", 1)[-1]


def _same_title(a: str, b: str) -> bool:
    def n(t):
        return re.sub(r"[^a-z0-9]", "", (t or "").lower())
    return bool(n(a)) and n(a) == n(b)


_SELECT = "id,doi,display_name,cited_by_count,publication_year,referenced_works,locations"


def _by_title(aid: str, title: str, session) -> dict | None:
    """arXiv DOI 로 안 잡히는 논문(학술지 DOI 로 등록된 경우)을 제목으로 찾는다. arXiv ID 또는 제목이 정확히 맞을 때만."""
    q = re.sub(r"[,:|]", " ", title)[:200]
    for w in _get(session, {"filter": f"title.search:{q}", "per-page": 5, "select": _SELECT}):
        if _arxiv_of(w) == aid or _same_title(w.get("display_name"), title):
            return w
    return None


def build(sources: list[str], titles: dict | None = None, session=None, progress=None) -> dict:
    """그래프에 없는 arXiv 논문을 OpenAlex 에서 받아 추가한다. 이미 있는 논문은 건너뛴다.
    arXiv DOI 로 50편씩 묶어 찾고, 못 찾은 논문은 titles(arXiv ID → 제목)가 있으면 제목으로 다시 찾는다."""
    from rag.arxiv_source import http_session

    session = session or http_session()
    with _LOCK:
        g = load()
        titles = titles or {}
        retry = [s for s in g["missing"] if s in titles and s in sources]  # 예전에 DOI 로 못 찾은 논문은 제목으로 재시도
        g["missing"] = [s for s in g["missing"] if s not in retry]
        todo = [s for s in dict.fromkeys(sources)
                if is_arxiv_id(s) and s not in g["papers"] and s not in g["missing"]]
        for i in range(0, len(todo), _BATCH):
            batch = todo[i:i + _BATCH]
            flt = "|".join(f"https://doi.org/10.48550/arxiv.{s}" for s in batch)
            works = _get(session, {"filter": f"doi:{flt}", "per-page": _BATCH, "select": _SELECT})
            found = {}
            for w in works:
                if _arxiv_of(w) in batch:
                    found[_arxiv_of(w)] = w
            for aid in batch:
                if aid not in found and titles.get(aid):
                    w = _by_title(aid, titles[aid], session)
                    if w:
                        found[aid] = w
            for aid, w in found.items():
                g["papers"][aid] = {"oa": _short(w["id"]), **_brief(w), "arxiv": aid,
                                    "refs": [_short(r) for r in w.get("referenced_works") or []]}
            g["missing"] += [s for s in batch if s not in found]
            if progress:
                progress(min(i + _BATCH, len(todo)), len(todo))
        save(g)
    return g


def _oa_index(g: dict) -> dict[str, str]:
    return {p["oa"]: aid for aid, p in g["papers"].items()}


def related_in_corpus(arxiv_id: str, g: dict | None = None) -> dict:
    """코퍼스 안 선행(이 논문이 인용)·후속(이 논문을 인용) 논문. 피인용 수 순."""
    g = g or load()
    me = g["papers"].get(arxiv_id)
    if not me:
        return {"prior": [], "later": []}
    idx = _oa_index(g)
    prior = [idx[r] for r in me["refs"] if r in idx and idx[r] != arxiv_id]
    later = [aid for aid, p in g["papers"].items() if me["oa"] in p["refs"] and aid != arxiv_id]

    def order(ids):
        return sorted(dict.fromkeys(ids), key=lambda a: -g["papers"][a]["cited_by"])

    return {"prior": order(prior), "later": order(later)}


def _resolve(oa_ids: list[str], g: dict, session) -> None:
    """OpenAlex ID 의 제목·피인용 수를 받아 g['works'] 에 캐시."""
    todo = [w for w in dict.fromkeys(oa_ids) if w not in g["works"]]
    for i in range(0, len(todo), _BATCH):
        batch = todo[i:i + _BATCH]
        works = _get(session, {"filter": "openalex_id:" + "|".join(batch), "per-page": _BATCH,
                               "select": "id,doi,display_name,cited_by_count,publication_year,locations"})
        for w in works:
            g["works"][_short(w["id"])] = _brief(w)


def prior_external(arxiv_id: str, n: int = 5, session=None) -> list[dict]:
    """이 논문의 참고문헌 중 많이 인용된 논문 n편(코퍼스 밖 포함). 처음 한 번만 조회."""
    from rag.arxiv_source import http_session

    with _LOCK:
        g = load()
        me = g["papers"].get(arxiv_id)
        if not me or not me["refs"]:
            return []
        if any(r not in g["works"] for r in me["refs"]):
            _resolve(me["refs"], g, session or http_session())
            save(g)
        items = [{"oa": r, **g["works"][r]} for r in me["refs"] if r in g["works"]]
    return sorted(items, key=lambda x: -x["cited_by"])[:n]


def later_external(arxiv_id: str, n: int = 5, session=None) -> list[dict]:
    """이 논문을 인용한 논문 중 많이 인용된 n편(코퍼스 밖 포함). 처음 한 번만 조회."""
    from rag.arxiv_source import http_session

    with _LOCK:
        g = load()
        me = g["papers"].get(arxiv_id)
        if not me:
            return []
        if arxiv_id not in g["later"]:
            works = _get(session or http_session(), {
                "filter": f"cites:{me['oa']}", "sort": "cited_by_count:desc", "per-page": 10,
                "select": "id,doi,display_name,cited_by_count,publication_year,locations"})
            g["later"][arxiv_id] = [{"oa": _short(w["id"]), **_brief(w)} for w in works]
            save(g)
        return g["later"][arxiv_id][:n]


def common_references(arxiv_ids: list[str], g: dict | None = None, min_count: int = 2) -> list[tuple[str, int]]:
    """여러 논문이 공통으로 인용한 논문(OpenAlex ID, 인용한 논문 수) — 많이 겹친 순.
    파생 논문 여러 편이 함께 인용한 논문은 대개 그 주제의 원 논문이다."""
    g = g or load()
    c: Counter = Counter()
    for a in dict.fromkeys(arxiv_ids):
        p = g["papers"].get(a)
        if p:
            c.update(set(p["refs"]))
    return [(w, k) for w, k in c.most_common() if k >= min_count]
