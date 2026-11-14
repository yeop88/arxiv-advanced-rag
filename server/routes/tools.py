"""답변 부가 기능: 근거 논문 비교표(LLM), 문헌 조사 초안(LLM), 인용 관계(LLM 없음)."""
from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from config import settings
from server.routes.deps import get_runtime
from server.runtime import Runtime
from server.schemas import CompareRequest, SurveyRequest, survey_json, table_json
from server.streaming import sse_job

router = APIRouter(prefix="/api")


@router.post("/compare-table")
async def compare_table(req: CompareRequest, rt: Runtime = Depends(get_runtime)):
    from rag.features.compare_table import MAX_PAPERS, build_table

    papers = list(dict.fromkeys(s for s in req.sources if s))[:MAX_PAPERS]

    def job(emit) -> None:
        pipe = rt.wait_ready(emit)
        with rt.session(emit):
            table = build_table(papers, pipe.vectorstore, progress=emit)
        emit("result", **table_json(table))

    return sse_job(job)


@router.post("/survey")
async def survey(req: SurveyRequest, rt: Runtime = Depends(get_runtime)):
    from rag.features.survey import build_survey

    def job(emit) -> None:
        pipe = rt.wait_ready(emit)
        with rt.session(emit):
            sv = build_survey(req.topic.strip(), pipe.vectorstore, progress=emit)
        emit("result", **survey_json(sv))

    return sse_job(job)


@router.get("/citations")
def citations(sources: list[str] = Query(default=[]), external: bool = False) -> dict:
    """근거 논문의 코퍼스 안 선행(이 논문이 인용)·후속(이 논문을 인용) 논문.
    external=true 이고 OpenAlex 가 켜져 있으면 코퍼스 밖 주요 논문도(처음 한 번 조회, 이후 캐시)."""
    from rag.features import citations as C

    g = C.load()

    def brief(a: str) -> dict:
        return {"id": a, "title": g["papers"][a]["title"]}

    out = []
    for src in dict.fromkeys(s for s in sources if s in g["papers"]):
        me, rel = g["papers"][src], C.related_in_corpus(src, g)
        item = {"source": src, "title": me["title"], "year": me.get("year"), "cited_by": me.get("cited_by", 0),
                "prior": [brief(a) for a in rel["prior"][:3]], "later": [brief(a) for a in rel["later"][:3]],
                "n_prior": len(rel["prior"]), "n_later": len(rel["later"])}
        if external and settings.openalex_enabled:
            try:
                item["external"] = {"prior": C.prior_external(src, 3), "later": C.later_external(src, 3)}
            except Exception as e:  # noqa: BLE001 — 네트워크 실패는 화면에만 알림
                item["external_error"] = type(e).__name__
        out.append(item)
    return {"papers": out, "openalex_enabled": settings.openalex_enabled}
