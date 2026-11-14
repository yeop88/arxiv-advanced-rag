"""
tests/test_server.py — FastAPI 서버(SSE 스트리밍) 테스트. 모델·LLM 없이 가짜 로더·파이프라인으로 확인한다.

  - 답변 토큰이 새 글자(delta)로 하나씩 오고, 이어 붙이면 최종 답변과 같은가
  - 요청별 설정이 실행 중에만 적용되고 끝나면 원래대로 돌아오는가
  - 준비 전이면 waiting 알림 → 준비되면 이어서 답하는가 / 준비 실패는 fail 로 끝나는가
  - 화면이 연결을 끊으면 작업이 멈추고 잠금이 풀리는가
  - 질문이 기록되고 👍/👎 피드백이 붙는가(임시 DB)
실행: python tests/test_server.py
"""
from __future__ import annotations

import json
import sys
import tempfile
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient  # noqa: E402

from config import settings  # noqa: E402
from rag.core.schemas import PipelineResult, RetrievedChunk, VerificationResult  # noqa: E402
from server.app import create_app  # noqa: E402
from server.conversations import ConversationStore  # noqa: E402
from server.querylog import QueryLog  # noqa: E402
from server.runtime import Runtime  # noqa: E402

ANSWER = "리랭킹은 정밀도를 높입니다 [1]. 크로스인코더가 질문과 문단을 함께 봅니다 [2]."


class FakePipeline:
    def __init__(self, endless: bool = False):
        self.vectorstore = None
        self.endless = endless
        self.seen_settings: dict = {}
        self.stopped = threading.Event()

    def run(self, question, chat_history=None, progress=None):
        self.seen_settings = {"mode": settings.query_transform_mode, "top_k": settings.top_k_rerank}
        progress("transform_start", mode=settings.query_transform_mode)
        progress("transform_done", queries=[question], sec=0.1)
        progress("generate_start", n_chunks=2)
        try:
            if self.endless:   # 연결이 끊길 때까지 토큰을 계속 낸다
                while True:
                    progress("token", delta="가")
                    time.sleep(0.01)
            for i in range(0, len(ANSWER), 3):
                progress("token", delta=ANSWER[i:i + 3])
        except BaseException:
            self.stopped.set()
            raise
        progress("generate_done", sec=0.2, chars=len(ANSWER))
        r = PipelineResult(question=question, answer=ANSWER, transformed_queries=[question],
                           reranked=[RetrievedChunk(content="근거 " * 200, source="2004.04906", title="DPR",
                                                    rerank_score=0.9)],
                           verification=VerificationResult(grounded=True, faithfulness=0.95))
        return r


class FakeLoader:
    def __init__(self, pipe, ready_after: float = 0.0, error: str | None = None):
        self.t0 = time.monotonic()
        self.pipe, self.ready_after, self._error = pipe, ready_after, error
        self.done = [("임베딩 모델 로딩", 0.5)]
        self.step = "BM25 키워드 색인 구축"

    @property
    def state(self) -> str:
        if self._error:
            return "error"
        return "ready" if time.monotonic() - self.t0 >= self.ready_after else "loading"

    @property
    def error(self):
        return self._error

    @property
    def ready(self) -> bool:
        return self.state == "ready"

    def since_start(self) -> float:
        return time.monotonic() - self.t0


_TMP = tempfile.mkdtemp(prefix="rag-test-")


def make_client(pipe, qlog: QueryLog | None = None, **loader_kw) -> tuple[TestClient, Runtime]:
    rt = Runtime(loader_factory=lambda: FakeLoader(pipe, **loader_kw), llm_status_fn=lambda: ("ok", "fake"))
    qlog = qlog or QueryLog(Path(_TMP) / f"log-{time.monotonic_ns()}.db")   # 실제 기록 DB 를 건드리지 않게
    conv = ConversationStore(Path(_TMP) / f"conv-{time.monotonic_ns()}.db")
    return TestClient(create_app(rt, frontend_dist=None, query_log=qlog, conversations=conv)), rt


def read_events(resp) -> list[tuple[str, dict]]:
    events, ev = [], None
    for line in resp.iter_lines():
        if line.startswith("event: "):
            ev = line[7:]
        elif line.startswith("data: "):
            events.append((ev, json.loads(line[6:])))
    return events


def test_chat_stream() -> None:
    pipe = FakePipeline()
    client, _ = make_client(pipe)
    keep = (settings.query_transform_mode, settings.top_k_rerank)
    with client, client.stream("POST", "/api/chat", json={
            "question": "리랭킹은 왜 필요한가요?",
            "options": {"query_transform_mode": "hyde", "top_k_rerank": 3}}) as r:
        assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
        events = read_events(r)
    names = [e for e, _ in events]
    assert names[0] == "start" and names[-2:] == ["result", "end"], names
    tokens = [d["delta"] for e, d in events if e == "token"]
    assert len(tokens) > 5 and "".join(tokens) == ANSWER, "토큰은 새 글자만, 이어 붙이면 답변"
    result = dict(events)["result"]
    assert result["answer"] == ANSWER and result["sources"][0]["source"] == "2004.04906"
    assert result["sources"][0]["snippet"].endswith("..."), "근거는 앞부분만 보낸다"
    assert result["verification"]["grounded"] is True
    assert pipe.seen_settings == {"mode": "hyde", "top_k": 3}, "요청 옵션이 실행 중 적용돼야 함"
    assert (settings.query_transform_mode, settings.top_k_rerank) == keep, "끝나면 설정이 원래대로"
    print("  test_chat_stream ✅")


def test_bad_request() -> None:
    client, _ = make_client(FakePipeline())
    with client:
        assert client.post("/api/chat", json={"question": ""}).status_code == 422
        assert client.post("/api/chat", json={"question": "q", "options": {"top_k_rerank": 99}}).status_code == 422
    print("  test_bad_request ✅")


def test_waiting_then_ready() -> None:
    client, rt = make_client(FakePipeline(), ready_after=1.2)
    with client:
        st = client.get("/api/status").json()
        assert st["loader"]["state"] == "loading" and st["llm"]["state"] == "ok"
        with client.stream("POST", "/api/chat", json={"question": "q"}) as r:
            names = [e for e, _ in read_events(r)]
    assert names[0] == "waiting" and "ready" in names and names[-1] == "end", names
    print("  test_waiting_then_ready ✅")


def test_not_ready_fails() -> None:
    client, _ = make_client(FakePipeline(), error="CUDA OOM")
    with client, client.stream("POST", "/api/chat", json={"question": "q"}) as r:
        events = read_events(r)
    assert events[-1][0] == "fail" and "CUDA OOM" in events[-1][1]["message"], events
    print("  test_not_ready_fails ✅")


def test_disconnect_stops_job() -> None:
    pipe = FakePipeline(endless=True)
    client, rt = make_client(pipe)
    with client:
        with client.stream("POST", "/api/chat", json={"question": "q"}) as r:
            n = 0
            for line in r.iter_lines():
                n += line.startswith("event: token")
                if n >= 5:
                    break   # 화면이 중지 버튼을 누름
        assert pipe.stopped.wait(5), "연결이 끊기면 작업이 멈춰야 함"
        assert rt.lock.acquire(timeout=5), "멈춘 뒤 LLM 작업 잠금이 풀려야 함"
        rt.lock.release()
    print("  test_disconnect_stops_job ✅")


def test_config_and_citations() -> None:
    client, _ = make_client(FakePipeline())
    with client:
        cfg = client.get("/api/config").json()
        assert set(cfg["defaults"]) >= {"query_transform_mode", "top_k_rerank", "auto_fetch"}
        c = client.get("/api/citations", params={"sources": ["없는-논문"]}).json()
        assert c["papers"] == []
    print("  test_config_and_citations ✅")


def test_query_log_and_feedback() -> None:
    qlog = QueryLog(Path(_TMP) / "feedback.db")
    client, _ = make_client(FakePipeline(), qlog=qlog)
    with client:
        with client.stream("POST", "/api/chat", json={"question": "리랭킹?", "options": {"top_k_rerank": 3}}) as r:
            result = dict(read_events(r))["result"]
        lid = result["log_id"]
        assert lid, "결과에 기록 id 가 있어야 함"
        assert client.post("/api/feedback", json={"log_id": lid, "rating": -1, "note": "근거 부족"}).json() == {"ok": True}
        assert client.post("/api/feedback", json={"log_id": "없음", "rating": 1}).status_code == 404
        assert client.post("/api/feedback", json={"log_id": lid, "rating": 5}).status_code == 422
        logs = client.get("/api/logs", params={"rating": -1}).json()
    item = logs["items"][0]
    assert logs["stats"] == {"total": 1, "up": 0, "down": 1, "errors": 0}, logs["stats"]
    assert item["question"] == "리랭킹?" and item["answer"] == ANSWER and item["note"] == "근거 부족"
    assert item["sources"][0]["source"] == "2004.04906" and item["options"] == {"top_k_rerank": 3}
    # 실패한 질문도 기록된다
    client, _ = make_client(FakePipeline(), qlog=qlog, error="CUDA OOM")
    with client, client.stream("POST", "/api/chat", json={"question": "q"}) as r:
        read_events(r)
    assert qlog.stats()["errors"] == 1
    client, _ = make_client(FakePipeline(), qlog=qlog)
    with client:
        m = client.get("/api/metrics").json()
    assert m["enabled"] and m["total"] == 2 and m["errors"] == 1 and m["down"] == 1
    assert m["by_day"][-1]["n"] == 2 and len(m["by_day"]) == 14 and m["recent"][0]["error"]
    print("  test_query_log_and_feedback ✅")


def test_label_api() -> None:
    """사람 판정: 문항 목록, 판정 저장(없는 문항·문장 번호는 거절), 다시 읽으면 남아 있음."""
    keep = (settings.human_set_path, settings.human_labels_path)
    settings.human_set_path = str(Path(_TMP) / "human_set.jsonl")
    settings.human_labels_path = str(Path(_TMP) / "human_labels.json")
    item = {"id": "q1", "kind": "single", "question": "DPR?", "expected_sources": ["2004.04906"], "answer": "A [1]. B [1].",
            "sources": [{"source": "2004.04906", "title": "DPR", "section": "", "kind": "", "content": "x"}],
            "llm_verification": {"grounded": True, "faithfulness": 1.0, "unsupported": []},
            "citation_check": {"threshold": 0.05, "sentences": [{"text": "A", "cites": [1], "status": "supported"},
                                                                 {"text": "B", "cites": [1], "status": "weak"}]}}
    Path(settings.human_set_path).write_text(json.dumps(item, ensure_ascii=False) + "\n", encoding="utf-8")
    try:
        client, _ = make_client(FakePipeline())
        with client:
            st = client.get("/api/label/items").json()
            assert len(st["items"]) == 1 and st["done"] == 0
            body = {"sentences": {"0": "supported", "1": "unsupported"}, "overall": "partial", "note": "B 근거 없음"}
            assert client.post("/api/label/q1", json=body).json() == {"ok": True, "done": 1}
            assert client.post("/api/label/없음", json=body).status_code == 404
            assert client.post("/api/label/q1", json={"sentences": {"5": "supported"}}).status_code == 422
            assert client.post("/api/label/q1", json={"overall": "maybe"}).status_code == 422
            st = client.get("/api/label/items").json()
        assert st["labels"]["q1"]["sentences"]["1"] == "unsupported" and st["done"] == 1
    finally:
        settings.human_set_path, settings.human_labels_path = keep
    print("  test_label_api ✅")


def test_enricher_waits_for_idle() -> None:
    """표·그림·수식 백그라운드 처리: 사용자가 조용할 때만 시작, 페이지마다 작업 잠금, 끝나면 청크 추가·BM25 재구성·기록."""
    from rag.ingest import enrich, visual
    from rag.stages import stage2_rerank as s2
    from server.enricher import Enricher

    class VS:
        def __init__(self):
            self.added = []

        def add_documents(self, docs):
            self.added += docs

    pipe = FakePipeline()
    pipe.vectorstore = VS()
    rt = Runtime(loader_factory=lambda: FakeLoader(pipe), llm_status_fn=lambda: ("ok", "fake"))
    rt.start()
    rt.loader.pipe = pipe
    seen = {"locked": [], "marked": None, "reset": 0}

    def fake_enrich(src, meta, vs, kinds, progress=None, guard=None, add=None):
        for i in range(3):
            with guard():
                seen["locked"].append(rt.lock.locked())
                progress("visual_page", i=i + 1, n=3, page=i + 1, labels=["Table 1"])
        add(["doc"])
        return [{"label": "Table 1", "kind": "table", "page": 1, "number_match": 1.0}]

    keep = (enrich.enrich, enrich.mark_done, enrich.load_done, visual.default_kinds, s2.reset_bm25)
    enrich.enrich, visual.default_kinds = fake_enrich, lambda: ("table",)
    enrich.mark_done = lambda src, kinds, items: seen.__setitem__("marked", (src, kinds, len(items)))
    enrich.load_done = lambda: {}
    s2.reset_bm25 = lambda: seen.__setitem__("reset", seen["reset"] + 1)
    try:
        en = Enricher(rt, idle=0.4, poll=0.05, scan_on_start=False)
        rt.last_active = time.monotonic()      # 방금 질문이 끝남
        en.enqueue("2604.20606", {"source": "2604.20606", "title": "Beyond ZOH"})
        time.sleep(0.2)
        assert not seen["locked"], "조용해지기 전에는 시작하지 않음"
        deadline = time.monotonic() + 5
        while seen["marked"] is None and time.monotonic() < deadline:
            time.sleep(0.05)
        en.stop()
    finally:
        enrich.enrich, enrich.mark_done, enrich.load_done, visual.default_kinds, s2.reset_bm25 = keep
    assert seen["locked"] == [True, True, True], "페이지마다 작업 잠금을 잡음"
    assert not rt.lock.locked(), "끝나면 잠금을 놓음"
    assert pipe.vectorstore.added == ["doc"] and seen["reset"] == 1
    assert seen["marked"] == ("2604.20606", ("table",), 1)
    assert en.status()["recent"][-1]["source"] == "2604.20606" and en.status()["current"] is None
    print("  test_enricher_waits_for_idle ✅")


def test_conversations_api() -> None:
    """대화 저장: 저장(제목은 첫 질문), 목록, 다시 읽기, 빈 목록이면 삭제."""
    client, _ = make_client(FakePipeline())
    msgs = [{"id": "u1", "role": "user", "content": "리랭킹은 RAG에서 왜 필요한가요? " * 3},
            {"id": "a1", "role": "assistant", "content": "정밀도를 높입니다 [1].", "phase": "done"}]
    with client:
        r = client.put("/api/conversations/c1", json={"messages": msgs}).json()
        assert r["n"] == 2 and r["title"].startswith("리랭킹은 RAG에서") and r["title"].endswith("…")
        client.put("/api/conversations/c2", json={"messages": msgs[:1], "title": "두 번째"})
        items = client.get("/api/conversations").json()["items"]
        assert [i["id"] for i in items] == ["c2", "c1"], "최근 저장 순"
        got = client.get("/api/conversations/c1").json()
        assert got["messages"][1]["content"] == "정밀도를 높입니다 [1]."
        assert client.put("/api/conversations/c2", json={"messages": []}).json()["deleted"] is True
        assert client.get("/api/conversations/c2").status_code == 404
        assert client.delete("/api/conversations/c1").json() == {"ok": True}
        assert client.get("/api/conversations").json()["items"] == []
    print("  test_conversations_api ✅")


def main() -> None:
    print("서버 테스트 실행:")
    test_chat_stream()
    test_bad_request()
    test_waiting_then_ready()
    test_not_ready_fails()
    test_disconnect_stops_job()
    test_config_and_citations()
    test_query_log_and_feedback()
    test_label_api()
    test_enricher_waits_for_idle()
    test_conversations_api()
    print("모든 서버 테스트 통과 ✅")


if __name__ == "__main__":
    main()
