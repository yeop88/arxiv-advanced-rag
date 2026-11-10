"""
ui_progress.py — 챗봇에서 4단계 진행 상황을 실시간으로 보여주는 화면 도우미

파이프라인은 백그라운드 스레드에서 돌리고(진행 알림을 큐에 넣기만 함),
화면 갱신은 Streamlit 메인 스레드가 0.5초마다 한다. 그래서 알림이 없는 긴 구간
(예: 검증 20초)에도 경과 시간이 계속 오르고, 답변은 생성되는 대로 표시된다.
"""
from __future__ import annotations

import streamlit as st


def _handle_event(event: str, info: dict, status, stage: str) -> str:
    """진행 알림 하나를 화면 로그로 옮기고, 현재 단계 이름을 돌려준다."""
    if event == "contextualize_start":
        return "0/4 · 대화 맥락 반영 중"
    if event == "contextualize_done":
        if info.get("changed"):
            status.write(f"🔁 이전 대화를 반영해 질문을 바꿨습니다: **{info['standalone']}**")
        return stage
    if event == "transform_start":
        status.write(f"1️⃣ **질의 변환** — 검색이 잘 되도록 질문을 여러 형태로 바꾸는 중 (`{info.get('mode')}`)")
        return "1/4 · 질의 변환 중"
    if event == "transform_done":
        qs = info.get("queries") or []
        status.write(f"　✓ 검색 질의 {len(qs)}개 ({info['sec']:.1f}초)")
        for q in qs:
            status.caption(f"　　· {q}")
        return stage
    if event == "retrieve_start":
        status.write(f"2️⃣ **검색·리랭킹** — 질의 {info.get('n_queries')}개로 논문을 찾고 관련도순으로 고르는 중")
        return "2/4 · 검색·리랭킹 중"
    if event == "retrieve_done":
        papers = info.get("papers") or []
        status.write(f"　✓ 후보 {info.get('n_candidates')}개 중 근거 {len(papers)}개 선택 ({info['sec']:.1f}초)")
        for i, p in enumerate(papers, 1):
            status.caption(f"　　[{i}] {p[:80]}")
        return stage
    if event == "generate_start":
        status.write(f"3️⃣ **답변 작성** — 근거 {info.get('n_chunks')}개로 답변을 쓰는 중 (아래에 실시간 표시)")
        return "3/4 · 답변 작성 중"
    if event == "generate_done":
        status.write(f"　✓ 답변 작성 완료 ({info['sec']:.1f}초, {info.get('chars')}자)")
        return stage
    if event == "verify_start":
        again = "다시 " if info.get("again") else ""
        status.write(f"4️⃣ **답변 {again}검증** — 답변의 각 주장이 근거에 실제로 있는지 대조하는 중")
        return f"4/4 · 답변 {again}검증 중"
    if event == "verify_done":
        ok = "✅ 근거 충실" if info.get("grounded") else "⚠️ 근거 부족"
        n_bad = len(info.get("unsupported") or [])
        status.write(f"　✓ {ok} · faithfulness {info.get('faithfulness', 0):.2f}"
                     + (f" · 근거 없는 주장 {n_bad}개" if n_bad else "") + f" ({info['sec']:.1f}초)")
        return stage
    if event == "regenerate_start":
        status.write(f"🔁 **다시 작성** (시도 {info.get('attempt')}) — 근거 없는 주장을 빼고 답변을 고치는 중")
        return "3/4 · 답변 다시 작성 중"
    if event == "error":
        status.write(f"⚠️ {info.get('stage')} 단계 오류: {info.get('message', '')[:200]}")
        return stage
    return stage


def wait_until_ready(loader, llm_status_fn, timeout: float = 1800, down_grace: float = 60):
    """모델·DB 로딩과 LLM 서버 준비를 기다린 뒤 파이프라인을 돌려준다(이미 준비됐으면 즉시).

    LLM 서버가 '모델 올리는 중'이면 계속 기다리고, 아예 응답이 없는 상태가 down_grace 초를
    넘으면 안내를 띄우고 멈춘다(서버가 꺼진 경우 무한 대기 방지).
    """
    import time

    llm_state, llm_desc = llm_status_fn()
    if loader.ready and llm_state in ("ok", "n/a"):
        return loader.pipe

    status = st.status("⏳ 준비되면 바로 답합니다", expanded=True)
    t0 = time.perf_counter()
    shown = 0
    down_since = None
    while True:
        while shown < len(loader.done):  # 끝난 로딩 단계를 로그로
            name, sec = loader.done[shown]
            status.write(f"✓ {name} ({sec:.1f}초)")
            shown += 1
        if loader.state == "error":
            status.update(label="⚠️ 모델·DB 준비 실패", state="error", expanded=True)
            st.error(f"준비 중 오류: {loader.error}")
            st.stop()
        llm_state, llm_desc = llm_status_fn()
        if loader.ready and llm_state in ("ok", "n/a"):
            break
        waited = time.perf_counter() - t0
        if llm_state == "down":
            down_since = down_since or time.perf_counter()
            if time.perf_counter() - down_since > down_grace:
                status.update(label="⚠️ LLM 서버가 응답하지 않습니다", state="error", expanded=True)
                st.error(f"LLM 서버 {llm_desc}. FreeToken 이 실행 중인지, .env 의 LOCAL_BASE_URL(포트)을 확인하세요.")
                st.stop()
        else:
            down_since = None
        if waited > timeout:
            status.update(label="⚠️ 준비 대기 시간 초과", state="error", expanded=True)
            st.stop()
        parts = []
        if not loader.ready:
            parts.append(f"모델·DB: {loader.step} ({len(loader.done)}/4)")
        if llm_state not in ("ok", "n/a"):
            parts.append(f"LLM 서버: {llm_desc}")
        status.update(label="⏳ 준비되면 바로 답합니다 · " + " / ".join(parts) + f" · {waited:.0f}초")
        time.sleep(1)
    status.update(label=f"✅ 준비 완료 ({time.perf_counter() - t0:.0f}초 기다림)", state="complete", expanded=False)
    return loader.pipe


def run_with_progress(pipe, prompt: str, history):
    """파이프라인을 백그라운드에서 돌리며 단계별 진행·경과 시간·답변 초안을 실시간으로 그린다."""
    import queue
    import threading
    import time

    events: queue.Queue = queue.Queue()
    box: dict = {}

    def worker():
        try:
            box["result"] = pipe.run(prompt, chat_history=history,
                                     progress=lambda ev, **kw: events.put((ev, kw)))
        except Exception as e:  # noqa: BLE001
            box["error"] = e

    status = st.status("질문을 받았습니다 · 처리 준비 중", expanded=True)
    answer_box = st.empty()
    t0 = time.perf_counter()
    stage = "준비 중 (모델·벡터DB 로딩)"
    thread = threading.Thread(target=worker, daemon=True)
    thread.start()

    while thread.is_alive() or not events.empty():
        draft = None
        try:
            ev, info = events.get(timeout=0.5)
            while True:  # 쌓인 알림을 한 번에 처리(토큰은 마지막 것만 그림)
                if ev == "token":
                    draft = info["text"]
                elif ev == "regenerate_start":
                    answer_box.empty()
                    stage = _handle_event(ev, info, status, stage)
                else:
                    stage = _handle_event(ev, info, status, stage)
                ev, info = events.get_nowait()
        except queue.Empty:
            pass
        if draft is not None:
            answer_box.markdown(draft + " ▌")
        status.update(label=f"{stage} · {time.perf_counter() - t0:.0f}초 경과")

    total = time.perf_counter() - t0
    if "error" in box:
        status.update(label=f"⚠️ 오류로 중단 · {total:.0f}초", state="error", expanded=True)
        st.error(f"처리 중 오류가 발생했습니다: {box['error']}")
        st.stop()
    result = box["result"]
    tm = result.timings or {}
    parts = [("변환", "1_query_transform"), ("검색", "2_retrieve_rerank"), ("작성", "3_generate"), ("검증", "4_verify")]
    detail = " · ".join(f"{n} {tm[k]:.0f}s" for n, k in parts if k in tm)
    status.update(label=f"✅ 완료 · 총 {total:.0f}초 ({detail})", state="complete", expanded=False)
    answer_box.markdown(result.answer)
    return result


