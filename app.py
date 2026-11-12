"""
app.py — Streamlit 챗봇형 RAG 테스트/배포 UI

대화형 챗봇 인터페이스로 arXiv Advanced RAG 를 테스트·시연합니다.
- 상단: 대화 히스토리(사용자 질문 / AI 답변)
- 각 답변 아래: 검증 배지 + "단계별 내부 동작" 펼쳐보기 (학습·디버깅용)
- 사이드바: 파이프라인 설정 실시간 조정

실행: streamlit run app.py
배포: streamlit 커뮤니티 클라우드 / 사내 서버에 그대로 올릴 수 있음
      (단, ANTHROPIC_API_KEY 는 secrets 로 주입, 벡터DB는 사전 구축 필요)
"""
from __future__ import annotations

import streamlit as st

from config import settings
from rag.warmup import PipelineLoader, llm_status
from ui_progress import run_with_progress, wait_until_ready

st.set_page_config(page_title="arXiv RAG 챗봇", page_icon="📚", layout="centered")


@st.cache_resource(show_spinner=False)
def get_loader() -> PipelineLoader:
    """앱 프로세스당 1개. 만들어지는 즉시 백그라운드에서 임베딩·리랭커·벡터DB·BM25 를 올린다.
    페이지가 처음 열리는 순간 시작되므로, 사용자가 질문을 입력하는 동안 로딩이 진행된다."""
    return PipelineLoader()


loader = get_loader()


@st.fragment(run_every=2)
def status_panel() -> None:
    """사이드바 상태 표시 — 2초마다 갱신(로딩 진행, LLM 서버 상태)."""
    if loader.state == "ready":
        st.success(f"모델·DB 준비 완료 ({loader.since_start():.0f}초)", icon="✅")
    elif loader.state == "error":
        st.error(f"모델·DB 준비 실패: {loader.error}")
    else:
        st.info(f"모델·DB 로딩 중 ({len(loader.done)}/4)\n\n{loader.step} · {loader.since_start():.0f}초", icon="⏳")
    state, desc = llm_status()
    if state == "ok":
        st.success(f"LLM 서버: {desc}", icon="✅")
    elif state == "loading":
        st.warning(f"LLM 서버: {desc}", icon="⏳")
    elif state == "down":
        st.error(f"LLM 서버: {desc}")
    else:
        st.caption(f"LLM: {desc}")


# ---------------------------------------------------------------------------
# 사이드바: 파이프라인 설정
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("⚙️ 설정")
    status_panel()
    settings.query_transform_mode = st.selectbox(
        "1단계 · 질의 변환",
        options=["multi_query", "hyde", "decompose", "none"],
        index=0,
        help="multi_query: 여러 각도 / hyde: 가상답변 / decompose: 하위질문 분해",
    )
    settings.use_hybrid = st.checkbox("2단계 · 하이브리드 검색(BM25+벡터)", value=True)
    settings.top_k_rerank = st.slider("2단계 · 근거 수(top-k)", 1, 10, settings.top_k_rerank)
    settings.enable_verification = st.checkbox("4단계 · 검증", value=True)
    settings.auto_fetch = st.toggle(
        "📥 근거 부족 시 논문 자동 수집",
        value=settings.auto_fetch,
        key="auto_fetch",
        help="DB 에 맞는 근거가 없으면 arXiv 에서 관련 논문을 찾아 전문을 DB 에 추가하고 다시 검색합니다"
             "(관련도 기준을 넘는 논문만, 질문당 최대 3편). 추가하면 질문이 수십 초 더 걸립니다.",
    )

    # --- Thinking 켜기/끄기 ---
    # Qwen3 같은 로컬 모델은 thinking 을 on/off 로 제어한다(Claude·OpenAI 는 등급형 effort → .env 에서 설정).
    st.markdown("---")
    if settings.llm_provider == "local":
        think_main = st.toggle(
            "🧠 Thinking 모드 (생성·검증)",
            value=settings.local_thinking == "on",
            key="think_main",
            help="켜면 답하기 전에 추론해 다중 홉 비교·종합에 유리하지만 느려집니다"
                 "(측정: 질문당 off 약 1분, on 약 3~4분).",
        )
        settings.local_thinking = "on" if think_main else "off"
        think_light = st.toggle(
            "질의변환에도 Thinking",
            value=settings.local_thinking_light == "on",
            key="think_light",
            help="질의변환은 쉬운 작업이라 보통 꺼 둡니다(켜면 질문마다 수십 초 추가).",
        )
        settings.local_thinking_light = "on" if think_light else "off"
        st.caption("⏱️ 현재: " + ("생각하며 답함 — 질문당 수 분" if think_main else "빠른 답변 — 질문당 약 1분"))
    else:
        st.caption("🧠 Thinking 토글은 로컬 모델(LLM_PROVIDER=local) 전용입니다. "
                   "Claude·OpenAI 는 .env 의 effort 설정을 쓰세요.")

    st.markdown("---")
    st.subheader("📝 문헌 조사 초안")
    survey_topic = st.text_input("주제", placeholder="예: 검색 증강 생성(RAG)의 발전 흐름", key="survey_topic")
    survey_clicked = st.button("초안 만들기", disabled=not survey_topic.strip())
    st.caption("DB 의 관련 논문 5~8편을 소주제별 흐름으로 묶어 인용이 달린 초안을 씁니다(로컬 모델 약 1~2분).")

    st.markdown("---")
    st.subheader("📬 관심 분야 알림")
    from rag import watch as _watch

    _topics = _watch.load()["topics"]
    for _t in _topics:
        st.caption(f"• {_t['topic']} — 마지막 확인 {_t.get('last_checked') or '아직 없음'}")
    watch_new = st.text_input("주제 추가", placeholder="예: 검색 증강 생성", key="watch_new")
    _c1, _c2 = st.columns(2)
    watch_add = _c1.button("추가", disabled=not watch_new.strip())
    watch_run = _c2.button("새 논문 확인", disabled=not _topics)
    st.caption("arXiv 최신 논문 중 관련 있는 것을 골라 초록을 한국어로 요약합니다(주제당 약 30초). "
               "정기 실행은 `scripts/watch_papers.py --run`.")

    st.markdown("---")
    st.caption(f"LLM: `{settings.llm_provider}` / {settings.llm_model}")
    st.caption(f"임베딩: `{settings.embed_model}`")
    if st.button("🗑️ 대화 초기화"):
        st.session_state.messages = []
        st.rerun()

# ---------------------------------------------------------------------------
# 헤더
# ---------------------------------------------------------------------------
st.title("📚 arXiv RAG 챗봇")
st.caption("학술논문 근거 기반 질의응답 · 질의변환→리랭킹→생성→검증 4단계")

# ---------------------------------------------------------------------------
# 대화 상태
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []  # [{role, content, result?}]


def render_survey() -> None:
    """사이드바에서 요청한 문헌 조사 초안을 만들고(버튼을 누른 실행에서) 화면 위쪽에 보여 준다."""
    if survey_clicked:
        if not loader.ready:
            st.info("모델·DB 를 준비하는 중입니다. 잠시 뒤 다시 눌러 주세요.")
            return
        from rag.survey import build_survey

        with st.status(f"문헌 조사 초안: {survey_topic}", expanded=True) as box:
            def progress(ev, **kw):
                msgs = {"survey_gather": "관련 논문 모으는 중", "survey_verify": "초안 검증 중"}
                if ev == "survey_write":
                    box.write(f"근거 {kw['n']}개(논문 {kw['papers']}편)로 초안 작성 중")
                elif ev in msgs:
                    box.write(msgs[ev])

            st.session_state.survey = build_survey(survey_topic.strip(), loader.pipe.vectorstore, progress=progress)
            box.update(label="초안 완성", state="complete", expanded=False)
    sv = st.session_state.get("survey")
    if not sv:
        return
    with st.expander(f"📝 문헌 조사 초안 — {sv.topic}", expanded=True):
        st.markdown(sv.text)
        st.markdown("**참고문헌**  \n" + "  \n".join(sv.references_md().splitlines()))
        v = sv.verification
        st.caption(("✅ 검증 통과" if v.grounded else "⚠️ 근거와 다를 수 있는 문장: " + "; ".join(v.unsupported_claims))
                   if v else "검증 생략")
        st.download_button("마크다운으로 저장", sv.to_markdown(), file_name="survey_draft.md", mime="text/markdown")


render_survey()


def render_watch() -> None:
    """관심 분야: 주제 추가·새 논문 확인(버튼을 누른 실행에서), 최신 소식지 표시."""
    from rag import watch

    if watch_add:
        with st.spinner("검색어를 만드는 중"):
            watch.add_topic(watch_new.strip())
        st.rerun()
    if watch_run:
        if not loader.ready:
            st.info("모델·DB 를 준비하는 중입니다. 잠시 뒤 다시 눌러 주세요.")
        else:
            with st.status("관심 분야 새 논문 확인", expanded=True) as box:
                watch.run(loader.pipe.vectorstore,
                          progress=lambda ev, **kw: box.write(f"{kw['i']}/{kw['n']} {kw['topic']}"))
                box.update(label="소식지 완성", state="complete", expanded=False)
    latest = watch.latest_digest()
    if latest:
        day, md = latest
        with st.expander(f"📬 관심 분야 소식 ({day})", expanded=bool(watch_run)):
            st.markdown(md, unsafe_allow_html=True)


render_watch()


def _thinking_label() -> str:
    """답변에 붙일 thinking 모드 표시(로컬 모델일 때만)."""
    if settings.llm_provider != "local":
        return ""
    return f"🧠 thinking {settings.local_thinking or '서버 기본값'} (질의변환 {settings.local_thinking_light or '기본값'})"


def render_sources(result) -> None:
    """답변의 [n] 인용에 대응하는 근거 출처 패널.

    reranked 순서 = 생성 프롬프트의 [1][2]... 번호와 일치하므로, 그대로 나열하면
    사용자가 각 인용의 원문·섹션·arXiv 링크를 바로 확인할 수 있다.
    """
    if not result.reranked:
        return
    with st.expander(f"📑 근거 출처 {len(result.reranked)}개 (인용 [n] 대응)", expanded=False):
        for i, c in enumerate(result.reranked, 1):
            sec = f" · {c.section}" if c.section else ""
            link = c.link
            title_md = f"[{c.title}]({link})" if link else (c.title or "unknown")
            st.markdown(f"**[{i}]** {title_md}  \n"
                        f"<small>arXiv:{c.source or 'N/A'}{sec}</small>",
                        unsafe_allow_html=True)
            snippet = c.content.strip()
            st.caption(snippet[:300] + ("..." if len(snippet) > 300 else ""))


def render_citations(result, key: str) -> None:
    """근거 논문의 인용 관계: 코퍼스 안 선행(이 논문이 인용)·후속(이 논문을 인용) 논문.
    코퍼스 밖 주요 논문은 체크하면 OpenAlex 에서 조회(처음 한 번, 이후 캐시)."""
    from rag import citations

    g = citations.load()
    papers = [s for s in dict.fromkeys(c.source for c in result.reranked) if s in g["papers"]]
    if not papers:
        return
    with st.expander(f"🔗 인용 관계 — 근거 논문 {len(papers)}편의 선행·후속 연구", expanded=False):
        external = settings.openalex_enabled and st.checkbox("코퍼스 밖 주요 논문도 보기 (OpenAlex 조회)", key=f"cite_ext_{key}")
        for src in papers:
            me = g["papers"][src]
            rel = citations.related_in_corpus(src, g)
            st.markdown(f"**{me['title']}** <small>({me.get('year') or '-'}, 피인용 {me['cited_by']:,})</small>",
                        unsafe_allow_html=True)

            def names(ids):
                return " · ".join(f"[{g['papers'][a]['title'][:50]}](https://arxiv.org/abs/{a})" for a in ids[:3]) or "없음"

            st.markdown(f"- 선행(DB 안 {len(rel['prior'])}편): {names(rel['prior'])}\n"
                        f"- 후속(DB 안 {len(rel['later'])}편): {names(rel['later'])}")
            if external:
                try:
                    pri, lat = citations.prior_external(src, 3), citations.later_external(src, 3)
                except Exception as e:  # noqa: BLE001  (네트워크 실패는 화면만 알림)
                    st.caption(f"OpenAlex 조회 실패: {type(e).__name__}")
                    continue

                def ext(items):
                    return " · ".join(
                        (f"[{w['title'][:50]}](https://arxiv.org/abs/{w['arxiv']})" if w.get("arxiv") else w["title"][:50])
                        + f" ({w['cited_by']:,})" for w in items) or "없음"

                st.caption(f"주요 선행: {ext(pri)}")
                st.caption(f"주요 후속: {ext(lat)}")


def render_compare_table(msg: dict, key: str) -> None:
    """근거 논문이 2편 이상이면 '비교표 만들기' 버튼. 만든 표는 메시지에 저장해 다시 그린다."""
    result = msg["result"]
    papers = list(dict.fromkeys(c.source for c in result.reranked if c.source))
    if len(papers) < 2:
        return
    table = msg.get("table")
    if table is None:
        if not st.button(f"📊 근거 논문 {min(len(papers), 5)}편 비교표 만들기", key=f"table_{key}"):
            return
        if not loader.ready:
            st.info("모델·DB 를 준비하는 중입니다. 잠시 뒤 다시 눌러 주세요.")
            return
        from rag.compare_table import build_table

        with st.status("비교표를 만드는 중 (논문마다 요약 → 검증)", expanded=True) as box:
            def progress(ev, **kw):
                if ev == "table_paper":
                    box.write(f"{kw['i']}/{kw['n']} {kw['title'][:60]}")
                elif ev == "table_verify":
                    box.write("표 검증 중")

            table = build_table(papers, loader.pipe.vectorstore, progress=progress)
            box.update(label="비교표 완성", state="complete", expanded=False)
        msg["table"] = table
    with st.expander("📊 논문 비교표 (방법·데이터·결과·한계)", expanded=True):
        st.markdown(table.to_markdown(marks=True))
        v = table.verification
        badge = ("✅ 검증 통과" if v.grounded else "⚠️ 일부 칸이 근거와 다를 수 있음") if v else "검증 생략"
        st.caption(f"{badge} · 채운 칸 {table.cells_filled}/{table.cells_total} · ⚠ 칸은 근거 원문을 확인하세요 · "
                   f"[n]은 아래 근거 번호")
        for i, c in enumerate(table.chunks, 1):
            sec = f" · {c.section}" if c.section else ""
            st.markdown(f"<small>[{i}] {c.title[:60]}{sec}</small>", unsafe_allow_html=True)


def render_inspector(result) -> None:
    """한 답변의 검증 배지 + 단계별 내부 동작을 렌더."""
    if getattr(result, "auto_fetched", None):
        added = result.auto_fetched
        st.info(f"📥 이 질문에 맞는 근거가 부족해 arXiv 논문 {len(added)}편을 DB 에 새로 추가했습니다:\n\n"
                + "\n".join(f"- [{a['title']}](https://arxiv.org/abs/{a['source']}) — "
                            f"{'전문' if a['fulltext'] else '초록'} {a['chunks']}청크" for a in added))
    if result.verification:
        v = result.verification
        c1, c2, c3 = st.columns(3)
        c1.metric("근거 충실", "✅" if v.grounded else "⚠️")
        c2.metric("Faithfulness", f"{v.faithfulness:.2f}")
        c3.metric("재생성", "🔁" if result.regenerated else "–")
        if v.unsupported_claims:
            st.warning("근거 없는 주장: " + "; ".join(v.unsupported_claims))

    with st.expander("🔍 단계별 내부 동작 보기"):
        if result.standalone_question and result.standalone_question != result.question:
            st.markdown("**0단계 · 멀티턴 문맥화**")
            st.markdown(f"- 원 질문: {result.question}")
            st.markdown(f"- 독립형: {result.standalone_question}")
        st.markdown(f"**1단계 · 질의변환 ({len(result.transformed_queries)}개)**")
        for q in result.transformed_queries:
            st.markdown(f"- {q}")
        st.markdown(f"**2단계 · 리랭킹 선택 ({len(result.reranked)}개)**")
        for i, c in enumerate(result.reranked, 1):
            rr = f"{c.rerank_score:.3f}" if c.rerank_score is not None else "–"
            st.markdown(f"{i}. `{c.title[:70]}` (arXiv:{c.source}) · 리랭크={rr}")
        if result.timings:
            st.caption("⏱️ " + " · ".join(f"{k}={v:.1f}s" for k, v in result.timings.items()))


# --- 지난 대화 렌더 ---
for idx, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("mode"):
            st.caption(msg["mode"])  # 이 답변을 만든 thinking 모드
        if msg["role"] == "assistant" and msg.get("result"):
            render_sources(msg["result"])
            render_citations(msg["result"], str(idx))
            render_compare_table(msg, str(idx))
            render_inspector(msg["result"])

# --- 입력 ---
if prompt := st.chat_input("논문에 대해 물어보세요 (예: 리랭킹은 왜 필요한가요?)"):
    # 사용자 메시지
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # AI 응답
    with st.chat_message("assistant"):
        # 모델·DB·LLM 서버가 아직 준비 전이면 진행 상황을 보여주며 기다린 뒤 이어서 답한다
        pipe = wait_until_ready(loader, llm_status)

        # 직전까지의 대화를 (role, content) 히스토리로 전달 → 멀티턴 문맥화
        history = [(m["role"], m["content"]) for m in st.session_state.messages[:-1]]
        mode = _thinking_label()
        result = run_with_progress(pipe, prompt, history)

        if mode:
            st.caption(mode)
        new_msg = {"role": "assistant", "content": result.answer, "result": result, "mode": mode}
        key = str(len(st.session_state.messages))  # 다음 실행에서 이 메시지의 위치(idx)와 같은 키
        render_sources(result)
        render_citations(result, key)
        render_compare_table(new_msg, key)
        render_inspector(result)

    st.session_state.messages.append(new_msg)
