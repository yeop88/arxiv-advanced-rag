// 화면 구성: 사이드바(상태·대화 목록·설정·도구) + 본문 탭(대화 / 지표 / 사람 판정). 탭은 주소의 #metrics 등으로 기억한다.
import { useCallback, useEffect, useState } from "react";
import { api } from "./api/client";
import type { AppConfig, ChatOptions, SurveyResult, WatchState } from "./api/types";
import { ChatView } from "./components/chat/ChatView";
import { Composer } from "./components/chat/Composer";
import { LabelView } from "./components/label/LabelView";
import { MetricsView } from "./components/metrics/MetricsView";
import { Sidebar } from "./components/sidebar/Sidebar";
import { SurveyCard } from "./components/sidebar/SurveyPanel";
import { WatchDigestCard } from "./components/sidebar/WatchPanel";
import { useChat } from "./hooks/useChat";
import { useJob } from "./hooks/useJob";
import { useStatus } from "./hooks/useStatus";
import { load, save } from "./lib/storage";

const OPTIONS_KEY = "arxiv-rag.options.v1";

export type View = "chat" | "metrics" | "label";
const VIEWS: { v: View; label: string }[] = [
  { v: "chat", label: "💬 대화" },
  { v: "metrics", label: "📈 지표" },
  { v: "label", label: "🧑‍⚖️ 사람 판정" },
];
const viewFromHash = (): View => (VIEWS.some((x) => `#${x.v}` === window.location.hash) ? (window.location.hash.slice(1) as View) : "chat");
const FALLBACK: ChatOptions = {
  query_transform_mode: "multi_query",
  use_hybrid: true,
  top_k_rerank: 5,
  enable_verification: true,
  auto_fetch: true,
  local_thinking: "off",
  local_thinking_light: "off",
};

export default function App() {
  const [config, setConfig] = useState<AppConfig | null>(null);
  const [options, setOptions] = useState<ChatOptions>(() => ({ ...FALLBACK, ...load<Partial<ChatOptions>>(OPTIONS_KEY, {}) }));
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [view, setView] = useState<View>(viewFromHash);
  const [watch, setWatch] = useState<WatchState | null>(null);
  const { status, offline } = useStatus();
  const chat = useChat();
  const survey = useJob<SurveyResult>("/api/survey");
  const watchRun = useJob<WatchState>("/api/watch/run");

  useEffect(() => {
    api
      .config()
      .then((c) => {
        setConfig(c);
        // 서버 기본값 위에 사용자가 고른 값(브라우저에 저장)을 덮는다
        setOptions({ ...c.defaults, ...load<Partial<ChatOptions>>(OPTIONS_KEY, {}) });
      })
      .catch(() => undefined);
    api.watch().then(setWatch).catch(() => undefined);
  }, []);

  useEffect(() => {
    if (watchRun.result) setWatch(watchRun.result);
  }, [watchRun.result]);

  useEffect(() => {
    const onHash = () => setView(viewFromHash());
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);
  const go = (v: View) => {
    window.location.hash = v === "chat" ? "" : v;
    setView(v);
  };

  const changeOptions = useCallback((o: ChatOptions) => {
    setOptions(o);
    save(OPTIONS_KEY, o);
  }, []);
  const send = useCallback((q: string) => chat.send(q, options), [chat, options]);

  return (
    <div className="app">
      <Sidebar
        open={sidebarOpen}
        onClose={() => setSidebarOpen(false)}
        config={config}
        status={status}
        offline={offline}
        options={options}
        onOptions={changeOptions}
        surveyRunning={survey.running}
        onSurvey={(topic) => survey.start({ topic })}
        watch={watch}
        onWatch={setWatch}
        watchRunning={watchRun.running}
        onWatchRun={() => watchRun.start({})}
        onClear={chat.clear}
        conversations={chat.conversations}
        convId={chat.convId}
        convEnabled={chat.serverStore}
        chatRunning={chat.running}
        onNewChat={() => {
          chat.newChat();
          setSidebarOpen(false);
        }}
        onOpenChat={(id) => {
          void chat.open(id);
          go("chat");
          setSidebarOpen(false);
        }}
        onDeleteChat={(id) => void chat.remove(id)}
      />
      <main className="main">
        <header className="topbar">
          <button type="button" className="icon-btn only-mobile" onClick={() => setSidebarOpen(true)} aria-label="설정 열기">
            ☰
          </button>
          <div>
            <div className="title">📚 arXiv RAG 챗봇</div>
            <div className="muted small">학술논문 근거 기반 질의응답 · 질의변환 → 리랭킹 → 생성 → 검증</div>
          </div>
          <nav className="tabs" aria-label="화면">
            {VIEWS.map((x) => (
              <button key={x.v} type="button" className={view === x.v ? "on" : ""} onClick={() => go(x.v)} aria-current={view === x.v}>
                {x.label}
              </button>
            ))}
          </nav>
        </header>
        {view === "label" && <LabelView />}
        {view === "metrics" && <MetricsView />}
        {view === "chat" && (
          <>
        <ChatView
          messages={chat.messages}
          onRate={chat.rate}
          onExample={send}
          header={
            <>
              <SurveyCard job={survey} onCancel={survey.cancel} onClose={survey.reset} />
              <WatchDigestCard state={watch} job={watchRun} onCancel={watchRun.cancel} />
            </>
          }
        />
        <div className="composer-wrap">
          <Composer running={chat.running} onSend={send} onStop={chat.stop} />
        </div>
          </>
        )}
      </main>
      {sidebarOpen && <div className="scrim only-mobile" onClick={() => setSidebarOpen(false)} />}
    </div>
  );
}

