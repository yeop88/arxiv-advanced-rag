// 화면 구성: 아이콘 레일 | (대화 화면) 대화 목록 · 대화 흐름 + 입력 카드 · 근거 패널 | 도구 · 지표 · 판정 화면.
// 화면은 주소의 #tools·#metrics·#label 로 기억하고, 테마(밝게/어둡게)는 브라우저에 기억한다.
import { useCallback, useEffect, useMemo, useState } from "react";
import { api } from "./api/client";
import type { AppConfig, ChatOptions, SurveyResult, WatchState } from "./api/types";
import { EvidenceDrawer } from "./components/answer/EvidenceDrawer";
import { ChatView } from "./components/chat/ChatView";
import { Composer } from "./components/chat/Composer";
import { ConversationList } from "./components/chat/ConversationList";
import type { EvidenceTab, OpenEvidence } from "./components/chat/MessageItem";
import { LabelView } from "./components/label/LabelView";
import { Rail, type View } from "./components/layout/Rail";
import { SettingsDialog } from "./components/layout/SettingsDialog";
import { MetricsView } from "./components/metrics/MetricsView";
import { ToolsView } from "./components/tools/ToolsView";
import { useChat } from "./hooks/useChat";
import { useJob } from "./hooks/useJob";
import { useStatus } from "./hooks/useStatus";
import { load, save } from "./lib/storage";

const OPTIONS_KEY = "arxiv-rag.options.v1";
const THEME_KEY = "arxiv-rag.theme.v1";
const FALLBACK: ChatOptions = {
  query_transform_mode: "multi_query",
  use_hybrid: true,
  top_k_rerank: 5,
  enable_verification: true,
  auto_fetch: true,
  local_thinking: "off",
  local_thinking_light: "off",
};
const VIEWS: View[] = ["chat", "tools", "metrics", "label"];
const viewFromHash = (): View => {
  const h = window.location.hash.slice(1) as View;
  return VIEWS.includes(h) ? h : "chat";
};
const initialTheme = (): "light" | "dark" =>
  load<"light" | "dark" | null>(THEME_KEY, null) ?? (window.matchMedia?.("(prefers-color-scheme: dark)").matches ? "dark" : "light");

interface Evidence {
  msgId: string;
  tab: EvidenceTab;
  cite: number | null;
}

export default function App() {
  const [config, setConfig] = useState<AppConfig | null>(null);
  const [options, setOptions] = useState<ChatOptions>(() => ({ ...FALLBACK, ...load<Partial<ChatOptions>>(OPTIONS_KEY, {}) }));
  const [view, setView] = useState<View>(viewFromHash);
  const [theme, setTheme] = useState<"light" | "dark">(initialTheme);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [convsOpen, setConvsOpen] = useState(false); // 좁은 화면에서 대화 목록 서랍
  const [evidence, setEvidence] = useState<Evidence | null>(null);
  const [watch, setWatch] = useState<WatchState | null>(null);
  const { status, offline } = useStatus();
  const chat = useChat();
  const survey = useJob<SurveyResult>("/api/survey");
  const watchRun = useJob<WatchState>("/api/watch/run");

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    save(THEME_KEY, theme);
  }, [theme]);

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

  const go = useCallback((v: View) => {
    window.location.hash = v === "chat" ? "" : v;
    setView(v);
  }, []);
  const changeOptions = useCallback((o: ChatOptions) => {
    setOptions(o);
    save(OPTIONS_KEY, o);
  }, []);
  const send = useCallback((q: string) => chat.send(q, options), [chat, options]);
  const openEvidence = useCallback<OpenEvidence>((msgId, tab, cite) => setEvidence({ msgId, tab, cite: cite ?? null }), []);

  // 근거 패널이 보는 답변(대화를 바꾸거나 지우면 닫힌다)
  const evMsg = useMemo(() => (evidence ? chat.messages.find((m) => m.id === evidence.msgId) : undefined), [evidence, chat.messages]);
  const evIndex = evMsg ? chat.messages.indexOf(evMsg) : -1;
  const evQuestion = evIndex > 0 ? chat.messages[evIndex - 1].content : (evMsg?.result?.question ?? "");
  useEffect(() => {
    if (evidence && !evMsg?.result) setEvidence(null);
  }, [evidence, evMsg]);

  const title = useMemo(() => chat.conversations.find((c) => c.id === chat.convId)?.title ?? (chat.messages[0]?.content || "새 대화"), [chat.conversations, chat.convId, chat.messages]);

  return (
    <div className={`shell view-${view}${evMsg?.result && view === "chat" ? " with-drawer" : ""}`}>
      <Rail view={view} onView={go} onSettings={() => setSettingsOpen(true)} theme={theme} onTheme={() => setTheme(theme === "dark" ? "light" : "dark")} status={status} offline={offline} />

      {view === "chat" && (
        <>
          <div className={`convs-wrap${convsOpen ? " open" : ""}`}>
            <ConversationList
              items={chat.conversations}
              current={chat.convId}
              enabled={chat.serverStore}
              running={chat.running}
              onNew={() => {
                chat.newChat();
                setEvidence(null);
                setConvsOpen(false);
              }}
              onOpen={(id) => {
                void chat.open(id);
                setEvidence(null);
                setConvsOpen(false);
              }}
              onDelete={(id) => void chat.remove(id)}
            />
          </div>
          {convsOpen && <div className="scrim" onClick={() => setConvsOpen(false)} />}
          <main className="stage">
            <header className="stage-head">
              <button type="button" className="icon-btn only-narrow" onClick={() => setConvsOpen(true)} aria-label="대화 목록">
                ☰
              </button>
              <div className="stage-title" title={title}>
                {chat.messages.length ? title : "새 대화"}
              </div>
              {chat.running && <span className="live-pill">답하는 중</span>}
            </header>
            <ChatView messages={chat.messages} onExample={send} onRate={chat.rate} onEvidence={openEvidence} selectedId={evidence?.msgId ?? null} />
            <div className="composer-dock">
              <Composer
                running={chat.running}
                onSend={send}
                onStop={chat.stop}
                options={options}
                onOptions={changeOptions}
                local={config?.provider === "local"}
                autoFocus
              />
              <p className="dock-note">답변은 DB 의 논문을 근거로 하지만 틀릴 수 있어요 — [n] 을 눌러 원문을 확인하세요.</p>
            </div>
          </main>
          {evMsg?.result && evidence && (
            <EvidenceDrawer
              key={evMsg.id}
              result={evMsg.result}
              question={evQuestion}
              tab={evidence.tab}
              highlight={evidence.cite}
              onTab={(tab) => setEvidence({ ...evidence, tab })}
              onCite={(n) => setEvidence({ ...evidence, tab: "sources", cite: n })}
              onClose={() => setEvidence(null)}
            />
          )}
        </>
      )}

      {view === "tools" && (
        <main className="stage page-stage">
          <ToolsView
            survey={survey}
            onSurvey={(topic) => survey.start({ topic })}
            onSurveyCancel={survey.cancel}
            onSurveyReset={survey.reset}
            watch={watch}
            onWatch={setWatch}
            watchRun={watchRun}
            onWatchRun={() => watchRun.start({})}
            onWatchCancel={watchRun.cancel}
          />
        </main>
      )}
      {view === "metrics" && (
        <main className="stage page-stage">
          <MetricsView />
        </main>
      )}
      {view === "label" && (
        <main className="stage page-stage">
          <LabelView />
        </main>
      )}

      {settingsOpen && (
        <SettingsDialog
          onClose={() => setSettingsOpen(false)}
          config={config}
          status={status}
          offline={offline}
          options={options}
          onOptions={changeOptions}
          onClear={() => {
            chat.clear();
            setEvidence(null);
          }}
        />
      )}
    </div>
  );
}
