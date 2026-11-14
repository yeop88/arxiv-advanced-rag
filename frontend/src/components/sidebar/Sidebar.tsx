// 사이드바: 상태 · 설정 · 문헌 조사 · 관심 분야 · 모델 정보 · 대화 초기화.
import type { AppConfig, ChatOptions, ServerStatus, WatchState } from "../../api/types";
import { SettingsPanel } from "./SettingsPanel";
import { StatusPanel } from "./StatusPanel";
import { SurveyPanel } from "./SurveyPanel";
import { WatchPanel } from "./WatchPanel";

interface Props {
  open: boolean;
  onClose: () => void;
  config: AppConfig | null;
  status: ServerStatus | null;
  offline: boolean;
  options: ChatOptions;
  onOptions: (o: ChatOptions) => void;
  surveyRunning: boolean;
  onSurvey: (topic: string) => void;
  watch: WatchState | null;
  onWatch: (s: WatchState) => void;
  watchRunning: boolean;
  onWatchRun: () => void;
  onClear: () => void;
}

export function Sidebar(p: Props) {
  return (
    <aside className={`sidebar${p.open ? " open" : ""}`}>
      <div className="side-head">
        <h1>⚙️ 설정</h1>
        <button type="button" className="icon-btn only-mobile" onClick={p.onClose} aria-label="닫기">
          ✕
        </button>
      </div>
      <StatusPanel status={p.status} offline={p.offline} />
      <section>
        <SettingsPanel config={p.config} options={p.options} onChange={p.onOptions} />
      </section>
      <section>
        <h3>📝 문헌 조사 초안</h3>
        <SurveyPanel running={p.surveyRunning} onStart={p.onSurvey} />
      </section>
      <section>
        <h3>📬 관심 분야 알림</h3>
        <WatchPanel state={p.watch} onState={p.onWatch} running={p.watchRunning} onRun={p.onWatchRun} />
      </section>
      <section className="muted small">
        {p.config && (
          <>
            <div>
              LLM: <code>{p.config.provider}</code> / {p.config.model}
            </div>
            <div>
              임베딩: <code>{p.config.embed_model}</code> · 벡터DB: <code>{p.config.vector_backend}</code>
            </div>
          </>
        )}
      </section>
      <button type="button" className="btn ghost" onClick={p.onClear}>
        🗑️ 대화 초기화
      </button>
    </aside>
  );
}
