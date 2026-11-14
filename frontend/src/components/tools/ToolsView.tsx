// 도구 화면: 문헌 조사 초안과 관심 분야 알림(새 논문 소식지). 대화와 따로 두어 대화 흐름을 가리지 않는다.
import type { SurveyResult, WatchState } from "../../api/types";
import type { JobState } from "../../hooks/useJob";
import { SurveyCard, SurveyPanel } from "../sidebar/SurveyPanel";
import { WatchDigestCard, WatchPanel } from "../sidebar/WatchPanel";

interface Props {
  survey: JobState<SurveyResult>;
  onSurvey: (topic: string) => void;
  onSurveyCancel: () => void;
  onSurveyReset: () => void;
  watch: WatchState | null;
  onWatch: (s: WatchState) => void;
  watchRun: JobState<WatchState>;
  onWatchRun: () => void;
  onWatchCancel: () => void;
}

export function ToolsView(p: Props) {
  return (
    <div className="page">
      <header className="page-head">
        <h1>도구</h1>
        <p className="muted">질문 하나로 끝나지 않는 일 — 주제의 문헌 흐름을 정리하거나, 새 논문을 꾸준히 받아 봅니다.</p>
      </header>
      <div className="tool-grid">
        <section className="tool-card">
          <div className="tool-head">
            <span className="tool-icon" aria-hidden>
              📝
            </span>
            <div>
              <h2>문헌 조사 초안</h2>
              <p className="muted small">관련 논문 5~8편을 소주제별 흐름으로 묶고 문장마다 인용을 답니다(약 1~2분).</p>
            </div>
          </div>
          <SurveyPanel running={p.survey.running} onStart={p.onSurvey} />
        </section>
        <section className="tool-card">
          <div className="tool-head">
            <span className="tool-icon" aria-hidden>
              📬
            </span>
            <div>
              <h2>관심 분야 알림</h2>
              <p className="muted small">등록한 주제의 arXiv 새 논문을 골라 한국어로 요약합니다(주제당 약 30초).</p>
            </div>
          </div>
          <WatchPanel state={p.watch} onState={p.onWatch} running={p.watchRun.running} onRun={p.onWatchRun} />
        </section>
      </div>
      <div className="tool-results">
        <SurveyCard job={p.survey} onCancel={p.onSurveyCancel} onClose={p.onSurveyReset} />
        <WatchDigestCard state={p.watch} job={p.watchRun} onCancel={p.onWatchCancel} />
      </div>
    </div>
  );
}
