// 관심 분야 알림: 주제 등록·삭제, 새 논문 확인(SSE) → 본문 위쪽에 최신 소식지(WatchDigestCard).
import { useState } from "react";
import { api } from "../../api/client";
import type { WatchState } from "../../api/types";
import type { JobState } from "../../hooks/useJob";
import { stripSmallTags } from "../../lib/markdown";
import { Collapsible } from "../common/Collapsible";
import { Elapsed } from "../common/Elapsed";
import { JobLog } from "../common/JobLog";
import { Markdown } from "../common/Markdown";

interface Props {
  state: WatchState | null;
  onState: (s: WatchState) => void;
  running: boolean;
  onRun: () => void;
}

export function WatchPanel({ state, onState, running, onRun }: Props) {
  const [topic, setTopic] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const add = async () => {
    if (!topic.trim()) return;
    setBusy(true);
    setError(null);
    try {
      onState(await api.addWatchTopic(topic.trim()));
      setTopic("");
    } catch (e) {
      setError(String((e as Error).message ?? e));
    } finally {
      setBusy(false);
    }
  };
  const remove = async (t: string) => {
    try {
      onState(await api.removeWatchTopic(t));
    } catch (e) {
      setError(String((e as Error).message ?? e));
    }
  };

  const topics = state?.topics ?? [];
  return (
    <div className="side-tool">
      <ul className="topics">
        {topics.map((t) => (
          <li key={t.topic}>
            <span>
              {t.topic}
              <span className="muted small"> — 마지막 확인 {t.last_checked ?? "아직 없음"}</span>
            </span>
            <button type="button" className="icon-btn" title="주제 삭제" onClick={() => remove(t.topic)}>
              ✕
            </button>
          </li>
        ))}
      </ul>
      <input
        value={topic}
        onChange={(e) => setTopic(e.target.value)}
        placeholder="주제 추가 (예: 검색 증강 생성)"
        onKeyDown={(e) => e.key === "Enter" && !e.nativeEvent.isComposing && add()}
      />
      <div className="row">
        <button type="button" className="btn" disabled={!topic.trim() || busy} onClick={add}>
          {busy ? "검색어 만드는 중…" : "추가"}
        </button>
        <button type="button" className="btn" disabled={!topics.length || running} onClick={onRun}>
          새 논문 확인
        </button>
      </div>
      {error && <p className="error small">{error}</p>}
      <p className="muted small">arXiv 최신 논문 중 관련 있는 것을 골라 초록을 한국어로 요약합니다(주제당 약 30초).</p>
    </div>
  );
}

export function WatchDigestCard({ state, job, onCancel }: { state: WatchState | null; job: JobState<WatchState>; onCancel: () => void }) {
  if (job.running || job.error) {
    return (
      <div className="card job-box">
        <div className="job-head">
          {job.running && <span className="spinner" />}
          <span>📬 관심 분야 새 논문 확인 {job.running ? `· ${job.stage}` : "실패"}</span>
          {job.running && job.startedAt && <Elapsed since={job.startedAt} />}
          {job.running && (
            <button type="button" className="btn small ghost" onClick={onCancel}>
              중지
            </button>
          )}
        </div>
        {job.error && <p className="error">{job.error}</p>}
        <JobLog lines={job.log} />
      </div>
    );
  }
  const d = state?.digest;
  if (!d) return null;
  return (
    <div className="card">
      <Collapsible title={`📬 관심 분야 소식 (${d.date})`} defaultOpen={!!job.result}>
        <Markdown text={stripSmallTags(d.markdown)} />
      </Collapsible>
    </div>
  );
}
