// 문헌 조사 초안: 사이드바에서 주제 입력 → 본문 위쪽 카드(SurveyCard)에 진행·결과.
import { useState } from "react";
import type { SurveyResult } from "../../api/types";
import type { JobState } from "../../hooks/useJob";
import { Collapsible } from "../common/Collapsible";
import { Elapsed } from "../common/Elapsed";
import { JobLog } from "../common/JobLog";
import { Markdown } from "../common/Markdown";

export function SurveyPanel({ running, onStart }: { running: boolean; onStart: (topic: string) => void }) {
  const [topic, setTopic] = useState("");
  return (
    <div className="side-tool">
      <input
        value={topic}
        onChange={(e) => setTopic(e.target.value)}
        placeholder="예: 검색 증강 생성(RAG)의 발전 흐름"
        onKeyDown={(e) => e.key === "Enter" && !e.nativeEvent.isComposing && topic.trim() && !running && onStart(topic.trim())}
      />
      <button type="button" className="btn" disabled={!topic.trim() || running} onClick={() => onStart(topic.trim())}>
        초안 만들기
      </button>
      <p className="muted small">DB 의 관련 논문 5~8편을 소주제별 흐름으로 묶어 인용이 달린 초안을 씁니다(약 1~2분).</p>
    </div>
  );
}

function download(name: string, text: string) {
  const url = URL.createObjectURL(new Blob([text], { type: "text/markdown;charset=utf-8" }));
  const a = Object.assign(document.createElement("a"), { href: url, download: name });
  a.click();
  URL.revokeObjectURL(url);
}

export function SurveyCard({ job, onCancel, onClose }: { job: JobState<SurveyResult>; onCancel: () => void; onClose: () => void }) {
  if (!job.running && !job.result && !job.error) return null;
  const sv = job.result;
  if (!sv) {
    return (
      <div className="card job-box">
        <div className="job-head">
          {job.running && <span className="spinner" />}
          <span>📝 문헌 조사 초안 {job.running ? `· ${job.stage}` : "실패"}</span>
          {job.running && job.startedAt && <Elapsed since={job.startedAt} />}
          <button type="button" className="btn small ghost" onClick={job.running ? onCancel : onClose}>
            {job.running ? "중지" : "닫기"}
          </button>
        </div>
        {job.error && <p className="error">{job.error}</p>}
        <JobLog lines={job.log} />
      </div>
    );
  }
  const v = sv.verification;
  return (
    <div className="card">
      <Collapsible title={`📝 문헌 조사 초안 — ${sv.topic}`} defaultOpen>
        <Markdown text={sv.text} />
        <h4>참고문헌</h4>
        <ul className="plain small">
          {sv.references.map((r, i) => (
            <li key={i}>{r}</li>
          ))}
        </ul>
        <p className="muted small">
          {v ? (v.grounded ? "✅ 검증 통과" : `⚠️ 근거와 다를 수 있는 문장: ${v.unsupported.join("; ")}`) : "검증 생략"}
        </p>
        <div className="row">
          <button type="button" className="btn small" onClick={() => download("survey_draft.md", sv.markdown)}>
            마크다운으로 저장
          </button>
          <button type="button" className="btn small ghost" onClick={onClose}>
            닫기
          </button>
        </div>
      </Collapsible>
    </div>
  );
}
