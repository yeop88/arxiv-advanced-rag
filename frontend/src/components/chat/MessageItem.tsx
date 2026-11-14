// 대화 메시지 하나. 사용자: 오른쪽 말풍선. 답변: 아바타 + 진행 표시 → 본문(타자 효과) → 근거 확인 줄(근거·인용·숫자·검증) → 👍👎·복사.
// 근거·인용 확인·인용 관계·비교표·내부 동작은 오른쪽 근거 패널(EvidenceDrawer)에서 연다.
// memo: 쓰는 중인 답변 하나만 다시 그리고 지난 메시지는 그대로 둔다.
import { memo, useCallback, useMemo, useState } from "react";
import type { ChatMessage, RateHandler } from "../../hooks/useChat";
import { isActive } from "../../hooks/useChat";
import { FeedbackBar } from "../answer/FeedbackBar";
import { AutoFetchedNotice } from "../answer/Inspector";
import { StreamingAnswer } from "../answer/StreamingAnswer";
import { ThinkingBox } from "../answer/ThinkingBox";
import { Collapsible } from "../common/Collapsible";
import { Markdown } from "../common/Markdown";
import { ProgressPanel } from "./ProgressPanel";

export type EvidenceTab = "sources" | "check" | "citations" | "compare" | "inside";
export type OpenEvidence = (msgId: string, tab: EvidenceTab, cite?: number) => void;

interface Props {
  m: ChatMessage;
  onRate?: RateHandler;
  onEvidence?: OpenEvidence;
  selected?: boolean; // 근거 패널이 이 답변을 보고 있음
}

function CopyButton({ text }: { text: string }) {
  const [done, setDone] = useState(false);
  return (
    <button
      type="button"
      className="act icon"
      title="답변 복사"
      onClick={() => {
        void navigator.clipboard?.writeText(text).then(() => {
          setDone(true);
          window.setTimeout(() => setDone(false), 1200);
        });
      }}
    >
      {done ? "✓" : "⧉"}
    </button>
  );
}

function AssistantMessage({ m, onRate, onEvidence, selected }: Props) {
  const r = m.result;
  const active = isActive(m);
  // 글자가 더 올 수 있는 동안(작성 중)만 커서 — 검증 중에는 답변이 완성돼 있다
  const streaming = m.phase === "working" || m.phase === "writing";
  const citeTitles = useMemo(() => (r ? r.sources.map((s) => s.title || s.source) : m.citeTitles), [r, m.citeTitles]);
  const onCite = useCallback((n: number) => onEvidence?.(m.id, "sources", n), [onEvidence, m.id]);
  const cc = r?.citation_check;
  const v = r?.verification;

  return (
    <div className={`msg assistant${selected ? " selected" : ""}`}>
      <div className="avatar" aria-hidden>
        ◆
      </div>
      <div className="msg-body">
        <ProgressPanel m={m} />
        <ThinkingBox text={m.reasoning ?? ""} live={streaming && !m.content} />
        {m.drafts?.map((d, i) => (
          <Collapsible key={i} title={`고치기 전 초안 ${m.drafts!.length > 1 ? i + 1 : ""}`} className="draft">
            <Markdown text={d} />
          </Collapsible>
        ))}
        {(m.content || (streaming && !m.reasoning)) && (
          <StreamingAnswer text={m.content} streaming={streaming} citeTitles={citeTitles} onCite={r ? onCite : undefined} />
        )}
        {m.phase === "checking" && <p className="muted small checking">답변의 각 주장이 근거에 있는지 확인하는 중…</p>}
        {m.error && <p className="error">⚠️ {m.error}</p>}
        {r && (
          <div className="extras">
            <AutoFetchedNotice result={r} />
            <div className="actions">
              <button type="button" className="act" onClick={() => onEvidence?.(m.id, "sources")} title="근거 원문 보기">
                📑 근거 {r.sources.length}
              </button>
              {cc && cc.summary.claims > 0 && (
                <button
                  type="button"
                  className={`act ${cc.summary.supported === cc.summary.claims ? "ok" : "warn"}`}
                  onClick={() => onEvidence?.(m.id, "check")}
                  title="문장마다 인용한 근거와 맞는지(리랭커·숫자 대조, LLM 없음)"
                >
                  🔎 인용 {cc.summary.supported}/{cc.summary.claims}
                </button>
              )}
              {cc && cc.summary.numeric > 0 && (
                <button
                  type="button"
                  className={`act ${cc.summary.number_mismatch ? "warn" : "ok"}`}
                  onClick={() => onEvidence?.(m.id, "check")}
                  title="답변 숫자가 인용한 근거 원문에 있는지"
                >
                  🔢 {cc.summary.number_mismatch ? `숫자 ${cc.summary.number_mismatch}곳 확인` : "숫자 일치"}
                </button>
              )}
              {v && (
                <button
                  type="button"
                  className={`act ${v.grounded ? "" : "warn"}`}
                  onClick={() => onEvidence?.(m.id, "inside")}
                  title="4단계 LLM 검증(로컬 모델 자기 채점)"
                >
                  {v.grounded ? "✓" : "⚠"} 검증 {v.faithfulness.toFixed(2)}
                  {r.regenerated ? " · 고쳐 씀" : ""}
                </button>
              )}
              <span className="spacer" />
              {r.log_id && onRate && <FeedbackBar value={m.feedback} onRate={(rating, note) => onRate(m.id, rating, note)} />}
              <CopyButton text={m.content} />
            </div>
            {v && v.unsupported.length > 0 && <p className="warn-text small">근거 없는 주장: {v.unsupported.join("; ")}</p>}
          </div>
        )}
        {m.mode && !active && <p className="muted tiny">{m.mode}</p>}
      </div>
    </div>
  );
}

export const MessageItem = memo(function MessageItem({ m, onRate, onEvidence, selected }: Props) {
  if (m.role === "user") {
    return (
      <div className="msg user">
        <div className="bubble">{m.content}</div>
      </div>
    );
  }
  return <AssistantMessage m={m} onRate={onRate} onEvidence={onEvidence} selected={selected} />;
});
