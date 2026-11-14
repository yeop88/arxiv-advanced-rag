// 대화 메시지 하나. 답변은 진행 줄 → 본문(타자 효과) → 검증 배지 → 근거·인용·비교표·내부 동작 순.
// memo: 쓰는 중인 답변 하나만 다시 그리고 지난 메시지는 그대로 둔다.
import { memo, useCallback, useMemo, useState } from "react";
import type { ChatMessage, RateHandler } from "../../hooks/useChat";
import { isActive } from "../../hooks/useChat";
import { CitationsPanel } from "../answer/CitationsPanel";
import { CompareTablePanel } from "../answer/CompareTablePanel";
import { FeedbackBar } from "../answer/FeedbackBar";
import { AutoFetchedNotice, Inspector, VerificationBar } from "../answer/Inspector";
import { SourcesPanel } from "../answer/SourcesPanel";
import { StreamingAnswer } from "../answer/StreamingAnswer";
import { ThinkingBox } from "../answer/ThinkingBox";
import { Collapsible } from "../common/Collapsible";
import { Markdown } from "../common/Markdown";
import { ProgressPanel } from "./ProgressPanel";

function AssistantMessage({ m, onRate }: { m: ChatMessage; onRate?: RateHandler }) {
  const [sourcesOpen, setSourcesOpen] = useState(false);
  const [highlight, setHighlight] = useState<number | null>(null);
  const r = m.result;
  const active = isActive(m);
  // 글자가 더 올 수 있는 동안(작성 중)만 커서 — 검증 중에는 답변이 완성돼 있다
  const streaming = m.phase === "working" || m.phase === "writing";

  const citeTitles = useMemo(() => (r ? r.sources.map((s) => s.title || s.source) : m.citeTitles), [r, m.citeTitles]);
  const papers = useMemo(() => (r ? [...new Set(r.sources.map((s) => s.source).filter(Boolean))] : []), [r]);
  const onCite = useCallback((n: number) => {
    setSourcesOpen(true);
    setHighlight(n);
  }, []);

  return (
    <div className="msg assistant">
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
      {m.mode && !active && <p className="muted small">{m.mode}</p>}
      {r && (
        <div className="extras">
          <AutoFetchedNotice result={r} />
          <VerificationBar result={r} />
          <SourcesPanel
            sources={r.sources}
            open={sourcesOpen}
            onToggle={(o) => {
              setSourcesOpen(o);
              if (!o) setHighlight(null);
            }}
            highlight={highlight}
          />
          <CitationsPanel sources={papers} />
          <CompareTablePanel sources={papers} />
          <Inspector result={r} />
          {r.log_id && onRate && <FeedbackBar value={m.feedback} onRate={(rating, note) => onRate(m.id, rating, note)} />}
        </div>
      )}
    </div>
  );
}

export const MessageItem = memo(function MessageItem({ m, onRate }: { m: ChatMessage; onRate?: RateHandler }) {
  if (m.role === "user") {
    return (
      <div className="msg user">
        <div className="bubble">{m.content}</div>
      </div>
    );
  }
  return <AssistantMessage m={m} onRate={onRate} />;
});
