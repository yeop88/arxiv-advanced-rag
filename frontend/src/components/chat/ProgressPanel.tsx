// 답변 위의 진행 상황 줄: 현재 단계 + 경과 시간. 펼치면 단계별 로그.
// 답변을 쓰기 시작하면 자동으로 접혀 답변을 밀어내지 않는다.
import { useState } from "react";
import type { ChatMessage } from "../../hooks/useChat";
import { isActive } from "../../hooks/useChat";
import { timingSummary } from "../../lib/progress";
import { Elapsed } from "../common/Elapsed";
import { JobLog } from "../common/JobLog";

export function ProgressPanel({ m }: { m: ChatMessage }) {
  const active = isActive(m);
  const [userOpen, setUserOpen] = useState<boolean | null>(null);
  // 기본: 답변 쓰기 전에는 펼치고, 생각하거나 쓰기 시작하면 접는다. 사용자가 누르면 그 선택을 따른다.
  const open = userOpen ?? (m.phase === "working" && !m.content && !m.reasoning);

  let label: string;
  if (active) label = m.stage ?? "처리 중";
  else if (m.phase === "done") {
    const t = m.result ? timingSummary(m.result.timings) : "";
    label = `✅ 완료${t ? ` (${t})` : ""}`;
  } else if (m.phase === "stopped") label = "⏹ 중지됨";
  else label = "⚠️ 오류로 중단";

  return (
    <div className={`progress ${m.phase}`}>
      <button type="button" className="progress-head" onClick={() => setUserOpen(!open)} aria-expanded={open}>
        {active ? <span className="spinner" /> : null}
        <span className="progress-label">{label}</span>
        {m.startedAt && (
          <span className="muted small">
            · <Elapsed since={m.startedAt} until={active ? undefined : m.finishedAt} />
          </span>
        )}
        <span className="chev small" aria-hidden>
          {open ? "▾" : "▸"}
        </span>
      </button>
      {open && <JobLog lines={m.log ?? []} />}
    </div>
  );
}
