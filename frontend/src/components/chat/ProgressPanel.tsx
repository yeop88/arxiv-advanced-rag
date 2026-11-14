// 답변 위 진행 표시: 4단계(질의 변환·검색·작성·검증) 점 + 지금 하는 일 + 경과 시간. 누르면 단계별 로그.
// 답변을 쓰기 시작하면 로그는 접혀 답변을 밀어내지 않는다.
import { useState } from "react";
import type { ChatMessage } from "../../hooks/useChat";
import { isActive } from "../../hooks/useChat";
import { timingSummary } from "../../lib/progress";
import { Elapsed } from "../common/Elapsed";
import { JobLog } from "../common/JobLog";

const STEPS = ["질의 변환", "검색", "작성", "검증"];

function stepOf(m: ChatMessage): number {
  if (m.phase === "done") return STEPS.length;
  if (m.phase === "checking") return 3;
  if (m.phase === "writing") return 2;
  const n = /^([1-4])\/4/.exec(m.stage ?? "");
  return n ? Number(n[1]) - 1 : 0;
}

export function ProgressPanel({ m }: { m: ChatMessage }) {
  const active = isActive(m);
  const [userOpen, setUserOpen] = useState<boolean | null>(null);
  const open = userOpen ?? false;
  const step = stepOf(m);

  let label: string;
  if (active) label = m.stage ?? "처리 중";
  else if (m.phase === "done") {
    const t = m.result ? timingSummary(m.result.timings) : "";
    label = `완료${t ? ` · ${t}` : ""}`;
  } else if (m.phase === "stopped") label = "중지됨";
  else label = "오류로 중단";

  return (
    <div className={`progress ${m.phase}`}>
      <button type="button" className="progress-head" onClick={() => setUserOpen(!open)} aria-expanded={open} title="단계별 진행 보기">
        <span className="steps" aria-hidden>
          {STEPS.map((s, i) => (
            <span key={s} className={`step ${i < step ? "done" : i === step && active ? "now" : ""}`} title={s} />
          ))}
        </span>
        {active && <span className="spinner" />}
        <span className="progress-label">{label}</span>
        {m.startedAt && (
          <span className="elapsed-chip">
            <Elapsed since={m.startedAt} until={active ? undefined : m.finishedAt} />
          </span>
        )}
        <span className="chev" aria-hidden>
          {open ? "▾" : "▸"}
        </span>
      </button>
      {open && <JobLog lines={m.log ?? []} />}
    </div>
  );
}
