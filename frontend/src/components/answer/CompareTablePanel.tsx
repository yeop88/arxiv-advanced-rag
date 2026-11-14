// 근거 논문이 2편 이상이면 '비교표 만들기'(방법·데이터·결과·한계). LLM 작업이라 진행 상황을 보여 준다.
import type { CompareTableResult } from "../../api/types";
import { useJob } from "../../hooks/useJob";
import { Collapsible } from "../common/Collapsible";
import { Elapsed } from "../common/Elapsed";
import { Markdown } from "../common/Markdown";
import { JobLog } from "../common/JobLog";

const MAX_PAPERS = 5;

export function CompareTablePanel({ sources }: { sources: string[] }) {
  const job = useJob<CompareTableResult>("/api/compare-table");
  if (sources.length < 2) return null;
  const t = job.result;

  if (!t && !job.running && !job.error) {
    return (
      <button type="button" className="btn ghost" onClick={() => job.start({ sources })}>
        📊 근거 논문 {Math.min(sources.length, MAX_PAPERS)}편 비교표 만들기
      </button>
    );
  }
  if (!t) {
    return (
      <div className="job-box">
        <div className="job-head">
          {job.running && <span className="spinner" />}
          <span>비교표 {job.running ? `만드는 중 · ${job.stage}` : "실패"}</span>
          {job.startedAt && job.running && <Elapsed since={job.startedAt} />}
          {job.running ? (
            <button type="button" className="btn small ghost" onClick={job.cancel}>
              중지
            </button>
          ) : (
            <button type="button" className="btn small ghost" onClick={() => job.start({ sources })}>
              다시 시도
            </button>
          )}
        </div>
        {job.error && <p className="error">{job.error}</p>}
        <JobLog lines={job.log} />
      </div>
    );
  }
  const v = t.verification;
  const badge = v ? (v.grounded ? "✅ 검증 통과" : "⚠️ 일부 칸이 근거와 다를 수 있음") : "검증 생략";
  return (
    <Collapsible title="📊 논문 비교표 (방법·데이터·결과·한계)" defaultOpen>
      <Markdown text={t.markdown} />
      <p className="muted small">
        {badge} · 채운 칸 {t.cells_filled}/{t.cells_total} · ⚠ 칸은 근거 원문을 확인하세요 · [n]은 아래 근거 번호
        {t.number_check && t.number_check.rows > 0 && (
          <>
            {" · "}
            {t.number_check.mismatch.length
              ? `🔢 숫자 불일치 ${t.number_check.mismatch.length}행`
              : `🔢 숫자 일치 ${t.number_check.rows}행`}
          </>
        )}
      </p>
      {t.number_check?.mismatch.map((r, i) => (
        <p key={i} className="warn-text small">
          🔢 "{r.text.slice(0, 80)}" — 숫자 {r.numbers_missing.join(", ")} 이(가) 단 근거에 없음
          {r.numbers_found_in.length > 0 && ` → [${r.numbers_found_in.join("], [")}]에 있음`}
        </p>
      ))}
      <ol className="plain small muted">
        {t.chunks.map((c, i) => (
          <li key={i}>
            [{i + 1}] {c.title.slice(0, 60)}
            {c.section ? ` · ${c.section}` : ""}
          </li>
        ))}
      </ol>
    </Collapsible>
  );
}
