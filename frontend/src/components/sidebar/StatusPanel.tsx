// 서버 상태: 모델·DB 로딩 단계와 LLM 서버 준비 여부.
import type { ServerStatus } from "../../api/types";

export function StatusPanel({ status, offline }: { status: ServerStatus | null; offline: boolean }) {
  if (offline) return <div className="status err">🔌 서버에 연결할 수 없습니다 (python -m server 실행 확인)</div>;
  if (!status) return <div className="status">상태 확인 중…</div>;
  const { loader, llm } = status;
  const llmClass = llm.state === "ok" ? "ok" : llm.state === "loading" ? "wait" : llm.state === "down" ? "err" : "";
  return (
    <div className="status-list">
      {loader.state === "ready" && <div className="status ok">✅ 모델·DB 준비 완료 ({loader.seconds.toFixed(0)}초)</div>}
      {loader.state === "error" && <div className="status err">⚠️ 모델·DB 준비 실패: {loader.error}</div>}
      {loader.state === "loading" && (
        <div className="status wait">
          ⏳ 모델·DB 로딩 중 ({loader.done.length}/4) · {loader.step} · {loader.seconds.toFixed(0)}초
        </div>
      )}
      <div className={`status ${llmClass}`}>
        {llm.state === "ok" ? "✅" : llm.state === "loading" ? "⏳" : llm.state === "down" ? "⚠️" : "ℹ️"} LLM: {llm.desc}
      </div>
      {status.enrich?.current && (
        <div className="status wait" title="질문이 없을 때만 한 페이지씩 처리합니다. 질문하면 길어야 한 페이지만 기다립니다.">
          📊 표·그림·수식 처리 중: {status.enrich.current.title.slice(0, 30) || status.enrich.current.source}
          {status.enrich.current.pages > 0 && ` (${status.enrich.current.page}/${status.enrich.current.pages}쪽)`}
          {status.enrich.queue > 0 && ` · 대기 ${status.enrich.queue}편`}
        </div>
      )}
    </div>
  );
}
