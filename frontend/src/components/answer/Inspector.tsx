// 검증 결과 배지 + 자동 수집 안내 + 단계별 내부 동작(학습·디버깅용).
import type { ChatResult } from "../../api/types";
import { Collapsible } from "../common/Collapsible";

export function VerificationBar({ result }: { result: ChatResult }) {
  const v = result.verification;
  if (!v) return null;
  return (
    <div className="verify">
      <div className="verify-row">
        <span className={`pill ${v.grounded ? "ok" : "warn"}`}>{v.grounded ? "✅ 근거 충실" : "⚠️ 근거 확인 필요"}</span>
        <span className="pill">Faithfulness {v.faithfulness.toFixed(2)}</span>
        {result.regenerated && <span className="pill">🔁 검증 후 고쳐 씀</span>}
      </div>
      {v.unsupported.length > 0 && <p className="warn-text small">근거 없는 주장: {v.unsupported.join("; ")}</p>}
    </div>
  );
}

export function AutoFetchedNotice({ result }: { result: ChatResult }) {
  const added = result.auto_fetched;
  if (!added?.length) return null;
  return (
    <div className="notice">
      📥 이 질문에 맞는 근거가 부족해 arXiv 논문 {added.length}편을 DB 에 새로 추가했습니다:
      <ul>
        {added.map((a) => (
          <li key={a.source}>
            <a href={`https://arxiv.org/abs/${a.source}`} target="_blank" rel="noreferrer">
              {a.title}
            </a>{" "}
            — {a.fulltext ? "전문" : "초록"} {a.chunks}청크
          </li>
        ))}
      </ul>
    </div>
  );
}

export function Inspector({ result }: { result: ChatResult }) {
  return (
    <Collapsible title="🔍 단계별 내부 동작 보기">
      {result.standalone_question && result.standalone_question !== result.question && (
        <>
          <h4>0단계 · 멀티턴 문맥화</h4>
          <ul>
            <li>원 질문: {result.question}</li>
            <li>독립형: {result.standalone_question}</li>
          </ul>
        </>
      )}
      <h4>1단계 · 질의변환 ({result.transformed_queries.length}개)</h4>
      <ul>
        {result.transformed_queries.map((q, i) => (
          <li key={i}>{q}</li>
        ))}
      </ul>
      <h4>
        2단계 · 리랭킹 선택 ({result.sources.length}개 / 후보 {result.n_candidates}개)
      </h4>
      <ol>
        {result.sources.map((c, i) => (
          <li key={i}>
            <code>{c.title.slice(0, 70)}</code> (arXiv:{c.source}) · 리랭크={c.rerank_score != null ? c.rerank_score.toFixed(3) : "–"}
          </li>
        ))}
      </ol>
      {Object.keys(result.timings).length > 0 && (
        <p className="muted small">
          ⏱️{" "}
          {Object.entries(result.timings)
            .map(([k, v]) => `${k}=${v.toFixed(1)}s`)
            .join(" · ")}
        </p>
      )}
    </Collapsible>
  );
}
