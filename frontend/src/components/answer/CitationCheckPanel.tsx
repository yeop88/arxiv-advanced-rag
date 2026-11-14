// 4b 문장 단위 인용 확인 결과: 배지(맞는 문장 수) + 확인이 필요한 문장 목록(누르면 해당 근거를 펼침).
import type { ChatResult, SentenceCheck, SentenceStatus } from "../../api/types";
import type { CiteHandler } from "../common/Markdown";
import { Collapsible } from "../common/Collapsible";

const LABEL: Record<Exclude<SentenceStatus, "supported">, string> = {
  weak: "근거가 약함 — 근거를 넘어선 종합·추론일 수 있어요",
  miscited: "다른 근거가 더 맞음",
  uncited: "인용 번호 없음",
  bad_ref: "없는 근거 번호",
};

function Item({ s, onCite }: { s: SentenceCheck; onCite?: CiteHandler }) {
  const status = s.status as Exclude<SentenceStatus, "supported">;
  return (
    <li className={`cc-item cc-${status}`}>
      <span className="cc-text">{s.text}</span>
      <span className="cc-why">
        {LABEL[status]}
        {s.cites.length > 0 &&
          s.cites.map((n) => (
            <button key={n} type="button" className="cite" title="인용한 근거 보기" onClick={() => onCite?.(n)}>
              {n}
            </button>
          ))}
        {status === "miscited" && s.best != null && (
          <>
            {" → "}
            <button type="button" className="cite suggest" title="더 맞는 근거 보기" onClick={() => onCite?.(s.best!)}>
              {s.best}
            </button>
          </>
        )}
        {s.support != null && <span className="muted"> · 점수 {s.support.toFixed(2)}</span>}
      </span>
    </li>
  );
}

export function CitationCheckPanel({ result, onCite }: { result: ChatResult; onCite?: CiteHandler }) {
  const cc = result.citation_check;
  if (!cc || cc.summary.claims === 0) return null;
  const { claims, supported } = cc.summary;
  const all = supported === claims;
  return (
    <div className="cite-check">
      <span
        className={`pill ${all ? "ok" : "warn"}`}
        title={`문장마다 인용한 근거와 맞는지 리랭커 점수(기준 ${cc.threshold})로 확인했습니다. LLM 채점과 별개입니다.`}
      >
        🔎 인용 확인 {supported}/{claims}문장
      </span>
      {cc.flagged.length > 0 && (
        <Collapsible title={`확인이 필요한 문장 ${cc.flagged.length}개`} className="cc-list">
          <ul>
            {cc.flagged.map((s, i) => (
              <Item key={i} s={s} onCite={onCite} />
            ))}
          </ul>
        </Collapsible>
      )}
    </div>
  );
}
