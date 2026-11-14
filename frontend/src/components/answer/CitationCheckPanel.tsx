// 4b 문장 단위 인용 확인 결과: 배지(맞는 문장 수·숫자 불일치) + 확인이 필요한 문장 목록(번호를 누르면 해당 근거를 펼침).
import type { ChatResult, SentenceCheck, SentenceStatus } from "../../api/types";
import type { CiteHandler } from "../common/Markdown";
import { Collapsible } from "../common/Collapsible";

type OkStatus = "supported" | "table_row" | "inherited" | "summary"; // 점수 경보가 아닌 상태
const OK: SentenceStatus[] = ["supported", "table_row", "inherited", "summary"];
const LABEL: Record<Exclude<SentenceStatus, OkStatus>, string> = {
  weak: "근거가 약함 — 근거를 넘어선 종합·추론일 수 있어요",
  miscited: "다른 근거가 더 맞음",
  uncited: "인용 번호 없음",
  bad_ref: "없는 근거 번호",
  contradicted: "근거와 반대되거나 다른 말일 수 있어요(함의 판정)",
};

function CiteButton({ n, onCite, suggest }: { n: number; onCite?: CiteHandler; suggest?: boolean }) {
  return (
    <button
      type="button"
      className={`cite${suggest ? " suggest" : ""}`}
      title={suggest ? "더 맞는 근거 보기" : "인용한 근거 보기"}
      onClick={() => onCite?.(n)}
    >
      {n}
    </button>
  );
}

function Item({ s, onCite }: { s: SentenceCheck; onCite?: CiteHandler }) {
  const scoreIssue = !OK.includes(s.status); // 표 행은 점수 없이 숫자만, 이어받은·종합 문장은 숫자만
  const numIssue = s.numbers_missing.length > 0;
  return (
    <li className={`cc-item cc-${s.status}${numIssue ? " cc-number" : ""}`}>
      <span className="cc-text">
        {s.status === "table_row" && <span className="muted">표 행 · </span>}
        {s.text}
      </span>
      {scoreIssue && (
        <span className="cc-why">
          {LABEL[s.status as Exclude<SentenceStatus, OkStatus>]}
          {s.cites.map((n) => (
            <CiteButton key={n} n={n} onCite={onCite} />
          ))}
          {s.status === "miscited" && s.best != null && (
            <>
              {" → "}
              <CiteButton n={s.best} onCite={onCite} suggest />
            </>
          )}
          {s.status === "contradicted" && s.contradiction != null ? (
            <span className="muted"> · 모순 확률 {s.contradiction.toFixed(2)}</span>
          ) : (
            s.support != null && <span className="muted"> · 점수 {s.support.toFixed(2)}</span>
          )}
        </span>
      )}
      {numIssue && (
        <span className="cc-why cc-num">
          🔢 숫자 {s.numbers_missing.join(", ")} 이(가) {s.cites.length ? "인용한 근거" : "근거"}에 없음
          {!scoreIssue && s.cites.map((n) => <CiteButton key={n} n={n} onCite={onCite} />)}
          {s.numbers_found_in.length > 0 && (
            <>
              {" → 있는 근거"}
              {s.numbers_found_in.map((n) => (
                <CiteButton key={n} n={n} onCite={onCite} suggest />
              ))}
            </>
          )}
        </span>
      )}
    </li>
  );
}

export function CitationCheckPanel({ result, onCite, defaultOpen = false }: { result: ChatResult; onCite?: CiteHandler; defaultOpen?: boolean }) {
  const cc = result.citation_check;
  if (!cc || cc.summary.claims === 0) return null;
  const { claims, supported, numeric, number_mismatch } = cc.summary;
  return (
    <div className="cite-check">
      <div className="verify-row">
        <span
          className={`pill ${supported === claims ? "ok" : "warn"}`}
          title={`문장마다 인용한 근거와 맞는지 리랭커 점수(기준 ${cc.threshold})와 함의 판정으로 확인했습니다. LLM 채점과 별개입니다.`}
        >
          🔎 인용 확인 {supported}/{claims}문장
        </span>
        {numeric > 0 && (
          <span
            className={`pill ${number_mismatch ? "warn" : "ok"}`}
            title="답변의 숫자(소수·두 자리 이상)가 인용한 근거 원문에 그대로 있는지 확인했습니다."
          >
            🔢 숫자 {number_mismatch ? `불일치 ${number_mismatch}문장` : `일치 ${numeric}문장`}
          </span>
        )}
      </div>
      {cc.flagged.length === 0 && <p className="muted small">모든 주장 문장이 인용한 근거와 맞고, 숫자도 근거 원문에 있습니다.</p>}
      {cc.flagged.length > 0 && (
        <Collapsible title={`확인이 필요한 문장 ${cc.flagged.length}개`} className="cc-list" defaultOpen={defaultOpen}>
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
