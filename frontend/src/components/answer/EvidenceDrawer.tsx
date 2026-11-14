// 오른쪽 근거 패널: 고른 답변의 근거 원문·문장 단위 인용 확인·인용 관계·비교표·단계별 내부 동작.
// 답변의 [n] 칩이나 근거 줄 버튼으로 연다. 탭을 바꿔도 내용은 그대로 둔다(비교표 작업이 끊기지 않게).
import { useEffect, useMemo, useRef } from "react";
import type { ChatResult } from "../../api/types";
import type { EvidenceTab } from "../chat/MessageItem";
import { CitationCheckPanel } from "./CitationCheckPanel";
import { CitationsPanel } from "./CitationsPanel";
import { CompareTablePanel } from "./CompareTablePanel";
import { Inspector, VerificationBar } from "./Inspector";

const KIND_LABEL: Record<string, string> = { table: "표", figure: "그림", equation: "수식" };
const TABS: { t: EvidenceTab; label: string }[] = [
  { t: "sources", label: "근거" },
  { t: "check", label: "인용 확인" },
  { t: "citations", label: "인용 관계" },
  { t: "compare", label: "비교표" },
  { t: "inside", label: "내부 동작" },
];

interface Props {
  result: ChatResult;
  question: string;
  tab: EvidenceTab;
  highlight: number | null;
  onTab: (t: EvidenceTab) => void;
  onCite: (n: number) => void;
  onClose: () => void;
}

function Sources({ result, highlight }: { result: ChatResult; highlight: number | null }) {
  const refs = useRef<(HTMLLIElement | null)[]>([]);
  useEffect(() => {
    if (highlight) refs.current[highlight - 1]?.scrollIntoView({ block: "center", behavior: "smooth" });
  }, [highlight]);
  return (
    <ol className="src-list">
      {result.sources.map((c, i) => (
        <li key={i} ref={(el) => (refs.current[i] = el)} className={highlight === i + 1 ? "hl" : ""}>
          <div className="src-top">
            <span className="src-num">{i + 1}</span>
            <div className="src-meta">
              {c.link ? (
                <a href={c.link} target="_blank" rel="noreferrer" className="src-title">
                  {c.title || c.source}
                </a>
              ) : (
                <span className="src-title">{c.title || "unknown"}</span>
              )}
              <div className="muted tiny">
                arXiv:{c.source || "N/A"}
                {c.section ? ` · ${c.section}` : ""}
                {c.rerank_score != null ? ` · 관련도 ${c.rerank_score.toFixed(2)}` : ""}
              </div>
            </div>
            <div className="src-tags">
              {KIND_LABEL[c.kind] && <span className="tag">{KIND_LABEL[c.kind]}</span>}
              {c.origin === "auto" && <span className="tag">자동 수집</span>}
            </div>
          </div>
          <p className="snippet">{c.snippet}</p>
        </li>
      ))}
    </ol>
  );
}

export function EvidenceDrawer({ result, question, tab, highlight, onTab, onCite, onClose }: Props) {
  const papers = useMemo(() => [...new Set(result.sources.map((s) => s.source).filter(Boolean))], [result]);
  return (
    <aside className="drawer" aria-label="근거 패널">
      <header className="drawer-head">
        <div className="drawer-title">
          <div className="muted tiny">답변의 근거</div>
          <div className="drawer-q" title={question}>
            {question}
          </div>
        </div>
        <button type="button" className="icon-btn" onClick={onClose} aria-label="근거 패널 닫기">
          ✕
        </button>
      </header>
      <nav className="drawer-tabs" role="tablist">
        {TABS.map((x) => (
          <button key={x.t} type="button" role="tab" aria-selected={tab === x.t} className={tab === x.t ? "on" : ""} onClick={() => onTab(x.t)}>
            {x.label}
            {x.t === "sources" && <span className="count">{result.sources.length}</span>}
          </button>
        ))}
      </nav>
      <div className="drawer-body">
        <div hidden={tab !== "sources"}>
          <Sources result={result} highlight={highlight} />
        </div>
        <div hidden={tab !== "check"}>
          {result.citation_check && result.citation_check.summary.claims > 0 ? (
            <CitationCheckPanel result={result} onCite={onCite} defaultOpen />
          ) : (
            <p className="muted small">확인할 주장 문장이 없습니다.</p>
          )}
        </div>
        <div hidden={tab !== "citations"}>
          <CitationsPanel sources={papers} defaultOpen />
        </div>
        <div hidden={tab !== "compare"}>
          {papers.length >= 2 ? <CompareTablePanel sources={papers} /> : <p className="muted small">근거 논문이 2편 이상일 때 만들 수 있습니다.</p>}
        </div>
        <div hidden={tab !== "inside"}>
          <VerificationBar result={result} />
          <Inspector result={result} defaultOpen />
        </div>
      </div>
    </aside>
  );
}
