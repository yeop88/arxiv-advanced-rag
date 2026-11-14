// 답변의 [n] 인용에 대응하는 근거 목록. 인용 칩을 누르면 펼쳐지고 그 근거가 강조된다.
import { useEffect, useRef } from "react";
import type { SourceChunk } from "../../api/types";
import { Collapsible } from "../common/Collapsible";

const KIND_LABEL: Record<string, string> = { table: "표", figure: "그림", equation: "수식" };

interface Props {
  sources: SourceChunk[];
  open: boolean;
  onToggle: (open: boolean) => void;
  highlight: number | null;
}

export function SourcesPanel({ sources, open, onToggle, highlight }: Props) {
  const refs = useRef<(HTMLLIElement | null)[]>([]);
  useEffect(() => {
    if (open && highlight) refs.current[highlight - 1]?.scrollIntoView({ block: "nearest", behavior: "smooth" });
  }, [open, highlight]);

  if (!sources.length) return null;
  return (
    <Collapsible title={`📑 근거 출처 ${sources.length}개 (인용 [n] 대응)`} open={open} onToggle={onToggle}>
      <ol className="sources">
        {sources.map((c, i) => (
          <li key={i} ref={(el) => (refs.current[i] = el)} className={highlight === i + 1 ? "hl" : ""}>
            <div className="src-head">
              <span className="src-num">[{i + 1}]</span>
              {c.link ? (
                <a href={c.link} target="_blank" rel="noreferrer">
                  {c.title || c.source}
                </a>
              ) : (
                <span>{c.title || "unknown"}</span>
              )}
              {KIND_LABEL[c.kind] && <span className="tag">{KIND_LABEL[c.kind]}</span>}
              {c.origin === "auto" && <span className="tag">자동 수집</span>}
            </div>
            <div className="muted small">
              arXiv:{c.source || "N/A"}
              {c.section ? ` · ${c.section}` : ""}
            </div>
            <p className="snippet">{c.snippet}</p>
          </li>
        ))}
      </ol>
    </Collapsible>
  );
}
