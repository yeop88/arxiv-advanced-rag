// 대화 목록(서버 저장): 새 대화, 검색, 날짜별 묶음(오늘·지난 7일·이전), 열기·지우기.
import { useMemo, useState } from "react";
import type { ConversationInfo } from "../../api/types";

interface Props {
  items: ConversationInfo[];
  current: string;
  enabled: boolean;
  running: boolean;
  onNew: () => void;
  onOpen: (id: string) => void;
  onDelete: (id: string) => void;
}

function group(t: number): string {
  const d = new Date(t * 1000);
  const now = new Date();
  if (d.toDateString() === now.toDateString()) return "오늘";
  const days = (now.getTime() - d.getTime()) / 86400000;
  return days < 7 ? "지난 7일" : "이전";
}

export function ConversationList({ items, current, enabled, running, onNew, onOpen, onDelete }: Props) {
  const [q, setQ] = useState("");
  const groups = useMemo(() => {
    const out = new Map<string, ConversationInfo[]>();
    for (const c of items) {
      if (q && !c.title.toLowerCase().includes(q.toLowerCase())) continue;
      const g = group(c.updated);
      out.set(g, [...(out.get(g) ?? []), c]);
    }
    return [...out.entries()];
  }, [items, q]);

  return (
    <aside className="convs" aria-label="대화 목록">
      <div className="convs-head">
        <button type="button" className="btn primary new-chat" onClick={onNew} disabled={running}>
          <span aria-hidden>＋</span> 새 대화
        </button>
        {items.length > 4 && (
          <input className="convs-search" value={q} onChange={(e) => setQ(e.target.value)} placeholder="대화 찾기" aria-label="대화 찾기" />
        )}
      </div>
      <div className="convs-body">
        {!enabled && <p className="muted small pad">서버 대화 저장이 꺼져 있어 지금 대화만 이 브라우저에 기억합니다.</p>}
        {enabled && items.length === 0 && <p className="muted small pad">아직 대화가 없습니다. 첫 질문을 해 보세요.</p>}
        {groups.map(([g, cs]) => (
          <div key={g} className="conv-group">
            <div className="conv-group-title">{g}</div>
            <ul>
              {cs.map((c) => (
                <li key={c.id} className={c.id === current ? "on" : ""}>
                  <button
                    type="button"
                    className="conv-open"
                    disabled={running && c.id !== current}
                    onClick={() => onOpen(c.id)}
                    title={c.title}
                  >
                    {c.title}
                  </button>
                  <button
                    type="button"
                    className="conv-del"
                    aria-label={`${c.title} 지우기`}
                    disabled={running}
                    onClick={() => window.confirm(`"${c.title}" 대화를 지울까요?`) && onDelete(c.id)}
                  >
                    ✕
                  </button>
                </li>
              ))}
            </ul>
          </div>
        ))}
      </div>
    </aside>
  );
}
