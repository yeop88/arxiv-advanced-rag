// 대화 목록(서버 저장): 새 대화, 지난 대화 열기·지우기. 서버 저장이 꺼져 있으면 지금 대화만 브라우저에 기억한다.
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

const when = (t: number) => {
  const d = new Date(t * 1000);
  const today = new Date();
  return d.toDateString() === today.toDateString()
    ? d.toLocaleTimeString("ko-KR", { hour: "2-digit", minute: "2-digit" })
    : d.toLocaleDateString("ko-KR", { month: "numeric", day: "numeric" });
};

export function ConversationPanel({ items, current, enabled, running, onNew, onOpen, onDelete }: Props) {
  return (
    <div className="conv">
      <button type="button" className="btn" onClick={onNew} disabled={running}>
        ＋ 새 대화
      </button>
      {!enabled && <p className="muted small">서버 대화 저장이 꺼져 있어 지금 대화만 이 브라우저에 기억합니다.</p>}
      {enabled && items.length === 0 && <p className="muted small">저장된 대화가 없습니다.</p>}
      <ul className="conv-list">
        {items.map((c) => (
          <li key={c.id} className={c.id === current ? "on" : ""}>
            <button
              type="button"
              className="conv-open"
              disabled={running || c.id === current}
              onClick={() => onOpen(c.id)}
              title={c.title}
            >
              <span className="conv-title">{c.title}</span>
              <span className="muted small">
                {when(c.updated)} · {Math.ceil(c.n / 2)}문답
              </span>
            </button>
            <button
              type="button"
              className="icon-btn conv-del"
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
  );
}
