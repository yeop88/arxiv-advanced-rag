// 왼쪽 아이콘 레일: 화면(대화·지표·판정·도구) 이동, 설정 열기, 밝은/어두운 테마, 서버 상태 점.
import type { ServerStatus } from "../../api/types";

export type View = "chat" | "metrics" | "label" | "tools";

const NAV: { v: View; icon: string; label: string }[] = [
  { v: "chat", icon: "💬", label: "대화" },
  { v: "tools", icon: "🧰", label: "도구" },
  { v: "metrics", icon: "📈", label: "지표" },
  { v: "label", icon: "⚖️", label: "판정" },
];

interface Props {
  view: View;
  onView: (v: View) => void;
  onSettings: () => void;
  theme: "light" | "dark";
  onTheme: () => void;
  status: ServerStatus | null;
  offline: boolean;
}

function statusOf(status: ServerStatus | null, offline: boolean): { cls: string; text: string } {
  if (offline) return { cls: "err", text: "서버 연결 끊김" };
  if (!status) return { cls: "wait", text: "상태 확인 중" };
  if (status.loader.state === "error") return { cls: "err", text: "모델·DB 준비 실패" };
  if (status.loader.state !== "ready") return { cls: "wait", text: `준비 중 · ${status.loader.step}` };
  if (status.llm.state === "down") return { cls: "err", text: "LLM 서버 꺼짐" };
  if (status.llm.state === "loading") return { cls: "wait", text: "LLM 준비 중" };
  if (status.enrich?.current) return { cls: "busy", text: "표·그림 처리 중" };
  return { cls: "ok", text: "준비됨" };
}

export function Rail({ view, onView, onSettings, theme, onTheme, status, offline }: Props) {
  const st = statusOf(status, offline);
  return (
    <nav className="rail" aria-label="화면">
      <div className="brand" title="arXiv RAG">
        <span className="brand-mark">◆</span>
      </div>
      <div className="rail-nav">
        {NAV.map((n) => (
          <button
            key={n.v}
            type="button"
            className={`rail-btn${view === n.v ? " on" : ""}`}
            onClick={() => onView(n.v)}
            aria-current={view === n.v ? "page" : undefined}
            title={n.label}
          >
            <span className="rail-icon" aria-hidden>
              {n.icon}
            </span>
            <span className="rail-label">{n.label}</span>
          </button>
        ))}
      </div>
      <div className="rail-foot">
        <button type="button" className="rail-btn" onClick={onTheme} title={theme === "dark" ? "밝은 테마" : "어두운 테마"}>
          <span className="rail-icon" aria-hidden>
            {theme === "dark" ? "☀️" : "🌙"}
          </span>
          <span className="rail-label">테마</span>
        </button>
        <button type="button" className="rail-btn" onClick={onSettings} title="설정">
          <span className="rail-icon" aria-hidden>
            ⚙️
          </span>
          <span className="rail-label">설정</span>
        </button>
        <button type="button" className={`status-dot ${st.cls}`} onClick={onSettings} title={st.text} aria-label={`서버 상태: ${st.text}`} />
      </div>
    </nav>
  );
}
