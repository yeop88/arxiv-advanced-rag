// Thinking 모드에서 답을 쓰기 전 모델의 추론을 실시간으로 보여 준다.
// 생각하는 동안: 고정 높이 상자에 마지막 몇 줄만 흘러가게(답변 자리를 밀어내지 않음).
// 답을 쓰기 시작하면: '생각 과정' 접힌 상자로 바뀐다.
import { Collapsible } from "../common/Collapsible";

const TAIL_CHARS = 600;

export function ThinkingBox({ text, live }: { text: string; live: boolean }) {
  if (!text) return null;
  const n = text.length.toLocaleString();
  if (live) {
    return (
      <div className="thinking live" aria-live="off">
        <div className="thinking-head">
          <span className="spinner" /> 🧠 생각하는 중 · {n}자 <span className="muted small">(다 생각하면 답을 씁니다)</span>
        </div>
        <div className="thinking-tail">{text.slice(-TAIL_CHARS)}</div>
      </div>
    );
  }
  return (
    <Collapsible title={`🧠 생각 과정 (${n}자)`} className="thinking">
      <div className="thinking-full">{text}</div>
    </Collapsible>
  );
}
