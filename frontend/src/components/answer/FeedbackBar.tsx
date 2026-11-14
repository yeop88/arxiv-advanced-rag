// 답변 평가 👍/👎. 👎 는 이유(선택)를 함께 받아 서버 질문 기록에 붙인다 → 평가셋 후보(scripts/export_feedback.py).
import { useState } from "react";
import type { Rating } from "../../api/types";

interface Props {
  value?: { rating: Rating; note?: string };
  onRate: (rating: Rating, note?: string) => Promise<void>;
}

export function FeedbackBar({ value, onRate }: Props) {
  const [asking, setAsking] = useState(false);
  const [note, setNote] = useState("");
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  const send = async (rating: Rating, n?: string) => {
    setBusy(true);
    setErr(null);
    try {
      await onRate(rating, n);
      setAsking(false);
    } catch (e) {
      setErr((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  const r = value?.rating ?? 0;
  return (
    <div className="feedback">
      <span className="muted small">이 답변이 도움이 됐나요?</span>
      <button
        type="button"
        className={`fb${r === 1 ? " on" : ""}`}
        disabled={busy}
        aria-pressed={r === 1}
        title="도움이 됐어요"
        onClick={() => send(r === 1 ? 0 : 1)}
      >
        👍
      </button>
      <button
        type="button"
        className={`fb${r === -1 ? " on" : ""}`}
        disabled={busy}
        aria-pressed={r === -1}
        title="부족해요 — 평가 질문 후보로 모읍니다"
        onClick={() => (r === -1 ? send(0) : setAsking((a) => !a))}
      >
        👎
      </button>
      {r === -1 && value?.note && <span className="muted small">“{value.note}”</span>}
      {asking && (
        <form
          className="fb-note"
          onSubmit={(e) => {
            e.preventDefault();
            void send(-1, note.trim());
          }}
        >
          <input
            autoFocus
            value={note}
            maxLength={1000}
            placeholder="무엇이 부족했나요? (선택 — 예: 근거와 다름, 숫자 틀림)"
            onChange={(e) => setNote(e.target.value)}
          />
          <button type="submit" className="btn" disabled={busy}>
            보내기
          </button>
        </form>
      )}
      {err && <span className="error small">{err}</span>}
    </div>
  );
}
