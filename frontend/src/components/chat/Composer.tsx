// 질문 입력창. Enter 로 보내고 Shift+Enter 로 줄바꿈. 답하는 중에는 '중지' 버튼.
import { useEffect, useRef, useState, type KeyboardEvent } from "react";

interface Props {
  running: boolean;
  onSend: (q: string) => void;
  onStop: () => void;
}

export function Composer({ running, onSend, onStop }: Props) {
  const [text, setText] = useState("");
  const ref = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 200)}px`;
  }, [text]);

  const submit = () => {
    if (!text.trim() || running) return;
    onSend(text);
    setText("");
  };
  const onKey = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
      e.preventDefault();
      submit();
    }
  };

  return (
    <div className="composer">
      <textarea
        ref={ref}
        rows={1}
        value={text}
        placeholder="논문에 대해 물어보세요 (예: 리랭킹은 왜 필요한가요?)"
        onChange={(e) => setText(e.target.value)}
        onKeyDown={onKey}
        aria-label="질문"
      />
      {running ? (
        <button type="button" className="btn stop" onClick={onStop} title="답변 중지">
          ■ 중지
        </button>
      ) : (
        <button type="button" className="btn primary" onClick={submit} disabled={!text.trim()}>
          보내기
        </button>
      )}
    </div>
  );
}
