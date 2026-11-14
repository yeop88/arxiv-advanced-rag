// 질문 입력 카드: Enter 로 보내고 Shift+Enter 로 줄바꿈. 아래 줄에 자주 바꾸는 설정 칩(Thinking·자동 수집)과 보내기/중지.
import { useEffect, useRef, useState, type KeyboardEvent } from "react";
import type { ChatOptions } from "../../api/types";

interface Props {
  running: boolean;
  onSend: (q: string) => void;
  onStop: () => void;
  options: ChatOptions;
  onOptions: (o: ChatOptions) => void;
  local: boolean; // 로컬 모델이면 Thinking 칩을 보인다
  autoFocus?: boolean;
}

export function Composer({ running, onSend, onStop, options, onOptions, local, autoFocus }: Props) {
  const [text, setText] = useState("");
  const ref = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 220)}px`;
  }, [text]);
  useEffect(() => {
    if (autoFocus) ref.current?.focus();
  }, [autoFocus]);

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
  const thinking = options.local_thinking === "on";

  return (
    <div className="composer">
      <textarea
        ref={ref}
        rows={1}
        value={text}
        placeholder="논문에 대해 물어보세요 — 예: DPR과 BM25의 검색 정확도를 비교해 주세요"
        onChange={(e) => setText(e.target.value)}
        onKeyDown={onKey}
        aria-label="질문"
      />
      <div className="composer-bar">
        <div className="chips">
          {local && (
            <button
              type="button"
              className={`chip${thinking ? " on" : ""}`}
              aria-pressed={thinking}
              onClick={() => onOptions({ ...options, local_thinking: thinking ? "off" : "on" })}
              title="켜면 답하기 전에 추론해 비교·종합에 유리하지만 느려집니다(질문당 약 1분 → 3~4분)"
            >
              🧠 깊게 생각
            </button>
          )}
          <button
            type="button"
            className={`chip${options.auto_fetch ? " on" : ""}`}
            aria-pressed={options.auto_fetch}
            onClick={() => onOptions({ ...options, auto_fetch: !options.auto_fetch })}
            title="DB 에 맞는 근거가 없으면 arXiv 에서 논문을 찾아 추가하고 다시 답합니다"
          >
            📥 논문 자동 수집
          </button>
          <span className="chip-hint">근거 {options.top_k_rerank}개 · {options.query_transform_mode}</span>
        </div>
        {running ? (
          <button type="button" className="send stop" onClick={onStop} title="답변 중지" aria-label="답변 중지">
            ■
          </button>
        ) : (
          <button type="button" className="send" onClick={submit} disabled={!text.trim()} title="보내기 (Enter)" aria-label="보내기">
            ↑
          </button>
        )}
      </div>
    </div>
  );
}
