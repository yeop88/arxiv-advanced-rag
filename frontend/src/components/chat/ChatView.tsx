// 대화 흐름 + 자동 스크롤. 사용자가 맨 아래를 보고 있을 때만 따라 내려간다(위로 올려 읽는 중이면 그대로).
// 글이 길어지는 것은 ResizeObserver 로 감지해 같은 프레임에 즉시 스크롤한다(부드러운 스크롤은 글자를 따라가다 출렁임).
// 대화가 비어 있으면 첫 화면(인사 + 예시 질문 카드).
import { useEffect, useRef } from "react";
import type { ChatMessage, RateHandler } from "../../hooks/useChat";
import { MessageItem, type OpenEvidence } from "./MessageItem";

const EXAMPLES = [
  { icon: "⚖️", q: "DPR과 BM25의 검색 정확도를 비교해 주세요", hint: "표의 숫자를 근거로 비교" },
  { icon: "🔁", q: "리랭킹은 RAG에서 왜 필요한가요?", hint: "개념 설명 + 인용" },
  { icon: "🧮", q: "Mamba 논문의 ZOH 이산화 공식은?", hint: "수식을 근거에서 찾기" },
  { icon: "🧭", q: "Lost in the Middle 논문의 핵심 발견은 무엇인가요?", hint: "논문 요약" },
];

interface Props {
  messages: ChatMessage[];
  onExample: (q: string) => void;
  onRate?: RateHandler;
  onEvidence?: OpenEvidence;
  selectedId?: string | null;
}

export function ChatView({ messages, onExample, onRate, onEvidence, selectedId }: Props) {
  const scroller = useRef<HTMLDivElement>(null);
  const content = useRef<HTMLDivElement>(null);
  const pinned = useRef(true);

  useEffect(() => {
    const el = scroller.current;
    const inner = content.current;
    if (!el || !inner) return;
    const onScroll = () => {
      pinned.current = el.scrollHeight - el.scrollTop - el.clientHeight < 80;
    };
    const ro = new ResizeObserver(() => {
      if (pinned.current) el.scrollTop = el.scrollHeight;
    });
    el.addEventListener("scroll", onScroll, { passive: true });
    ro.observe(inner);
    return () => {
      el.removeEventListener("scroll", onScroll);
      ro.disconnect();
    };
  }, []);

  // 새 질문을 보내면 맨 아래로
  const count = messages.length;
  useEffect(() => {
    pinned.current = true;
    const el = scroller.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [count]);

  return (
    <div className="thread" ref={scroller}>
      <div className="thread-inner" ref={content}>
        {messages.length === 0 && (
          <div className="hero">
            <div className="hero-mark" aria-hidden>
              ◆
            </div>
            <h1>논문을 근거로 답해 드려요</h1>
            <p className="muted">
              DB 의 arXiv 논문에서 근거를 찾아 문장마다 [n] 으로 인용하고, 인용이 맞는지 문장 단위로 확인합니다.
              <br />
              근거가 없으면 arXiv 에서 논문을 찾아 추가합니다.
            </p>
            <div className="examples">
              {EXAMPLES.map((e) => (
                <button key={e.q} type="button" className="example" onClick={() => onExample(e.q)}>
                  <span className="ex-icon" aria-hidden>
                    {e.icon}
                  </span>
                  <span className="ex-q">{e.q}</span>
                  <span className="ex-hint">{e.hint}</span>
                </button>
              ))}
            </div>
          </div>
        )}
        {messages.map((m) => (
          <MessageItem key={m.id} m={m} onRate={onRate} onEvidence={onEvidence} selected={selectedId === m.id} />
        ))}
      </div>
    </div>
  );
}
