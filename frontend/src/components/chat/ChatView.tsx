// 대화 목록 + 자동 스크롤. 사용자가 맨 아래를 보고 있을 때만 따라 내려간다(위로 올려 읽는 중이면 그대로).
// 글이 길어지는 것은 ResizeObserver 로 감지해 같은 프레임에 즉시 스크롤한다(부드러운 스크롤은 글자를 따라가다 출렁임).
import { useEffect, useRef, type ReactNode } from "react";
import type { ChatMessage } from "../../hooks/useChat";
import { MessageItem } from "./MessageItem";

const EXAMPLES = [
  "리랭킹은 RAG에서 왜 필요한가요?",
  "Mamba 논문의 ZOH 이산화 공식은?",
  "DPR과 BM25의 검색 성능을 비교해 주세요",
  "Lost in the Middle 논문의 핵심 발견은 무엇인가요?",
];

interface Props {
  messages: ChatMessage[];
  header?: ReactNode;
  onExample: (q: string) => void;
}

export function ChatView({ messages, header, onExample }: Props) {
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
    <div className="chat-scroll" ref={scroller}>
      <div className="chat-inner" ref={content}>
        {header}
        {messages.length === 0 && (
          <div className="empty">
            <h2>무엇이 궁금하세요?</h2>
            <p className="muted">DB 의 arXiv 논문을 근거로, 문장마다 인용 [n]을 달아 답합니다.</p>
            <div className="examples">
              {EXAMPLES.map((q) => (
                <button key={q} type="button" className="example" onClick={() => onExample(q)}>
                  {q}
                </button>
              ))}
            </div>
          </div>
        )}
        {messages.map((m) => (
          <MessageItem key={m.id} m={m} />
        ))}
      </div>
    </div>
  );
}
