// 대화 상태: 질문 보내기(SSE) → 진행 로그·답변 토큰·결과를 메시지에 반영. 중지·초기화·브라우저에 기억.
import { useCallback, useEffect, useRef, useState } from "react";
import { postSSE } from "../api/sse";
import { api } from "../api/client";
import type { ChatOptions, ChatResult, Rating } from "../api/types";
import { describeEvent, type LogLine } from "../lib/progress";
import { load, save } from "../lib/storage";

export type Phase = "working" | "writing" | "checking" | "done" | "error" | "stopped";

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  phase?: Phase;
  stage?: string;
  log?: LogLine[];
  startedAt?: number;
  finishedAt?: number;
  result?: ChatResult;
  drafts?: string[]; // 검증 뒤 고치기 전 초안
  reasoning?: string; // Thinking 모드의 추론(답변 본문과 따로)
  citeTitles?: string[]; // 결과 전에도 [n] 인용에 제목을 보여 주려고(retrieve_done)
  mode?: string;
  error?: string;
  feedback?: { rating: Rating; note?: string }; // 👍/👎(서버 기록에도 저장)
}

export type RateHandler = (id: string, rating: Rating, note?: string) => Promise<void>;

const STORE_KEY = "arxiv-rag.messages.v1";
const MAX_STORED = 60;
const newId = () => Math.random().toString(36).slice(2, 10) + Date.now().toString(36);

export const isActive = (m: ChatMessage) => m.phase === "working" || m.phase === "writing" || m.phase === "checking";

export function useChat() {
  const [messages, setMessages] = useState<ChatMessage[]>(() =>
    // 새로고침 전에 진행 중이던 답변은 중단된 것으로 표시
    load<ChatMessage[]>(STORE_KEY, []).map((m) => (isActive(m) ? { ...m, phase: "stopped" } : m)),
  );
  const abort = useRef<AbortController | null>(null);
  const latest = useRef(messages); // rate 가 토큰마다 바뀌지 않게(지난 메시지 memo 유지)
  latest.current = messages;
  const running = messages.some(isActive);

  useEffect(() => {
    if (!running) save(STORE_KEY, messages.slice(-MAX_STORED).map(({ reasoning: _r, ...m }) => m));
  }, [messages, running]);

  const update = useCallback((id: string, fn: (m: ChatMessage) => ChatMessage) => {
    setMessages((ms) => ms.map((m) => (m.id === id ? fn(m) : m)));
  }, []);

  const send = useCallback(
    async (question: string, options: ChatOptions) => {
      const q = question.trim();
      if (!q || running) return;
      const history = messages
        .filter((m) => m.content && (m.role === "user" || m.phase === "done"))
        .map((m) => ({ role: m.role, content: m.content }));
      const id = newId();
      setMessages((ms) => [
        ...ms,
        { id: newId(), role: "user", content: q },
        { id, role: "assistant", content: "", phase: "working", stage: "질문을 받았습니다", log: [], startedAt: Date.now(), drafts: [] },
      ]);

      const ctl = new AbortController();
      abort.current = ctl;
      const log = (m: ChatMessage, lines: LogLine[], stage?: string): ChatMessage =>
        lines.length || stage ? { ...m, log: [...(m.log ?? []), ...lines], stage: stage ?? m.stage } : m;

      try {
        await postSSE(
          "/api/chat",
          { question: q, history, options },
          (event, data) => {
            switch (event) {
              case "token":
                update(id, (m) => ({ ...m, content: m.content + data.delta, phase: m.phase === "working" ? "writing" : m.phase }));
                return;
              case "reasoning":
                update(id, (m) => ({ ...m, reasoning: (m.reasoning ?? "") + data.delta, stage: m.content ? m.stage : "3/4 · 생각하는 중" }));
                return;
              case "answer_reset":
                update(id, (m) => ({ ...m, content: "" }));
                return;
              case "start":
                update(id, (m) => ({ ...m, mode: data.mode || undefined }));
                return;
              case "result":
                update(id, (m) => ({
                  ...m,
                  // 스트리밍한 글과 같으면 그대로 두어 화면이 다시 그려지지 않게
                  content: m.content.trim() === data.answer.trim() ? m.content : data.answer,
                  result: data as ChatResult,
                  phase: "done",
                  finishedAt: Date.now(),
                }));
                return;
              case "fail":
                update(id, (m) => ({ ...m, phase: "error", error: data.message, finishedAt: Date.now() }));
                return;
            }
            const u = describeEvent(event, data);
            if (!u) return; // end 등 화면에 보일 것 없는 알림
            update(id, (m) => {
              let next = u ? log(m, u.lines, u.stage) : m;
              if (event === "retrieve_done") next = { ...next, citeTitles: data.papers ?? [] };
              if (event === "verify_start") next = { ...next, phase: "checking" };
              if (event === "regenerate_start" || (event === "generate_start" && data.again)) {
                // 고친 답변을 새로 쓴다 — 고치기 전 초안은 접어서 남긴다
                next = { ...next, drafts: m.content ? [...(m.drafts ?? []), m.content] : m.drafts, content: "", phase: "writing" };
              }
              return next;
            });
          },
          ctl.signal,
        );
        update(id, (m) => (isActive(m) ? { ...m, phase: "error", error: "서버 연결이 끊겼습니다.", finishedAt: Date.now() } : m));
      } catch (e) {
        update(id, (m) =>
          ctl.signal.aborted
            ? { ...m, phase: "stopped", finishedAt: Date.now() }
            : { ...m, phase: "error", error: `서버에 연결할 수 없습니다: ${(e as Error).message}`, finishedAt: Date.now() },
        );
      } finally {
        if (abort.current === ctl) abort.current = null;
      }
    },
    [messages, running, update],
  );

  // 👍/👎: 서버 기록에 붙이고, 성공하면 메시지에도 남긴다(새로고침 뒤에도 보이게)
  const rate = useCallback<RateHandler>(
    async (id, rating, note) => {
      const m = latest.current.find((x) => x.id === id);
      const logId = m?.result?.log_id;
      if (!logId) throw new Error("이 답변은 서버 기록이 없어 피드백을 남길 수 없습니다.");
      await api.feedback(logId, rating, note);
      update(id, (x) => ({ ...x, feedback: rating ? { rating, note } : undefined }));
    },
    [update],
  );

  const stop = useCallback(() => abort.current?.abort(), []);
  const clear = useCallback(() => {
    abort.current?.abort();
    setMessages([]);
    save(STORE_KEY, []);
  }, []);

  return { messages, running, send, stop, clear, rate };
}
