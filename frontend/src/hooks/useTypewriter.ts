// 받은 답변(target)을 프레임마다 조금씩 드러내는 훅. 속도 계산은 lib/typewriter.ts.
// 프레임마다 바뀌는 건 이 훅을 쓰는 컴포넌트(쓰는 중인 답변 하나)뿐이라 나머지 화면은 다시 그리지 않는다.
import { useEffect, useReducer, useRef } from "react";
import { commonPrefix, revealStep, safeCut, type RevealState } from "../lib/typewriter";

export function useTypewriter(target: string, streaming: boolean): { text: string; typing: boolean } {
  const state = useRef<RevealState>({ shown: streaming ? 0 : target.length, carry: 0 });
  const prevTarget = useRef(target);
  const targetRef = useRef(target);
  const streamingRef = useRef(streaming);
  const frame = useRef<number | null>(null);
  const [, rerender] = useReducer((x: number) => x + 1, 0);

  // 답변이 이어지지 않고 바뀌면(다시 작성 등) 같은 앞부분까지만 남기고 거기서부터 다시 쓴다
  if (prevTarget.current !== target) {
    if (!target.startsWith(prevTarget.current.slice(0, state.current.shown))) {
      state.current = { shown: Math.min(state.current.shown, commonPrefix(prevTarget.current, target)), carry: 0 };
    }
    prevTarget.current = target;
  }
  targetRef.current = target;
  streamingRef.current = streaming;

  useEffect(() => {
    if (frame.current !== null || state.current.shown >= target.length) return;
    let last = performance.now();
    const tick = (now: number) => {
      const next = revealStep(state.current, targetRef.current.length, now - last, streamingRef.current);
      last = now;
      const changed = next.shown !== state.current.shown;
      state.current = next;
      if (changed) rerender();
      if (next.shown < targetRef.current.length) {
        frame.current = requestAnimationFrame(tick);
      } else {
        frame.current = null; // 다 따라잡음 — 새 글자가 오면 effect 가 다시 시작
      }
    };
    frame.current = requestAnimationFrame(tick);
  }, [target, streaming]);

  useEffect(
    () => () => {
      if (frame.current !== null) cancelAnimationFrame(frame.current);
      frame.current = null;
    },
    [],
  );

  const shown = safeCut(target, state.current.shown);
  return { text: target.slice(0, shown), typing: shown < target.length };
}
