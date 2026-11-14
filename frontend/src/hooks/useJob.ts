// 비교표·문헌 조사·관심 분야 확인처럼 진행 알림 뒤 결과 하나를 받는 SSE 작업.
import { useCallback, useRef, useState } from "react";
import { postSSE } from "../api/sse";
import { describeEvent, type LogLine } from "../lib/progress";

export interface JobState<T> {
  running: boolean;
  stage: string;
  log: LogLine[];
  result: T | null;
  error: string | null;
  startedAt: number | null;
}

const idle = { running: false, stage: "", log: [], result: null, error: null, startedAt: null };

export function useJob<T>(url: string) {
  const [state, setState] = useState<JobState<T>>(idle);
  const abort = useRef<AbortController | null>(null);

  const start = useCallback(
    async (body: unknown) => {
      abort.current?.abort();
      const ctl = new AbortController();
      abort.current = ctl;
      setState({ ...idle, running: true, stage: "시작하는 중", startedAt: Date.now() });
      try {
        await postSSE(
          url,
          body,
          (event, data) => {
            if (event === "result") setState((s) => ({ ...s, result: data as T }));
            else if (event === "fail") setState((s) => ({ ...s, error: data.message }));
            else {
              const u = describeEvent(event, data);
              if (u) setState((s) => ({ ...s, stage: u.stage ?? s.stage, log: [...s.log, ...u.lines] }));
            }
          },
          ctl.signal,
        );
      } catch (e) {
        if (!ctl.signal.aborted) setState((s) => ({ ...s, error: String((e as Error).message ?? e) }));
      } finally {
        if (abort.current === ctl) setState((s) => ({ ...s, running: false }));
      }
    },
    [url],
  );

  const cancel = useCallback(() => abort.current?.abort(), []);
  const reset = useCallback(() => setState(idle), []);
  return { ...state, start, cancel, reset };
}
