// REST API 호출(짧은 요청). 오래 걸리는 작업은 sse.ts 의 postSSE 를 쓴다.
import { describeError } from "./sse";
import type { AppConfig, CitationsResponse, Rating, ServerStatus, WatchState } from "./types";

async function json<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, init);
  if (!res.ok) throw new Error(await describeError(res));
  return res.json() as Promise<T>;
}

const post = (body: unknown): RequestInit => ({
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body),
});

export const api = {
  status: () => json<ServerStatus>("/api/status"),
  config: () => json<AppConfig>("/api/config"),
  citations: (sources: string[], external = false) => {
    const q = new URLSearchParams();
    sources.forEach((s) => q.append("sources", s));
    if (external) q.set("external", "true");
    return json<CitationsResponse>(`/api/citations?${q}`);
  },
  feedback: (log_id: string, rating: Rating, note?: string) =>
    json<{ ok: boolean }>("/api/feedback", post({ log_id, rating, note: note || null })),
  watch: () => json<WatchState>("/api/watch"),
  addWatchTopic: (topic: string) => json<WatchState>("/api/watch/topics", post({ topic })),
  removeWatchTopic: (topic: string) =>
    json<WatchState>(`/api/watch/topics/${encodeURIComponent(topic)}`, { method: "DELETE" }),
};
