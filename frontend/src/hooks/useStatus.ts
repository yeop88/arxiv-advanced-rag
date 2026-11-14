// 서버 상태(모델·DB 로딩, LLM 서버)를 2초마다 확인.
import { useEffect, useState } from "react";
import { api } from "../api/client";
import type { ServerStatus } from "../api/types";

export function useStatus(intervalMs = 2000): { status: ServerStatus | null; offline: boolean } {
  const [status, setStatus] = useState<ServerStatus | null>(null);
  const [offline, setOffline] = useState(false);

  useEffect(() => {
    let alive = true;
    const poll = async () => {
      try {
        const s = await api.status();
        if (alive) {
          setStatus(s);
          setOffline(false);
        }
      } catch {
        if (alive) setOffline(true);
      }
    };
    poll();
    const id = window.setInterval(poll, intervalMs);
    return () => {
      alive = false;
      window.clearInterval(id);
    };
  }, [intervalMs]);

  return { status, offline };
}
