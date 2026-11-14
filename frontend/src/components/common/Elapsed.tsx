// 경과 시간(초). 자기 자신만 1초마다 다시 그린다 — 답변 영역은 건드리지 않는다.
import { useEffect, useState } from "react";

export function Elapsed({ since, until }: { since: number; until?: number }) {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    if (until) return;
    const id = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(id);
  }, [until]);
  return <span className="elapsed">{Math.max(0, Math.round(((until ?? now) - since) / 1000))}초</span>;
}
