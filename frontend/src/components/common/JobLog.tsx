// 진행 로그 줄 목록.
import type { LogLine } from "../../lib/progress";

export function JobLog({ lines }: { lines: LogLine[] }) {
  if (!lines.length) return null;
  return (
    <ul className="log">
      {lines.map((l, i) => (
        <li key={i} className={l.sub ? "sub" : ""}>
          {l.text}
        </li>
      ))}
    </ul>
  );
}
