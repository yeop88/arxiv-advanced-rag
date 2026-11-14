// SSE(Server-Sent Events) 파서와 POST 스트리밍 요청.
// EventSource 는 GET 만 되므로 fetch 의 ReadableStream 을 직접 읽는다.
// 받은 조각을 바로 해석해 이벤트마다 콜백을 부른다(모아 두지 않음) → 토큰이 도착하는 즉시 화면에 전달된다.

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export type SSEHandler = (event: string, data: any) => void;

export function createSSEParser(onEvent: SSEHandler) {
  let buf = "";

  function dispatch(block: string) {
    let event = "message";
    const data: string[] = [];
    for (const line of block.split("\n")) {
      if (!line || line.startsWith(":")) continue; // 주석(ping)
      const i = line.indexOf(":");
      const field = i < 0 ? line : line.slice(0, i);
      const value = i < 0 ? "" : line.slice(i + 1).replace(/^ /, "");
      if (field === "event") event = value;
      else if (field === "data") data.push(value);
    }
    if (!data.length) return;
    const raw = data.join("\n");
    let parsed: unknown = raw;
    try {
      parsed = JSON.parse(raw);
    } catch {
      /* 문자열 그대로 */
    }
    onEvent(event, parsed);
  }

  return {
    feed(chunk: string) {
      buf += chunk.replace(/\r/g, "");
      let i: number;
      while ((i = buf.indexOf("\n\n")) >= 0) {
        const block = buf.slice(0, i);
        buf = buf.slice(i + 2);
        dispatch(block);
      }
    },
  };
}

export async function postSSE(url: string, body: unknown, onEvent: SSEHandler, signal?: AbortSignal): Promise<void> {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
    body: JSON.stringify(body),
    signal,
  });
  if (!res.ok || !res.body) {
    throw new Error(await describeError(res));
  }
  const reader = res.body.pipeThrough(new TextDecoderStream()).getReader();
  const parser = createSSEParser(onEvent);
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    parser.feed(value);
  }
}

export async function describeError(res: Response): Promise<string> {
  try {
    const j = await res.json();
    if (Array.isArray(j.detail)) return j.detail.map((d: { msg: string }) => d.msg).join("; ");
    return j.detail ?? `HTTP ${res.status}`;
  } catch {
    return `HTTP ${res.status}`;
  }
}
