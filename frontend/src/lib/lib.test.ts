import { describe, expect, it } from "vitest";
import { createSSEParser } from "../api/sse";
import { linkCitations, splitBlocks, stripSmallTags } from "./markdown";
import { describeEvent, timingSummary } from "./progress";
import { FLUSH_SEC, LAG_SEC, MIN_CPS, commonPrefix, revealStep, safeCut, type RevealState } from "./typewriter";

/** 실제 화면처럼 16.7ms 프레임으로 돌리며 프레임마다 보여 준 글자 수를 모은다. */
function simulate(arrivals: [number, number][], totalMs: number, streamingUntil: number) {
  let st: RevealState = { shown: 0, carry: 0 };
  let received = 0;
  const perFrame: number[] = [];
  for (let t = 0; t <= totalMs; t += 1000 / 60) {
    for (const [at, n] of arrivals) if (at <= t && at > t - 1000 / 60) received += n;
    const before = st.shown;
    st = revealStep(st, received, 1000 / 60, t < streamingUntil);
    perFrame.push(st.shown - before);
  }
  return { st, received, perFrame };
}

describe("typewriter", () => {
  it("몰려 온 토큰도 한 번에 튀지 않고 여러 프레임에 나눠 보인다", () => {
    // 0.5초마다 40글자씩 덩어리로 도착(고르지 않은 수신)
    const arrivals: [number, number][] = Array.from({ length: 10 }, (_, i) => [i * 500, 40]);
    const { perFrame } = simulate(arrivals, 6000, 5000);
    expect(Math.max(...perFrame)).toBeLessThanOrEqual(3); // 한 프레임에 최대 몇 글자
  });

  it("수신이 잠깐 멈춰도(0.4초) 화면 글자는 계속 나온다", () => {
    // 초당 60글자를 고르게 받다가 1.0~1.4초 사이 공백
    const arrivals: [number, number][] = [];
    for (let t = 0; t < 2000; t += 50) if (t < 1000 || t >= 1400) arrivals.push([t, 3]);
    const { perFrame } = simulate(arrivals, 2000, 3000);
    const gapFrames = perFrame.slice(Math.round(1.05 * 60), Math.round(1.35 * 60));
    expect(gapFrames.reduce((a, b) => a + b, 0)).toBeGreaterThan(0);
  });

  it("뒤처짐은 목표 지연 근처로 유지되고, 끝나면 금방 다 쓴다", () => {
    const arrivals: [number, number][] = [];
    for (let t = 0; t < 3000; t += 40) arrivals.push([t, 4]); // 초당 100글자
    const mid = simulate(arrivals, 2000, 99999);
    expect(mid.received - mid.st.shown).toBeLessThan(100 * LAG_SEC * 1.6);
    const end = simulate(arrivals, 3000 + FLUSH_SEC * 1000 + 300, 3000);
    expect(end.st.shown).toBe(end.received);
  });

  it("밀린 글자가 적으면 최저 속도로 쓴다", () => {
    const st = revealStep({ shown: 0, carry: 0 }, 5, 1000, true);
    expect(st.shown).toBe(5);
    const slow = revealStep({ shown: 0, carry: 0 }, 1000, 100, true);
    expect(slow.shown).toBeGreaterThanOrEqual(Math.floor(MIN_CPS / 10));
  });

  it("서로게이트 쌍은 자르지 않는다", () => {
    const s = "가😀나";
    expect(safeCut(s, 2)).toBe(3);
    expect(safeCut(s, 1)).toBe(1);
    expect(commonPrefix("abc", "abd")).toBe(2);
  });
});

describe("markdown", () => {
  it("빈 줄로 블록을 나누되 코드·수식 블록 안에서는 나누지 않는다", () => {
    const md = "첫 문단\n\n```py\na = 1\n\nb = 2\n```\n\n$$\nx\n\ny\n$$\n\n- 항목";
    expect(splitBlocks(md)).toEqual(["첫 문단", "```py\na = 1\n\nb = 2\n```", "$$\nx\n\ny\n$$", "- 항목"]);
  });

  it("인용 [n]·[n, m] 을 링크로 바꾸고 마크다운 링크는 건드리지 않는다", () => {
    expect(linkCitations("정밀도를 높입니다 [1].")).toBe("정밀도를 높입니다 [1](#cite-1).");
    expect(linkCitations("근거 [1, 3]")).toBe("근거 [1](#cite-1)[3](#cite-3)");
    expect(linkCitations("[1](http://x)")).toBe("[1](http://x)");
  });

  it("소식지의 <small> 을 기울임으로", () => {
    expect(stripSmallTags("<small>검색어 `x`</small>")).toBe("_검색어 `x`_");
  });
});

describe("sse", () => {
  it("조각난 스트림에서도 이벤트를 하나씩 정확히 꺼낸다", () => {
    const got: [string, unknown][] = [];
    const p = createSSEParser((e, d) => got.push([e, d]));
    p.feed('event: token\ndata: {"delta":"가');
    p.feed('나"}\n\n: ping\n\nevent: result\ndata: {"answer":"x"}\n');
    expect(got).toEqual([["token", { delta: "가나" }]]);
    p.feed("\n");
    expect(got[1]).toEqual(["result", { answer: "x" }]);
  });
});

describe("progress", () => {
  it("단계 알림을 로그 문장과 단계 이름으로", () => {
    const u = describeEvent("retrieve_done", { n_candidates: 22, papers: ["A", "B"], sec: 0.4 });
    expect(u?.lines[0].text).toContain("후보 22개 중 근거 2개");
    expect(describeEvent("verify_start", {})?.stage).toBe("4/4 · 답변 검증 중");
    expect(describeEvent("end", {})).toBeNull();
    expect(timingSummary({ "1_query_transform": 2.2, "3_generate": 30.4 })).toBe("변환 2s · 작성 30s");
  });
});
