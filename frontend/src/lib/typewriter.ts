// 타자 효과의 속도 계산(순수 함수 — 테스트 가능).
//
// LLM 토큰은 고르게 오지 않는다: 몇 글자가 한꺼번에 오기도 하고 잠깐 멈추기도 한다.
// 받은 글자를 그대로 그리면 글이 덩어리로 튀거나 멈칫한다. 그래서 받은 글자는 버퍼에 두고,
// 화면 프레임(requestAnimationFrame, 초당 약 60번)마다 조금씩 꺼내 보여 준다.
//   - 속도 = max(최저 속도, 밀린 글자 수 / 목표 지연)
//     밀린 글자가 많으면 빨라지고 적으면 느려져서, 화면은 수신보다 약 LAG_SEC 만큼만 뒤처진다.
//     짧은 수신 공백(< LAG_SEC)은 버퍼가 메워 주므로 화면에서는 멈춤이 보이지 않는다.
//   - 생성이 끝나면(streaming=false) 남은 글자를 빠르게(FLUSH_SEC 비율, 최저 FLUSH_MIN_CPS) 마저 쓴다.

export const MIN_CPS = 35; // 최저 초당 글자 수(밀린 글자가 조금일 때)
export const LAG_SEC = 0.45; // 생성 중 화면이 수신보다 뒤처지는 목표 시간
export const FLUSH_SEC = 0.3; // 생성이 끝난 뒤 남은 글자를 다 쓰는 데 걸리는 시간
export const FLUSH_MIN_CPS = 140; // 끝난 뒤 최저 속도 — 마지막 몇 글자에서 느려지지 않게

export interface RevealState {
  shown: number; // 보여 준 글자 수(UTF-16 단위)
  carry: number; // 다음 프레임으로 넘기는 소수점 글자
}

export function revealStep(state: RevealState, targetLen: number, dtMs: number, streaming: boolean): RevealState {
  const backlog = targetLen - state.shown;
  if (backlog <= 0) return { shown: targetLen, carry: 0 };
  const cps = streaming ? Math.max(MIN_CPS, backlog / LAG_SEC) : Math.max(FLUSH_MIN_CPS, backlog / FLUSH_SEC);
  const budget = state.carry + (cps * Math.max(0, dtMs)) / 1000;
  const n = Math.floor(budget);
  return { shown: Math.min(targetLen, state.shown + n), carry: budget - n };
}

/** 이모지 같은 서로게이트 쌍을 반으로 자르지 않는 위치. */
export function safeCut(text: string, n: number): number {
  if (n <= 0) return 0;
  if (n >= text.length) return text.length;
  const c = text.charCodeAt(n - 1);
  return c >= 0xd800 && c <= 0xdbff ? n + 1 : n;
}

/** 두 문자열의 공통 앞부분 길이 — 답변이 바뀌어도 같은 부분은 다시 쓰지 않는다. */
export function commonPrefix(a: string, b: string): number {
  const n = Math.min(a.length, b.length);
  let i = 0;
  while (i < n && a.charCodeAt(i) === b.charCodeAt(i)) i++;
  return i;
}
