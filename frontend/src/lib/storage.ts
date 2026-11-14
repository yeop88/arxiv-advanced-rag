// 브라우저 저장(대화·설정 기억) — 사생활 모드 등에서 막혀도 화면은 정상 동작하도록 실패를 삼킨다.

export function load<T>(key: string, fallback: T): T {
  try {
    const raw = localStorage.getItem(key);
    return raw ? (JSON.parse(raw) as T) : fallback;
  } catch {
    return fallback;
  }
}

export function save(key: string, value: unknown): void {
  try {
    localStorage.setItem(key, JSON.stringify(value));
  } catch {
    /* 저장 공간 부족·차단 — 무시 */
  }
}
