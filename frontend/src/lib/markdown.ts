// 스트리밍 마크다운 도우미(순수 함수 — 테스트 가능).

/**
 * 답변을 빈 줄 기준 블록(문단·목록·표·코드)으로 나눈다.
 * 다 쓴 블록은 내용이 더 바뀌지 않으므로 화면에서 다시 해석하지 않고(memo),
 * 쓰는 중인 마지막 블록만 매 프레임 다시 그린다 → 답이 길어져도 글자당 비용이 일정하다.
 * 코드 블록(```)과 수식 블록($$) 안의 빈 줄에서는 나누지 않는다.
 */
export function splitBlocks(text: string): string[] {
  const blocks: string[] = [];
  let cur: string[] = [];
  let inFence = false;
  let inMath = false;
  for (const line of text.split("\n")) {
    const t = line.trim();
    if (t.startsWith("```")) inFence = !inFence;
    else if (!inFence && (t.match(/\$\$/g)?.length ?? 0) % 2 === 1) inMath = !inMath;
    if (t === "" && !inFence && !inMath) {
      if (cur.length) blocks.push(cur.join("\n"));
      cur = [];
    } else {
      cur.push(line);
    }
  }
  if (cur.length) blocks.push(cur.join("\n"));
  return blocks;
}

/** 본문의 [1], [1, 2], [1][3] 인용을 #cite-n 링크로 바꾼다(화면에서 근거 칩으로 그림). */
export function linkCitations(text: string): string {
  return text.replace(/\[(\d{1,2}(?:\s*[,，]\s*\d{1,2})*)\](?![(:])/g, (_m, nums: string) =>
    nums
      .split(/\s*[,，]\s*/)
      .map((n) => `[${n}](#cite-${n})`)
      .join(""),
  );
}

/** 관심 분야 소식지의 <small> 태그를 마크다운 기울임으로(원시 HTML 은 그리지 않는다). */
export function stripSmallTags(md: string): string {
  return md.replace(/<small>([\s\S]*?)<\/small>/g, "_$1_");
}
