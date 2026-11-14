// 파이프라인 진행 알림(rag/pipeline.py 의 progress 이벤트) → 화면 진행 로그 문장과 현재 단계 이름.

export interface LogLine {
  text: string;
  sub?: boolean; // 들여 쓴 보조 줄
}

export interface ProgressUpdate {
  stage?: string; // 바뀐 현재 단계(없으면 그대로)
  lines: LogLine[];
}

const sec = (v: unknown) => (typeof v === "number" ? `${v.toFixed(1)}초` : "");

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export function describeEvent(event: string, info: any): ProgressUpdate | null {
  switch (event) {
    case "waiting": {
      const parts: string[] = [];
      if (info.loader_state !== "ready") parts.push(`모델·DB: ${info.step} (${info.done?.length ?? 0}/4)`);
      if (!["ok", "n/a"].includes(info.llm_state)) parts.push(`LLM 서버: ${info.llm_desc}`);
      return { stage: `준비되면 바로 답합니다 · ${parts.join(" / ")}`, lines: [] };
    }
    case "ready":
      return { stage: "준비 완료", lines: [{ text: `✓ 모델·DB·LLM 준비 완료 (${sec(info.waited)} 기다림)` }] };
    case "queued":
      return {
        stage:
          info.background === "enrich"
            ? "📊 백그라운드 표·그림 처리 한 쪽을 마치는 중 — 끝나면 바로 답합니다"
            : "앞선 작업이 끝나길 기다리는 중",
        lines: [],
      };
    case "contextualize_start":
      return { stage: "0/4 · 대화 맥락 반영 중", lines: [] };
    case "contextualize_done":
      return info.changed ? { lines: [{ text: `🔁 이전 대화를 반영해 질문을 바꿨습니다: ${info.standalone}` }] } : null;
    case "transform_start":
      return {
        stage: "1/4 · 질의 변환 중",
        lines: [{ text: `1️⃣ 질의 변환 — 검색이 잘 되도록 질문을 여러 형태로 바꾸는 중 (${info.mode})` }],
      };
    case "transform_done":
      return {
        lines: [
          { text: `✓ 검색 질의 ${info.queries?.length ?? 0}개 (${sec(info.sec)})`, sub: true },
          ...(info.queries ?? []).map((q: string) => ({ text: `· ${q}`, sub: true })),
        ],
      };
    case "retrieve_start":
      return info.again
        ? { stage: "2/4 · 다시 검색 중", lines: [{ text: "2️⃣ 다시 검색 — 새로 추가한 논문까지 포함해 근거를 다시 고르는 중" }] }
        : {
            stage: "2/4 · 검색·리랭킹 중",
            lines: [{ text: `2️⃣ 검색·리랭킹 — 질의 ${info.n_queries}개로 논문을 찾고 관련도순으로 고르는 중` }],
          };
    case "retrieve_done":
      return {
        lines: [
          { text: `✓ 후보 ${info.n_candidates}개 중 근거 ${info.papers?.length ?? 0}개 선택 (${sec(info.sec)})`, sub: true },
          ...(info.papers ?? []).map((p: string, i: number) => ({ text: `[${i + 1}] ${p.slice(0, 80)}`, sub: true })),
        ],
      };
    case "fetch_start": {
      const why =
        info.reason === "answer"
          ? "답변이 '근거에서 찾을 수 없다'고 밝혔습니다"
          : `가장 관련 높은 근거 점수 ${(info.top_score ?? 0).toFixed(2)} < 기준 ${(info.threshold ?? 0).toFixed(2)}`;
      return { stage: "📥 논문 찾는 중", lines: [{ text: `📥 근거 부족 — ${why}. arXiv 에서 관련 논문을 찾아 DB 에 추가합니다` }] };
    }
    case "fetch_queries":
      return { lines: [{ text: `· arXiv 검색어: ${(info.queries ?? []).join(" / ")}`, sub: true }] };
    case "fetch_title_guess":
      return {
        lines: [{ text: `· ${info.more ? "원 논문 제목 후보 더" : "원 논문 제목 추정"}: ${info.title ?? ""}`, sub: true }],
      };
    case "fetch_warn":
      return { lines: [{ text: `⚠️ ${info.message ?? ""}`, sub: true }] };
    case "fetch_candidates": {
      const skipped = info.n_existing_skipped ? `(이미 있는 ${info.n_existing_skipped}편 제외)` : "";
      const src = (info.sources ?? []).join("·") || "검색";
      return {
        stage: "📥 후보 논문 고르는 중",
        lines: [{ text: `· ${src}에서 DB 에 없는 후보 논문 ${info.n}편${skipped} — 초록으로 관련도 채점 중`, sub: true }],
      };
    }
    case "fetch_selected": {
      const papers: [string, number][] = info.papers ?? [];
      if (!papers.length) {
        const br = info.best_rejected;
        return {
          lines: [
            {
              text: `· 기준을 넘는 관련 논문이 없어 추가하지 않습니다${br != null ? ` (가장 높은 점수 ${br.toFixed(2)})` : ""}`,
              sub: true,
            },
          ],
        };
      }
      return {
        lines: [
          { text: `· 관련 논문 ${papers.length}편 선택 — 전문(PDF)을 받아 DB 에 추가합니다`, sub: true },
          ...papers.map(([t, s]) => ({ text: `· ${t.slice(0, 80)} (관련도 ${s.toFixed(2)})`, sub: true })),
        ],
      };
    }
    case "fetch_paper_start":
      return { stage: `📥 논문 ${info.i}/${info.n} 전문 받는 중`, lines: [] };
    case "fetch_paper_done":
      return {
        lines: [
          {
            text: `✓ [${info.i}/${info.n}] ${String(info.title).slice(0, 60)} — ${info.fulltext ? "전문" : "초록(PDF 실패)"} ${info.chunks}청크 추가`,
            sub: true,
          },
        ],
      };
    case "fetch_evicted":
      return { lines: [{ text: `🧹 수집 예산을 넘어 오래 안 쓰인 논문 ${info.papers?.length ?? 0}편을 정리했습니다`, sub: true }] };
    case "fetch_done": {
      const n = info.added?.length ?? 0;
      return {
        lines: [
          {
            text: `✓ 논문 자동 수집 ${n ? `완료: ${n}편 추가` : "— 추가한 논문 없음(기존 근거로 진행)"} (${sec(info.sec)})`,
            sub: true,
          },
        ],
      };
    }
    case "generate_start":
      return info.again
        ? { stage: "3/4 · 답변 다시 작성 중", lines: [{ text: `3️⃣ 답변 다시 작성 — 새 논문을 포함한 근거 ${info.n_chunks}개로 다시 쓰는 중` }] }
        : { stage: "3/4 · 답변 작성 준비 중", lines: [{ text: `3️⃣ 답변 작성 — 근거 ${info.n_chunks}개로 답변을 쓰는 중` }] };
    case "generate_done":
      return { lines: [{ text: `✓ 답변 작성 완료 (${sec(info.sec)}, ${info.chars}자)`, sub: true }] };
    case "verify_start":
      return {
        stage: `4/4 · 답변 ${info.again ? "다시 " : ""}검증 중`,
        lines: [{ text: `4️⃣ 답변 ${info.again ? "다시 " : ""}검증 — 답변의 각 주장이 근거에 실제로 있는지 대조하는 중` }],
      };
    case "verify_done": {
      const bad = info.unsupported?.length ?? 0;
      return {
        lines: [
          {
            text: `✓ ${info.grounded ? "✅ 근거 충실" : "⚠️ 근거 부족"} · faithfulness ${(info.faithfulness ?? 0).toFixed(2)}${
              bad ? ` · 근거 없는 주장 ${bad}개` : ""
            } (${sec(info.sec)})`,
            sub: true,
          },
        ],
      };
    }
    case "cite_check_done": {
      const s = info.summary ?? {};
      return {
        stage: "4/4 · 인용 확인",
        lines: [
          {
            text: `🔎 문장별 인용 확인 — 주장 ${s.claims ?? 0}문장 중 ${s.supported ?? 0}문장이 인용한 근거와 맞음 (${sec(info.sec)})`,
            sub: true,
          },
        ],
      };
    }
    case "regenerate_start":
      return {
        stage: "3/4 · 답변 고치는 중",
        lines: [{ text: `🔁 다시 작성 (시도 ${info.attempt}) — 근거 없는 주장을 빼고 답변을 고치는 중` }],
      };
    case "error":
      return { lines: [{ text: `⚠️ ${info.stage} 단계 오류: ${String(info.message ?? "").slice(0, 200)}` }] };
    // 비교표·문헌 조사·관심 분야
    case "table_paper":
      return { stage: `논문 ${info.i}/${info.n} 요약 중`, lines: [{ text: `${info.i}/${info.n} ${String(info.title).slice(0, 60)}` }] };
    case "table_verify":
      return { stage: "표 검증 중", lines: [{ text: "표 검증 중" }] };
    case "survey_gather":
      return { stage: "관련 논문 모으는 중", lines: [{ text: "관련 논문 모으는 중" }] };
    case "survey_write":
      return { stage: "초안 작성 중", lines: [{ text: `근거 ${info.n}개(논문 ${info.papers}편)로 초안 작성 중` }] };
    case "survey_verify":
      return { stage: "초안 검증 중", lines: [{ text: "초안 검증 중" }] };
    case "watch_topic":
      return { stage: `${info.i}/${info.n} ${info.topic}`, lines: [{ text: `${info.i}/${info.n} ${info.topic}` }] };
    default:
      return null;
  }
}

const TIMING_LABELS: [string, string][] = [
  ["변환", "1_query_transform"],
  ["검색", "2_retrieve_rerank"],
  ["논문수집", "2b_auto_fetch"],
  ["재검색", "2c_retrieve_again"],
  ["작성", "3_generate"],
  ["재작성", "3c_generate_again"],
  ["검증", "4_verify"],
];

export function timingSummary(t: Record<string, number>): string {
  return TIMING_LABELS.filter(([, k]) => k in t)
    .map(([n, k]) => `${n} ${t[k].toFixed(0)}s`)
    .join(" · ");
}
