// 서버(server/schemas.py, server/routes/*) 응답과 같은 모양

export type QueryMode = "multi_query" | "hyde" | "decompose" | "none";
export type OnOff = "on" | "off";

export interface ChatOptions {
  query_transform_mode: QueryMode;
  use_hybrid: boolean;
  top_k_rerank: number;
  enable_verification: boolean;
  auto_fetch: boolean;
  local_thinking: OnOff;
  local_thinking_light: OnOff;
}

export interface AppConfig {
  provider: string;
  model: string;
  embed_model: string;
  vector_backend: string;
  openalex_enabled: boolean;
  defaults: ChatOptions;
}

export interface ServerStatus {
  loader: {
    state: "loading" | "ready" | "error";
    step: string;
    done: [string, number][];
    error: string | null;
    seconds: number;
  };
  llm: { state: "ok" | "loading" | "down" | "n/a"; desc: string };
  enrich?: EnrichStatus | null;
}

// 자동 수집 논문의 표·그림·수식 백그라운드 처리(server/enricher.py)
export interface EnrichStatus {
  enabled: boolean;
  queue: number;
  current: { source: string; title: string; page: number; pages: number } | null;
  recent: { source: string; title: string; added?: number; sec?: number; error?: string }[];
}

// 사용 지표(/api/metrics, 질문 기록에서 계산)
export interface Metrics {
  enabled: boolean;
  total: number;
  errors: number;
  error_rate: number | null;
  up: number;
  down: number;
  mean_total_sec: number | null;
  verify_grounded_rate: number | null;
  citation_precision: number | null;
  number_mismatch_rate: number | null;
  auto_fetch_rate: number | null;
  stages: { key: string; label: string; mean_sec: number | null; runs: number }[];
  by_day: { date: string; n: number }[];
  recent: {
    id: string;
    ts: number;
    question: string;
    rating: number | null;
    error: string | null;
    sec: number;
    citation: CitationCheck["summary"] | null;
    grounded: boolean | null;
  }[];
}

export interface SourceChunk {
  source: string;
  title: string;
  section: string;
  link: string;
  kind: string;
  origin: string;
  rerank_score: number | null;
  snippet: string;
}

export interface Verification {
  grounded: boolean;
  faithfulness: number;
  unsupported: string[];
  // 검증 v2(VERIFY_PROMPT=v2)만 채움: 질문에 대해 근거로 답할 수 있는데 빠뜨린 부분
  completeness?: "full" | "partial" | "none" | "";
  missed?: string[];
}

// inherited(앞뒤 문장의 인용을 이어받음)·summary(인용 없는 짧은 종합 문장)는 경보가 아니다 — 숫자가 틀릴 때만 목록에 나온다
export type SentenceStatus =
  | "supported"
  | "miscited"
  | "weak"
  | "uncited"
  | "bad_ref"
  | "table_row"
  | "contradicted"
  | "inherited"
  | "summary";

export interface SentenceCheck {
  text: string;
  cites: number[];
  status: SentenceStatus;
  support: number | null;
  best: number | null;
  best_score: number | null;
  numbers_missing: string[]; // 인용한 근거에 없는 숫자
  numbers_found_in: number[]; // 그 숫자가 있는 다른 근거 번호
  entail?: number | null; // 함의(NLI) 확률 — 인용한 근거 중 최고
  contradiction?: number | null; // 모순 확률 — 인용한 근거 중 최저
  relation?: boolean; // 근거 사이 관계·비교 단정("(추론)" 표시 없음)
}

// 4b 문장 단위 인용 확인(rag/stages/citation_check.py) — 리랭커로 문장과 인용한 근거를 대조, LLM 없음
export interface CitationCheck {
  threshold: number;
  summary: {
    claims: number;
    cited: number;
    supported: number;
    miscited: number;
    weak: number;
    uncited: number;
    bad_ref: number;
    inherited?: number;
    summary?: number;
    relation?: number; // 관계·비교 단정 문장 수(경보와 별도)
    precision: number | null;
    recall: number | null;
    numeric: number; // 숫자를 대조한 문장 수
    number_mismatch: number; // 숫자가 인용한 근거에 없는 문장 수
  };
  flagged: SentenceCheck[]; // 점수가 기준 미만이거나 숫자가 근거에 없는 문장
  relations?: SentenceCheck[]; // 경보는 아니지만 근거 사이 관계를 단정해 확인을 권하는 문장
}

export interface AutoFetched {
  source: string;
  title: string;
  fulltext: boolean;
  chunks: number;
}

export interface ChatResult {
  question: string;
  standalone_question: string;
  transformed_queries: string[];
  answer: string;
  sources: SourceChunk[];
  n_candidates: number;
  verification: Verification | null;
  regenerated: boolean;
  auto_fetched: AutoFetched[];
  citation_check?: CitationCheck | null;
  timings: Record<string, number>;
  log_id?: string | null; // 서버 질문 기록 id — 👍/👎 피드백에 쓴다(기록이 꺼져 있으면 없음)
}

export type Rating = 1 | -1 | 0;

export interface CompareTableResult {
  markdown: string;
  verification: Verification | null;
  cells_filled: number;
  cells_total: number;
  number_check?: { rows: number; mismatch: SentenceCheck[] }; // 칸 숫자가 단 근거 원문에 있는지
  chunks: { title: string; section: string; source: string }[];
}

export interface SurveyResult {
  topic: string;
  text: string;
  references: string[];
  verification: Verification | null;
  markdown: string;
}

export interface WatchState {
  topics: { topic: string; query: string; last_checked: string | null }[];
  digest: { date: string; markdown: string } | null;
}

export interface PaperRef {
  id: string;
  title: string;
}

export interface ExternalRef {
  title: string;
  arxiv?: string;
  cited_by: number;
}

export interface CitationPaper {
  source: string;
  title: string;
  year: number | null;
  cited_by: number;
  prior: PaperRef[];
  later: PaperRef[];
  n_prior: number;
  n_later: number;
  external?: { prior: ExternalRef[]; later: ExternalRef[] };
  external_error?: string;
}

export interface CitationsResponse {
  papers: CitationPaper[];
  openalex_enabled: boolean;
}

// 서버에 저장한 대화(server/conversations.py)
export interface ConversationInfo {
  id: string;
  title: string;
  created: number;
  updated: number;
  n: number;
}

// 사람 판정 기준 세트(eval/human_set.py, server/routes/label.py)
export type HumanSentenceLabel = "supported" | "unsupported" | "miscited";
export type HumanOverall = "correct" | "partial" | "wrong";

export interface HumanItem {
  id: string;
  kind: string;
  question: string;
  expected_sources: string[];
  answer: string;
  sources: { source: string; title: string; section: string; kind: string; content: string }[];
  llm_verification: { grounded: boolean; faithfulness: number; unsupported: string[] } | null;
  citation_check: { threshold: number; sentences: SentenceCheck[] };
}

export interface HumanLabel {
  sentences: Record<string, HumanSentenceLabel>;
  overall: HumanOverall | null;
  note: string | null;
  annotator?: string; // 판정한 주체 — 사람이면 human(주소에 ?annotator=이름 으로 바꿀 수 있음)
  model_draft?: { sentences: Record<string, HumanSentenceLabel>; overall: HumanOverall | null; note: string | null; annotator?: string };
}

export interface LabelState {
  items: HumanItem[];
  labels: Record<string, HumanLabel>;
  done: number;
  human_done: number; // 사람이 확인한 판정 수(초안 제외)
}
