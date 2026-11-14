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
}

export type SentenceStatus = "supported" | "miscited" | "weak" | "uncited" | "bad_ref";

export interface SentenceCheck {
  text: string;
  cites: number[];
  status: SentenceStatus;
  support: number | null;
  best: number | null;
  best_score: number | null;
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
    precision: number | null;
    recall: number | null;
  };
  flagged: SentenceCheck[]; // supported 가 아닌 문장만
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
