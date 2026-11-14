// 파이프라인 설정. 질문을 보낼 때 함께 보내 그 질문에만 적용된다(서버 전역 설정은 바뀌지 않음).
import type { AppConfig, ChatOptions, QueryMode } from "../../api/types";

interface Props {
  config: AppConfig | null;
  options: ChatOptions;
  onChange: (o: ChatOptions) => void;
}

const MODES: { value: QueryMode; label: string }[] = [
  { value: "multi_query", label: "multi_query — 여러 각도" },
  { value: "hyde", label: "hyde — 가상 답변" },
  { value: "decompose", label: "decompose — 하위 질문 분해" },
  { value: "none", label: "none — 변환 안 함" },
];

export function SettingsPanel({ config, options, onChange }: Props) {
  const set = <K extends keyof ChatOptions>(k: K, v: ChatOptions[K]) => onChange({ ...options, [k]: v });
  const local = config?.provider === "local";
  return (
    <div className="settings">
      <label className="field">
        <span>1단계 · 질의 변환</span>
        <select value={options.query_transform_mode} onChange={(e) => set("query_transform_mode", e.target.value as QueryMode)}>
          {MODES.map((m) => (
            <option key={m.value} value={m.value}>
              {m.label}
            </option>
          ))}
        </select>
      </label>
      <label className="check">
        <input type="checkbox" checked={options.use_hybrid} onChange={(e) => set("use_hybrid", e.target.checked)} />
        2단계 · 하이브리드 검색(BM25+벡터)
      </label>
      <label className="field">
        <span>2단계 · 근거 수(top-k): {options.top_k_rerank}</span>
        <input
          type="range"
          min={1}
          max={10}
          value={options.top_k_rerank}
          onChange={(e) => set("top_k_rerank", Number(e.target.value))}
        />
      </label>
      <label className="check">
        <input type="checkbox" checked={options.enable_verification} onChange={(e) => set("enable_verification", e.target.checked)} />
        4단계 · 검증
      </label>
      <label className="check" title="DB 에 맞는 근거가 없으면 arXiv 에서 관련 논문을 찾아 전문을 추가하고 다시 검색합니다(질문당 최대 3편, 수십 초 추가).">
        <input type="checkbox" checked={options.auto_fetch} onChange={(e) => set("auto_fetch", e.target.checked)} />
        📥 근거 부족 시 논문 자동 수집
      </label>
      <hr />
      {local ? (
        <>
          <label className="check" title="켜면 답하기 전에 추론해 비교·종합에 유리하지만 느려집니다(질문당 off 약 1분, on 약 3~4분).">
            <input
              type="checkbox"
              checked={options.local_thinking === "on"}
              onChange={(e) => set("local_thinking", e.target.checked ? "on" : "off")}
            />
            🧠 Thinking 모드 (생성·검증)
          </label>
          <label className="check" title="질의 변환은 쉬운 작업이라 보통 꺼 둡니다(켜면 질문마다 수십 초 추가).">
            <input
              type="checkbox"
              checked={options.local_thinking_light === "on"}
              onChange={(e) => set("local_thinking_light", e.target.checked ? "on" : "off")}
            />
            질의 변환에도 Thinking
          </label>
          <p className="muted small">
            ⏱️ 현재: {options.local_thinking === "on" ? "생각하며 답함 — 질문당 1~4분 (생각 과정이 실시간으로 보입니다)" : "빠른 답변 — 질문당 약 1분"}
          </p>
        </>
      ) : (
        <p className="muted small">🧠 Thinking 토글은 로컬 모델(LLM_PROVIDER=local) 전용입니다.</p>
      )}
    </div>
  );
}
