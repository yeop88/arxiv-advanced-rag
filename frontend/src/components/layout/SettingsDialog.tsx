// 설정 대화 상자: 서버 상태 · 검색·생성 설정 · 모델 정보 · 지금 대화 지우기 · 소스 코드(GitHub) 링크.
import type { AppConfig, ChatOptions, ServerStatus } from "../../api/types";
import { SettingsPanel } from "../sidebar/SettingsPanel";
import { StatusPanel } from "../sidebar/StatusPanel";
import { Dialog } from "./Dialog";

const REPO_URL = "https://github.com/yeop88/arxiv-advanced-rag";

interface Props {
  onClose: () => void;
  config: AppConfig | null;
  status: ServerStatus | null;
  offline: boolean;
  options: ChatOptions;
  onOptions: (o: ChatOptions) => void;
  onClear: () => void;
}

export function SettingsDialog({ onClose, config, status, offline, options, onOptions, onClear }: Props) {
  return (
    <Dialog title="설정" onClose={onClose}>
      <section className="dlg-section">
        <h3>서버 상태</h3>
        <StatusPanel status={status} offline={offline} />
      </section>
      <section className="dlg-section">
        <h3>검색 · 답변</h3>
        <p className="muted small">이 브라우저에서 보내는 질문에만 적용됩니다(서버 기본값은 그대로).</p>
        <SettingsPanel config={config} options={options} onChange={onOptions} />
      </section>
      {config && (
        <section className="dlg-section">
          <h3>모델</h3>
          <dl className="kv">
            <dt>LLM</dt>
            <dd>
              {config.provider} · {config.model}
            </dd>
            <dt>임베딩</dt>
            <dd>{config.embed_model}</dd>
            <dt>벡터 DB</dt>
            <dd>{config.vector_backend}</dd>
          </dl>
        </section>
      )}
      <section className="dlg-section danger">
        <h3>대화</h3>
        <button
          type="button"
          className="btn danger"
          onClick={() => {
            if (window.confirm("지금 대화를 지울까요? 서버에 저장된 기록도 지워집니다.")) {
              onClear();
              onClose();
            }
          }}
        >
          지금 대화 지우기
        </button>
      </section>
      <section className="dlg-section">
        <h3>소스 코드</h3>
        <p className="muted small">이 챗봇의 전체 코드를 GitHub 에서 받을 수 있습니다. 릴리즈는 버전별로 고정된 코드와 변경 내역입니다.</p>
        <p>
          <a href={REPO_URL} target="_blank" rel="noopener noreferrer">
            {REPO_URL.replace("https://", "")}
          </a>
          {" · "}
          <a href={`${REPO_URL}/releases/latest`} target="_blank" rel="noopener noreferrer">
            최신 릴리즈
          </a>
          {" · "}
          <a href={`${REPO_URL}/archive/refs/heads/main.zip`} rel="noopener noreferrer">
            ZIP 내려받기
          </a>
        </p>
      </section>
    </Dialog>
  );
}
