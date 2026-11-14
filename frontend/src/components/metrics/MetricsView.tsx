// 사용 지표: 질문 기록(storage/query_log.db)에서 계산한 숫자 타일 + 단계별 평균 시간(막대 한 색) + 날짜별 질문 수 + 최근 질문.
// 글자는 잉크 색만, 막대는 한 가지 색(크기를 나타냄)이고 값은 옆에 숫자로 함께 쓴다.
import { useCallback, useEffect, useState } from "react";
import { api } from "../../api/client";
import type { Metrics } from "../../api/types";

const pct = (v: number | null) => (v == null ? "—" : `${Math.round(v * 100)}%`);
const sec = (v: number | null) => (v == null ? "—" : `${v.toFixed(1)}초`);

function Tile({ label, value, hint }: { label: string; value: string; hint?: string }) {
  return (
    <div className="tile" title={hint}>
      <div className="tile-label">{label}</div>
      <div className="tile-value">{value}</div>
    </div>
  );
}

export function MetricsView() {
  const [m, setM] = useState<Metrics | null>(null);
  const [error, setError] = useState<string | null>(null);
  const reload = useCallback(() => {
    api
      .metrics()
      .then(setM)
      .catch((e) => setError((e as Error).message));
  }, []);
  useEffect(reload, [reload]);

  if (error) return <div className="metrics-view"><p className="error">⚠️ {error}</p></div>;
  if (!m) return <div className="metrics-view"><p className="muted">불러오는 중…</p></div>;
  if (!m.enabled)
    return (
      <div className="metrics-view">
        <h2>사용 지표</h2>
        <p className="muted">질문 기록이 꺼져 있습니다(QUERY_LOG=0).</p>
      </div>
    );

  const maxStage = Math.max(...m.stages.map((s) => s.mean_sec ?? 0), 0.001);
  const maxDay = Math.max(...m.by_day.map((d) => d.n), 1);
  return (
    <div className="metrics-view">
      <div className="label-head">
        <h2>사용 지표</h2>
        <span className="muted small">질문 기록에서 계산 · LLM 없음</span>
        <button type="button" className="btn ghost small-btn" onClick={reload}>
          새로고침
        </button>
      </div>
      {m.total === 0 && <p className="muted">아직 기록된 질문이 없습니다. 대화 탭에서 질문하면 여기에 쌓입니다.</p>}

      <div className="tiles">
        <Tile label="질문" value={`${m.total}`} hint="기록된 질문 수(실패 포함)" />
        <Tile label="👍 / 👎" value={`${m.up} / ${m.down}`} hint="👎 질문은 scripts/export_feedback.py 로 평가셋 후보가 됩니다" />
        <Tile label="평균 응답 시간" value={sec(m.mean_total_sec)} hint="단계별 시간의 합(성공한 질문)" />
        <Tile label="인용 정확도" value={pct(m.citation_precision)} hint="문장 단위 인용 확인: 인용 단 문장 중 근거가 뒷받침하는 비율(답변 평균)" />
        <Tile label="숫자 불일치 답변" value={pct(m.number_mismatch_rate)} hint="숫자가 인용한 근거에 없는 문장이 있는 답변 비율" />
        <Tile label="LLM 검증 통과" value={pct(m.verify_grounded_rate)} hint="4단계 검증의 '근거 충실' 비율(로컬 모델 자기 채점)" />
        <Tile label="자동 수집" value={pct(m.auto_fetch_rate)} hint="논문을 새로 수집한 질문 비율" />
        <Tile label="오류" value={pct(m.error_rate)} hint="실패한 질문 비율" />
      </div>

      <section className="metrics-block">
        <h3>단계별 평균 시간</h3>
        <table className="metrics-table">
          <tbody>
            {m.stages
              .filter((s) => s.runs > 0)
              .map((s) => (
                <tr key={s.key}>
                  <th scope="row">{s.label}</th>
                  <td className="bar-cell">
                    {(s.mean_sec ?? 0) > 0 && <div
                      className="hbar"
                      style={{ width: `${((s.mean_sec ?? 0) / maxStage) * 100}%` }}
                      title={`${s.label}: 평균 ${sec(s.mean_sec)} (${s.runs}회)`}
                    />}
                  </td>
                  <td className="num">{sec(s.mean_sec)}</td>
                  <td className="num muted">{s.runs}회</td>
                </tr>
              ))}
          </tbody>
        </table>
      </section>

      <section className="metrics-block">
        <h3>최근 14일 질문 수</h3>
        <table className="metrics-table">
          <tbody>
            {m.by_day.map((d) => (
              <tr key={d.date}>
                <th scope="row">{d.date.slice(5)}</th>
                <td className="bar-cell">
                  {d.n > 0 && <div className="hbar" style={{ width: `${(d.n / maxDay) * 100}%` }} title={`${d.date}: ${d.n}개`} />}
                </td>
                <td className="num">{d.n}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>

      <section className="metrics-block">
        <h3>최근 질문</h3>
        {m.recent.length === 0 ? (
          <p className="muted small">없음</p>
        ) : (
          <div className="table-wrap">
            <table className="metrics-table recent">
              <thead>
                <tr>
                  <th>시각</th>
                  <th>질문</th>
                  <th>시간</th>
                  <th>인용 확인</th>
                  <th>검증</th>
                  <th>평가</th>
                </tr>
              </thead>
              <tbody>
                {m.recent.map((r) => (
                  <tr key={r.id}>
                    <td className="muted">{new Date(r.ts * 1000).toLocaleString("ko-KR", { month: "numeric", day: "numeric", hour: "2-digit", minute: "2-digit" })}</td>
                    <td>{r.question}</td>
                    <td className="num">{r.error ? "실패" : `${r.sec}초`}</td>
                    <td className="num">{r.citation ? `${r.citation.supported}/${r.citation.claims}` : "—"}</td>
                    <td>{r.grounded == null ? "—" : r.grounded ? "통과" : "주의"}</td>
                    <td>{r.rating === 1 ? "👍" : r.rating === -1 ? "👎" : ""}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>
    </div>
  );
}
