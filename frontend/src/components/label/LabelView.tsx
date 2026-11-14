// 사람 판정 화면: 기준 세트(eval/human_set.py)의 답변을 문장마다 근거 있음/없음/번호 틀림, 답변 전체 맞음/일부/틀림으로 판정.
// 자동 판정(LLM 검증·문장 단위 인용 확인)은 사람 판단이 끌려가지 않게 접어 두고, 판정을 저장한 뒤에 펼쳐 볼 수 있다.
// 일치율 계산: python eval/human_agreement.py
import { useCallback, useEffect, useMemo, useState } from "react";
import { api } from "../../api/client";
import type { HumanItem, HumanLabel, HumanOverall, HumanSentenceLabel, LabelState } from "../../api/types";
import { Collapsible } from "../common/Collapsible";
import { Markdown } from "../common/Markdown";

const SENT: { v: HumanSentenceLabel; label: string }[] = [
  { v: "supported", label: "근거 있음" },
  { v: "unsupported", label: "근거 없음" },
  { v: "miscited", label: "번호 틀림" },
];
const OVERALL: { v: HumanOverall; label: string }[] = [
  { v: "correct", label: "맞음" },
  { v: "partial", label: "일부 틀림" },
  { v: "wrong", label: "틀림" },
];
const KIND: Record<string, string> = { single: "단일 논문", multi_hop: "다중 홉", table: "표", figure: "그림", equation: "수식" };
const EMPTY: HumanLabel = { sentences: {}, overall: null, note: null };
// 판정한 주체: 기본 human. 다른 모델이 판정하면 주소에 ?annotator=모델이름 — 결과를 사람 판정과 구분해 읽는다
const ANNOTATOR = new URLSearchParams(window.location.search).get("annotator") || "human";
const isHuman = (l?: HumanLabel) => !!l?.overall && (l.annotator ?? "human") === "human";

export function LabelView() {
  const [state, setState] = useState<LabelState | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [idx, setIdx] = useState(0);
  const [draft, setDraft] = useState<HumanLabel>(EMPTY);
  const [saving, setSaving] = useState(false);
  const [openSrc, setOpenSrc] = useState<number | null>(null);

  useEffect(() => {
    api
      .labelItems()
      .then((s) => {
        setState(s);
        // 사람이 아직 확인하지 않은 첫 문항부터(모델 초안만 있는 문항 포함)
        const first = s.items.findIndex((it) => (ANNOTATOR === "human" ? !isHuman(s.labels[it.id]) : !s.labels[it.id]?.overall));
        setIdx(first < 0 ? 0 : first);
      })
      .catch((e) => setError((e as Error).message));
  }, []);

  const item: HumanItem | undefined = state?.items[idx];
  useEffect(() => {
    if (item && state) {
      const l = state.labels[item.id];
      // 모델 초안은 판정만 미리 채우고 메모는 비운다(저장하면 사람의 판정·메모가 된다)
      setDraft(l ? (isHuman(l) || ANNOTATOR !== "human" ? l : { ...l, note: null }) : EMPTY);
    }
    setOpenSrc(null);
  }, [item, state]);

  const sentences = item?.citation_check.sentences ?? [];
  const complete = !!draft.overall && sentences.every((_, i) => draft.sentences[String(i)]);
  const current = item ? state?.labels[item.id] : undefined;
  const saved = ANNOTATOR === "human" ? isHuman(current) : !!current?.overall;
  const draftFrom = current?.overall && !isHuman(current) && ANNOTATOR === "human" ? current : undefined;

  const save = useCallback(
    async (next: boolean) => {
      if (!item || !state) return;
      setSaving(true);
      setError(null);
      try {
        const r = await api.saveLabel(item.id, { ...draft, annotator: ANNOTATOR });
        setState({ ...state, labels: { ...state.labels, [item.id]: { ...draft, annotator: ANNOTATOR } }, done: r.done, human_done: r.human_done });
        if (next && idx < state.items.length - 1) setIdx(idx + 1);
      } catch (e) {
        setError((e as Error).message);
      } finally {
        setSaving(false);
      }
    },
    [item, state, draft, idx],
  );

  const citeTitles = useMemo(() => item?.sources.map((s) => s.title || s.source), [item]);

  if (error && !state) return <div className="label-view"><p className="error">⚠️ {error}</p></div>;
  if (!state) return <div className="label-view"><p className="muted">불러오는 중…</p></div>;
  if (!state.items.length)
    return (
      <div className="label-view">
        <h2>사람 판정</h2>
        <p className="muted">
          기준 세트가 아직 없습니다. <code>python eval/human_set.py --build</code> 로 만든 뒤 새로고침하세요.
        </p>
      </div>
    );
  if (!item) return null;

  return (
    <div className="label-view">
      <div className="label-head">
        <h2>사람 판정</h2>
        <span className="pill" title="사람이 검토해 저장한 판정 수">
          사람 확인 {state.human_done}/{state.items.length}
        </span>
        {state.done > state.human_done && (
          <span className="pill" title="다른 모델이 먼저 채운 초안 — 검토해 저장하면 사람 판정이 됩니다">
            초안 {state.done - state.human_done}
          </span>
        )}
        <select value={idx} onChange={(e) => setIdx(Number(e.target.value))} aria-label="문항 고르기">
          {state.items.map((it, i) => (
            <option key={it.id} value={i}>
              {isHuman(state.labels[it.id]) ? "✓ " : state.labels[it.id]?.overall ? "◇ " : "· "}
              {i + 1}. [{KIND[it.kind] ?? it.kind}] {it.question.slice(0, 40)}
            </option>
          ))}
        </select>
      </div>
      <p className="muted small">
        답변의 문장마다 아래 근거 원문을 보고 판정해 주세요. "번호 틀림"은 내용은 근거에 있지만 단 번호가 다른 경우입니다.
      </p>

      <section className="label-card">
        {draftFrom && (
          <div className="draft-banner">
            <b>🤖 {draftFrom.annotator} 초안이 채워져 있습니다.</b> 근거 원문과 대조해 동의하면 그대로, 다르면 고쳐서 저장하세요 —
            저장하면 <b>사람 판정</b>으로 기록되고 초안은 비교용으로 따로 남습니다.
            {draftFrom.note && <div className="muted small">초안 메모: {draftFrom.note}</div>}
          </div>
        )}
        <div className="muted small">
          {idx + 1}/{state.items.length} · {KIND[item.kind] ?? item.kind} · 정답 논문 {item.expected_sources.join(", ")}
        </div>
        <h3 className="label-q">{item.question}</h3>
        <Collapsible title="답변 전체 보기" defaultOpen className="label-answer">
          <Markdown text={item.answer} citeTitles={citeTitles} onCite={(n) => setOpenSrc(n - 1)} />
        </Collapsible>

        <h4>문장별 판정 ({Object.keys(draft.sentences).length}/{sentences.length})</h4>
        {sentences.length === 0 && <p className="muted small">확인할 주장 문장이 없습니다(예: '근거에 없다'는 답). 답변 전체만 판정하세요.</p>}
        <ol className="label-sents">
          {sentences.map((s, i) => (
            <li key={i}>
              <div className="label-sent-text">
                {s.text}{" "}
                {s.cites.map((n) => (
                  <button key={n} type="button" className="cite" onClick={() => setOpenSrc(n - 1)} title="근거 원문 보기">
                    {n}
                  </button>
                ))}
              </div>
              <div className="seg">
                {SENT.map((o) => (
                  <button
                    key={o.v}
                    type="button"
                    className={draft.sentences[String(i)] === o.v ? "on" : ""}
                    onClick={() => setDraft({ ...draft, sentences: { ...draft.sentences, [String(i)]: o.v } })}
                  >
                    {o.label}
                  </button>
                ))}
              </div>
            </li>
          ))}
        </ol>

        <h4>근거 원문</h4>
        <div className="label-sources">
          {item.sources.map((s, i) => (
            <Collapsible
              key={i}
              title={`[${i + 1}] ${s.title || s.source} · ${s.section || s.kind || ""}`}
              open={openSrc === i}
              onToggle={(o) => setOpenSrc(o ? i : null)}
            >
              <pre className="label-src-text">{s.content}</pre>
            </Collapsible>
          ))}
        </div>

        <h4>답변 전체</h4>
        <div className="seg">
          {OVERALL.map((o) => (
            <button key={o.v} type="button" className={draft.overall === o.v ? "on" : ""} onClick={() => setDraft({ ...draft, overall: o.v })}>
              {o.label}
            </button>
          ))}
        </div>
        <textarea
          className="label-note"
          placeholder="메모(선택) — 예: 두 번째 논문 내용이 빠짐"
          value={draft.note ?? ""}
          maxLength={2000}
          onChange={(e) => setDraft({ ...draft, note: e.target.value || null })}
        />

        <div className="label-actions">
          <button type="button" className="btn ghost" disabled={idx === 0} onClick={() => setIdx(idx - 1)}>
            ← 이전
          </button>
          <button type="button" className="btn" disabled={!complete || saving} onClick={() => save(true)}>
            {saving ? "저장 중…" : "저장하고 다음 →"}
          </button>
          {!complete && <span className="muted small">모든 문장과 답변 전체를 판정하면 저장할 수 있어요</span>}
          {error && <span className="error small">{error}</span>}
        </div>

        {saved && (
          <Collapsible title="자동 판정 보기 (내 판정과 비교)" className="label-auto">
            <ul>
              <li>
                LLM 검증: {item.llm_verification ? `${item.llm_verification.grounded ? "근거 충실" : "근거 부족"} · faithfulness ${item.llm_verification.faithfulness.toFixed(2)}` : "없음"}
              </li>
              {sentences.map((s, i) => (
                <li key={i}>
                  문장 {i + 1}: 인용 확인 <b>{s.status}</b>
                  {s.numbers_missing.length > 0 && ` · 숫자 ${s.numbers_missing.join(", ")} 근거에 없음`} / 내 판정 {draft.sentences[String(i)] ?? "-"}
                </li>
              ))}
            </ul>
          </Collapsible>
        )}
      </section>
    </div>
  );
}
