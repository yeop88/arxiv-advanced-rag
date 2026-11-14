// 근거 논문의 인용 관계(코퍼스 안 선행·후속 연구). 펼칠 때 처음 한 번 불러온다.
import { useEffect, useState } from "react";
import { api } from "../../api/client";
import type { CitationsResponse, ExternalRef, PaperRef } from "../../api/types";
import { Collapsible } from "../common/Collapsible";

const arxiv = (id: string) => `https://arxiv.org/abs/${id}`;

function Refs({ items }: { items: PaperRef[] }) {
  if (!items.length) return <>없음</>;
  return (
    <>
      {items.map((p, i) => (
        <span key={p.id}>
          {i > 0 && " · "}
          <a href={arxiv(p.id)} target="_blank" rel="noreferrer">
            {p.title.slice(0, 50)}
          </a>
        </span>
      ))}
    </>
  );
}

function External({ items }: { items: ExternalRef[] }) {
  if (!items.length) return <>없음</>;
  return (
    <>
      {items.map((w, i) => (
        <span key={i}>
          {i > 0 && " · "}
          {w.arxiv ? (
            <a href={arxiv(w.arxiv)} target="_blank" rel="noreferrer">
              {w.title.slice(0, 50)}
            </a>
          ) : (
            w.title.slice(0, 50)
          )}{" "}
          ({w.cited_by.toLocaleString()})
        </span>
      ))}
    </>
  );
}

export function CitationsPanel({ sources, defaultOpen = false }: { sources: string[]; defaultOpen?: boolean }) {
  const [open, setOpen] = useState(defaultOpen);
  const [external, setExternal] = useState(false);
  const [data, setData] = useState<CitationsResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const key = sources.join(",");

  useEffect(() => {
    if (!open) return;
    let alive = true;
    api
      .citations(sources, external)
      .then((d) => alive && setData(d))
      .catch((e) => alive && setError(String(e.message ?? e)));
    return () => {
      alive = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open, external, key]);

  const n = data?.papers.length;
  return (
    <Collapsible title={`🔗 인용 관계 — 근거 논문의 선행·후속 연구${n != null ? ` (${n}편)` : ""}`} open={open} onToggle={setOpen}>
      {error && <p className="error">{error}</p>}
      {!data && !error && <p className="muted">불러오는 중…</p>}
      {data && !data.papers.length && <p className="muted">인용 그래프에 있는 근거 논문이 없습니다.</p>}
      {data?.openalex_enabled && (
        <label className="check">
          <input type="checkbox" checked={external} onChange={(e) => setExternal(e.target.checked)} />
          코퍼스 밖 주요 논문도 보기 (OpenAlex 조회)
        </label>
      )}
      {data?.papers.map((p) => (
        <div key={p.source} className="cite-paper">
          <div>
            <strong>{p.title}</strong>{" "}
            <span className="muted small">
              ({p.year ?? "-"}, 피인용 {p.cited_by.toLocaleString()})
            </span>
          </div>
          <ul>
            <li>
              선행(DB 안 {p.n_prior}편): <Refs items={p.prior} />
            </li>
            <li>
              후속(DB 안 {p.n_later}편): <Refs items={p.later} />
            </li>
            {p.external && (
              <>
                <li className="small">
                  주요 선행: <External items={p.external.prior} />
                </li>
                <li className="small">
                  주요 후속: <External items={p.external.later} />
                </li>
              </>
            )}
            {p.external_error && <li className="small muted">OpenAlex 조회 실패: {p.external_error}</li>}
          </ul>
        </div>
      ))}
    </Collapsible>
  );
}
