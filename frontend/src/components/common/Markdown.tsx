// 마크다운(표·수식 포함) 렌더러. 블록 단위로 memo 해서, 쓰는 중에는 마지막 블록만 다시 해석한다.
import { memo, useMemo, type ReactNode } from "react";
import ReactMarkdown, { type Components } from "react-markdown";
import rehypeKatex from "rehype-katex";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import { linkCitations, splitBlocks } from "../../lib/markdown";

const remarkPlugins = [remarkGfm, remarkMath];
const rehypePlugins = [rehypeKatex];

export type CiteHandler = (n: number) => void;

interface BlockProps {
  text: string;
  citeTitles?: string[];
  onCite?: CiteHandler;
}

const Block = memo(function Block({ text, citeTitles, onCite }: BlockProps) {
  const components = useMemo<Components>(
    () => ({
      a({ href, children }) {
        const m = href?.match(/^#cite-(\d+)$/);
        if (m) {
          const n = Number(m[1]);
          return (
            <button type="button" className="cite" title={citeTitles?.[n - 1] ?? `근거 ${n}`} onClick={() => onCite?.(n)}>
              {n}
            </button>
          );
        }
        return (
          <a href={href} target="_blank" rel="noreferrer">
            {children}
          </a>
        );
      },
      table({ children }) {
        return (
          <div className="table-wrap">
            <table>{children}</table>
          </div>
        );
      },
    }),
    [citeTitles, onCite],
  );
  return (
    <ReactMarkdown remarkPlugins={remarkPlugins} rehypePlugins={rehypePlugins} components={components}>
      {linkCitations(text)}
    </ReactMarkdown>
  );
});

interface Props {
  text: string;
  streaming?: boolean; // 쓰는 중이면 끝에 커서 표시
  citeTitles?: string[];
  onCite?: CiteHandler;
  className?: string;
  children?: ReactNode;
  chars?: number; // 드러낸 글자 수(data-chars — 화면 시험에서 타자 효과 측정용)
}

export function Markdown({ text, streaming, citeTitles, onCite, className, chars }: Props) {
  const blocks = useMemo(() => splitBlocks(text), [text]);
  return (
    <div className={`md${streaming ? " streaming" : ""}${className ? ` ${className}` : ""}`} data-chars={chars}>
      {blocks.map((b, i) => (
        <Block key={i} text={b} citeTitles={citeTitles} onCite={onCite} />
      ))}
      {streaming && blocks.length === 0 && <p className="caret-only" />}
    </div>
  );
}
