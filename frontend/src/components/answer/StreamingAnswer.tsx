// 답변 본문: 받은 글을 타자 효과로 한 글자씩 드러낸다(hooks/useTypewriter).
import { useTypewriter } from "../../hooks/useTypewriter";
import { Markdown, type CiteHandler } from "../common/Markdown";

interface Props {
  text: string;
  streaming: boolean; // 아직 글자가 더 올 수 있음
  citeTitles?: string[];
  onCite?: CiteHandler;
}

export function StreamingAnswer({ text, streaming, citeTitles, onCite }: Props) {
  const { text: shown, typing } = useTypewriter(text, streaming);
  return (
    <Markdown text={shown} chars={shown.length} streaming={streaming || typing} citeTitles={citeTitles} onCite={onCite} />
  );
}
