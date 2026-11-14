// 펼치기/접기 상자. 접혀 있으면 내용을 그리지 않는다(무거운 패널을 필요할 때만).
import { useState, type ReactNode } from "react";

interface Props {
  title: ReactNode;
  defaultOpen?: boolean;
  open?: boolean;
  onToggle?: (open: boolean) => void;
  children: ReactNode;
  className?: string;
}

export function Collapsible({ title, defaultOpen = false, open, onToggle, children, className }: Props) {
  const [inner, setInner] = useState(defaultOpen);
  const isOpen = open ?? inner;
  const toggle = () => {
    setInner(!isOpen);
    onToggle?.(!isOpen);
  };
  return (
    <section className={`collapsible${isOpen ? " open" : ""}${className ? ` ${className}` : ""}`}>
      <button type="button" className="collapsible-head" onClick={toggle} aria-expanded={isOpen}>
        <span className="chev" aria-hidden>
          ▸
        </span>
        <span className="collapsible-title">{title}</span>
      </button>
      {isOpen && <div className="collapsible-body">{children}</div>}
    </section>
  );
}
