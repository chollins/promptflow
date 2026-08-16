import type { ReactNode } from "react";

interface Props {
  text: string;
  className?: string;
}

function stripRawMarkers(value: string) {
  return value
    .replace(/\*\*\*([^*]+)\*\*\*/g, "$1")
    .replace(/\*\*([^*]+)\*\*/g, "$1")
    .replace(/\*([^*]+)\*/g, "$1")
    .replace(/`([^`]+)`/g, "$1")
    .replace(/\*{2,}/g, "")
    .trim();
}

function renderInlineMarkdown(text: string) {
  const parts: ReactNode[] = [];
  let remaining = text;
  let key = 0;

  const tokenRegex = /(\*\*\*[^*]+\*\*\*|\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`)/;

  while (remaining.length > 0) {
    const match = remaining.match(tokenRegex);
    if (!match || match.index == null) {
      const cleaned = stripRawMarkers(remaining);
      if (cleaned) {
        parts.push(cleaned);
      }
      break;
    }

    if (match.index > 0) {
      const plain = stripRawMarkers(remaining.slice(0, match.index));
      if (plain) {
        parts.push(plain);
      }
    }

    const token = match[0];
    const inner = token.replace(/^\*{1,3}|^`|`$/g, "").trim();

    if (!inner) {
      remaining = remaining.slice(match.index + token.length);
      continue;
    }

    if (token.startsWith("***")) {
      parts.push(
        <strong key={`strong-${key += 1}`}>
          <em>{inner}</em>
        </strong>,
      );
    } else if (token.startsWith("**")) {
      parts.push(<strong key={`strong-${key += 1}`}>{inner}</strong>);
    } else if (token.startsWith("*")) {
      parts.push(<em key={`em-${key += 1}`}>{inner}</em>);
    } else {
      parts.push(<code key={`code-${key += 1}`}>{inner}</code>);
    }

    remaining = remaining.slice(match.index + token.length);
  }

  return parts;
}

export default function MarkdownPreview({ text, className = "" }: Props) {
  const lines = text.split(/\r?\n/);
  const blocks: ReactNode[] = [];
  let listItems: ReactNode[] = [];

  const flushList = () => {
    if (listItems.length === 0) {
      return;
    }
    blocks.push(
      <ul key={`ul-${blocks.length}`} className="list-disc pl-5 space-y-1">
        {listItems}
      </ul>,
    );
    listItems = [];
  };

  lines.forEach((line, index) => {
    const trimmed = line.trim();
    if (!trimmed) {
      flushList();
      return;
    }

    if (/^#{1,6}\s+/.test(trimmed)) {
      flushList();
      const content = stripRawMarkers(trimmed.replace(/^#{1,6}\s+/, ""));
      blocks.push(
        <h3 key={`h-${index}`} className="mb-2 mt-4 text-base font-semibold text-gray-900">
          {renderInlineMarkdown(content)}
        </h3>,
      );
      return;
    }

    if (trimmed.startsWith("- ")) {
      listItems.push(
        <li key={`li-${index}`}>{renderInlineMarkdown(stripRawMarkers(trimmed.slice(2)))}</li>,
      );
      return;
    }

    if (/^\d+\.\s+/.test(trimmed)) {
      flushList();
      blocks.push(
        <p key={`ol-${index}`} className="mb-3 last:mb-0">
          {renderInlineMarkdown(stripRawMarkers(trimmed.replace(/^\d+\.\s+/, "")))}
        </p>,
      );
      return;
    }

    flushList();
    blocks.push(
      <p key={`p-${index}`} className="mb-3 last:mb-0">
        {renderInlineMarkdown(stripRawMarkers(trimmed))}
      </p>,
    );
  });

  flushList();

  return <div className={className}>{blocks}</div>;
}
