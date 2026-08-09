import { useMemo, useState } from "react";
import { Check, Copy } from "lucide-react";
import { Button } from "@/components/ui-kit";

export function formatDebugValue(value: unknown): string {
  if (value === undefined) return "undefined";
  if (value === null) return "null";
  if (typeof value === "string") {
    try {
      const parsed = JSON.parse(value);
      if (parsed && typeof parsed === "object") {
        return JSON.stringify(parsed, null, 2);
      }
    } catch {
      return value;
    }
    return value;
  }
  if (typeof value === "object") {
    try {
      return JSON.stringify(value, null, 2);
    } catch {
      return String(value);
    }
  }
  return String(value);
}

export function DebugJsonViewer({
  value,
  className = "",
}: {
  value: unknown;
  className?: string;
}) {
  const [copied, setCopied] = useState(false);
  const text = useMemo(() => formatDebugValue(value), [value]);

  async function copyText() {
    try {
      await navigator.clipboard?.writeText(text);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1200);
    } catch {
      setCopied(false);
    }
  }

  return (
    <div className={`overflow-hidden rounded-lg border border-border bg-background ${className}`.trim()}>
      <div className="flex items-center justify-end border-b border-border px-3 py-2">
        <Button variant="ghost" size="sm" onClick={() => void copyText()}>
          {copied ? <Check className="h-4 w-4" /> : <Copy className="h-4 w-4" />}
          {copied ? "Copied" : "Copy"}
        </Button>
      </div>
      <pre className="max-h-[28rem] overflow-auto whitespace-pre-wrap break-words p-4 font-mono text-xs leading-6 text-foreground">
        {text}
      </pre>
    </div>
  );
}
