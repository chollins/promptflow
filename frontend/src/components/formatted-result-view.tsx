import { useMemo, useState } from "react";
import ReactMarkdown from "react-markdown";
import { Check, Copy, Code2, Sparkles, Image as ImageIcon } from "lucide-react";
import { processLlmResult } from "@/lib/result-formatter";
import { cn } from "@/lib/utils";

interface FormattedResultViewProps {
  content: string;
  className?: string;
  defaultView?: "readable" | "raw";
  showImagePreview?: boolean;
}

export function FormattedResultView({
  content,
  className = "",
  defaultView = "readable",
  showImagePreview = true,
}: FormattedResultViewProps) {
  const [viewMode, setViewMode] = useState<"readable" | "raw">(defaultView);
  const [copied, setCopied] = useState(false);

  const processed = useMemo(() => processLlmResult(content), [content]);

  const handleCopy = async () => {
    try {
      const textToCopy =
        viewMode === "raw"
          ? processed.rawJsonText || content
          : processed.readableMarkdown;
      await navigator.clipboard.writeText(textToCopy);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      // ignore clipboard error
    }
  };

  return (
    <div className={cn("space-y-3", className)}>
      {/* Header controls if JSON or structured output is detected */}
      {processed.hasJson && (
        <div className="flex items-center justify-between pb-2 border-b border-border/60">
          <div className="flex items-center gap-1.5 p-0.5 rounded-lg bg-muted/60 border border-border text-xs">
            <button
              type="button"
              onClick={() => setViewMode("readable")}
              className={cn(
                "flex items-center gap-1.5 px-2.5 py-1 rounded-md font-medium transition-colors",
                viewMode === "readable"
                  ? "bg-background text-foreground shadow-sm"
                  : "text-muted-foreground hover:text-foreground"
              )}
            >
              Readable View
            </button>
            <button
              type="button"
              onClick={() => setViewMode("raw")}
              className={cn(
                "flex items-center gap-1.5 px-2.5 py-1 rounded-md font-medium transition-colors",
                viewMode === "raw"
                  ? "bg-background text-foreground shadow-sm"
                  : "text-muted-foreground hover:text-foreground"
              )}
            >
              <Code2 className="h-3.5 w-3.5" />
              Raw JSON
            </button>
          </div>

          <button
            type="button"
            onClick={handleCopy}
            className="flex items-center gap-1 px-2.5 py-1 rounded-md text-xs font-medium text-muted-foreground hover:text-foreground border border-border bg-background hover:bg-muted transition-colors"
            title="Copy output"
          >
            {copied ? (
              <>
                <Check className="h-3.5 w-3.5 text-green-600" />
                <span>Copied!</span>
              </>
            ) : (
              <>
                <Copy className="h-3.5 w-3.5" />
                <span>Copy</span>
              </>
            )}
          </button>
        </div>
      )}

      {/* Main content display */}
      {processed.hasJson && viewMode === "raw" ? (
        <pre className="whitespace-pre-wrap font-mono text-xs bg-muted/40 p-4 rounded-lg overflow-x-auto border border-border text-foreground leading-relaxed">
          <code>{processed.rawJsonText || content}</code>
        </pre>
      ) : (
        <div className="text-sm text-foreground max-w-none leading-relaxed">
          <ReactMarkdown
            components={{
              h1: ({ node, ...props }) => (
                <h1 className="text-xl font-bold mt-4 mb-3 text-foreground" {...props} />
              ),
              h2: ({ node, ...props }) => (
                <h2 className="text-lg font-semibold mt-4 mb-2 text-foreground" {...props} />
              ),
              h3: ({ node, ...props }) => (
                <h3 className="text-base font-semibold mt-3 mb-1.5 text-foreground" {...props} />
              ),
              h4: ({ node, ...props }) => (
                <h4 className="text-sm font-semibold mt-2.5 mb-1 text-foreground" {...props} />
              ),
              p: ({ node, ...props }) => (
                <p className="mb-2.5 leading-relaxed text-foreground" {...props} />
              ),
              ul: ({ node, ...props }) => (
                <ul className="list-disc pl-5 mb-3 space-y-1.5 text-foreground" {...props} />
              ),
              ol: ({ node, ...props }) => (
                <ol className="list-decimal pl-5 mb-3 space-y-1.5 text-foreground" {...props} />
              ),
              li: ({ node, ...props }) => (
                <li className="leading-relaxed" {...props} />
              ),
              strong: ({ node, ...props }) => (
                <strong className="font-semibold text-foreground" {...props} />
              ),
              code: ({ node, inline, ...props }: any) =>
                inline ? (
                  <code
                    className="bg-muted px-1.5 py-0.5 rounded text-xs font-mono text-foreground"
                    {...props}
                  />
                ) : (
                  <pre className="bg-muted/70 p-3 rounded-lg overflow-x-auto text-xs font-mono my-2.5 border border-border">
                    <code {...props} />
                  </pre>
                ),
              blockquote: ({ node, ...props }) => (
                <blockquote
                  className="border-l-4 border-muted-foreground/30 pl-3.5 italic text-muted-foreground my-2"
                  {...props}
                />
              ),
            }}
          >
            {processed.readableMarkdown}
          </ReactMarkdown>
        </div>
      )}

      {/* Generated image preview if detected */}
      {showImagePreview && processed.imageUrl && (
        <div className="rounded-lg border border-border bg-muted/20 p-3 mt-3">
          <div className="flex items-center gap-1.5 text-xs font-medium uppercase tracking-wide text-muted-foreground mb-2">
            <ImageIcon className="h-3.5 w-3.5" />
            Generated Image Preview
          </div>
          <img
            src={processed.imageUrl}
            alt="Generated Output Preview"
            className="max-h-80 rounded-md object-contain border border-border bg-background"
            onError={(e) => {
              (e.target as HTMLElement).style.display = "none";
            }}
          />
        </div>
      )}
    </div>
  );
}
