import { useState } from "react";

type PromptData = {
  system?: string;
  user?: string;
};

type PromptTabsProps = {
  template: PromptData | null;
  rendered: PromptData | null;
};

export function PromptTabs({
  template,
  rendered,
}: PromptTabsProps) {
  const [activeTab, setActiveTab] = useState<"template" | "rendered">(
    "template",
  );

  const prompt = activeTab === "template" ? template : rendered;

  return (
    <div className="space-y-4">
      {/* Tabs */}
      <div className="flex w-fit gap-1 rounded-lg border border-border bg-muted p-1">
        <button
          type="button"
          onClick={() => setActiveTab("template")}
          className={`rounded-md px-3 py-1.5 text-sm transition-colors ${
            activeTab === "template"
              ? "bg-background font-medium shadow-sm"
              : "text-muted-foreground hover:text-foreground"
          }`}
        >
          Prompt Template
        </button>

        <button
          type="button"
          onClick={() => setActiveTab("rendered")}
          className={`rounded-md px-3 py-1.5 text-sm transition-colors ${
            activeTab === "rendered"
              ? "bg-background font-medium shadow-sm"
              : "text-muted-foreground hover:text-foreground"
          }`}
        >
          Rendered Prompt
        </button>
      </div>

      {/* Prompt Content */}
      {prompt ? (
        <div className="space-y-4">
          <div>
            <div className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
              System
            </div>

            <pre className="mt-1 whitespace-pre-wrap rounded-lg border border-border bg-background p-4 text-sm">
              {prompt.system || "—"}
            </pre>
          </div>

          <div>
            <div className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
              User
            </div>

            <pre className="mt-1 whitespace-pre-wrap rounded-lg border border-border bg-background p-4 text-sm">
              {prompt.user || "—"}
            </pre>
          </div>
        </div>
      ) : (
        <div className="text-sm text-muted-foreground">
          Execute a step to inspect the prompt.
        </div>
      )}
    </div>
  );
}