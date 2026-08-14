import { createFileRoute, Link, redirect, isRedirect } from "@tanstack/react-router";
import { useEffect, useState } from "react";
import { AppShell, PageHeader } from "@/components/app-shell";
import { Card } from "@/components/ui-kit";
import { apiGet } from "@/lib/api";

type FlowItem = {
  id: string;
  version: string;
  name: string;
  description: string;
  runtime?: { mode: "guided" | "automatic"; default_review_required: boolean };
  steps: Array<{ id: string; name: string; sequence: number }>;
};

export const Route = createFileRoute("/flows")({
  beforeLoad: async () => {
    try {
      await apiGet<{ role: string | null }>("/auth/me");
    } catch (e) {
      if (isRedirect(e)) throw e;
      throw redirect({ to: "/login" });
    }
  },
  component: FlowsPage,
});

function FlowsPage() {
  const [items, setItems] = useState<FlowItem[]>([]);

  useEffect(() => {
    apiGet<{ items: FlowItem[] }>("/flows")
      .then((data) => setItems(data.items))
      .catch(() => setItems([]));
  }, []);

  return (
    <AppShell>
      <PageHeader
        title="Flows"
        description="Prompt flows are loaded from the backend database and run step by step."
      />
      <div className="grid grid-cols-1 gap-4">
        {items.length === 0 ? (
          <Card className="p-5">
            <div className="font-medium">No flows found</div>
            <p className="mt-2 text-sm text-muted-foreground">
              Add flows and flow steps in the backend database to populate this page.
            </p>
          </Card>
        ) : (
          items.map((flow) => (
            <Card key={flow.id} className="p-5">
              <div className="flex items-start justify-between gap-4">
                <div>
                  <div className="font-medium">{flow.name}</div>
                  <p className="mt-2 text-sm text-muted-foreground">
                    {flow.description || "No description."}
                  </p>
                  <p className="mt-2 text-xs text-muted-foreground">
                    {flow.steps.length} steps · {flow.runtime?.mode ?? "guided"}
                  </p>
                </div>
                <Link
                  to="/flows/$flowId"
                  params={{ flowId: flow.id }}
                  className="inline-flex h-10 items-center justify-center rounded-md border border-border bg-background px-4 text-sm font-medium text-foreground transition-colors hover:bg-muted"
                >
                  Open
                </Link>
              </div>
            </Card>
          ))
        )}
      </div>
    </AppShell>
  );
}
