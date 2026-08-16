import { createFileRoute } from "@tanstack/react-router";
import { Workflow } from "lucide-react";
import { AppShell, PageHeader } from "@/components/app-shell";
import { Card, Badge } from "@/components/ui-kit";
import { FLOWS } from "@/lib/mock-store";

export const Route = createFileRoute("/flows")({
  component: FlowsPage,
});

function FlowsPage() {
  return (
    <AppShell>
      <PageHeader
        title="Flows"
        description="AI workflows available in your organization."
      />
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {FLOWS.map((flow) => (
          <Card key={flow.name} className={flow.enabled ? "" : "opacity-60"}>
            <div className="flex items-start justify-between mb-4">
              <div className="h-9 w-9 rounded-lg border border-border flex items-center justify-center">
                <Workflow className="h-4 w-4" />
              </div>
              <Badge tone={flow.enabled ? "neutral" : "outline"}>
                {flow.enabled ? "Enabled" : "Disabled"}
              </Badge>
            </div>
            <div className="font-medium">{flow.name}</div>
            <div className="text-xs text-muted-foreground mt-1 leading-relaxed">
              {flow.description}
            </div>
          </Card>
        ))}
      </div>
    </AppShell>
  );
}
