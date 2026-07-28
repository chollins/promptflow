import { createFileRoute } from "@tanstack/react-router";
import { Workflow } from "lucide-react";
import { AppShell, PageHeader } from "@/components/app-shell";
import { Card } from "@/components/ui-kit";

export const Route = createFileRoute("/admin/flows")({
  component: FlowsCatalog,
});

function FlowsCatalog() {
  return (
    <AppShell>
      <PageHeader
        title="Flows catalog"
        description="The backend will provide the flow catalog here."
      />
      <Card className="p-5">
        <div className="flex items-start gap-3">
          <div className="h-9 w-9 rounded-lg border border-border flex items-center justify-center shrink-0">
            <Workflow className="h-4 w-4" />
          </div>
          <div>
        <div className="font-medium">Flow catalog pending</div>
            <p className="mt-2 text-sm text-muted-foreground">
              This section will be populated from the backend once the flow API is ready.
            </p>
          </div>
        </div>
      </Card>
    </AppShell>
  );
}
