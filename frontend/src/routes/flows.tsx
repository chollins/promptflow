import { createFileRoute } from "@tanstack/react-router";
import { AppShell, PageHeader } from "@/components/app-shell";
import { Card } from "@/components/ui-kit";

export const Route = createFileRoute("/flows")({
  component: FlowsPage,
});

function FlowsPage() {
  return (
    <AppShell>
      <PageHeader
        title="Flows"
        description="Flow data will load from the backend once the API is connected."
      />
      <div className="grid grid-cols-1 gap-4">
        <Card className="p-5">
          <div className="font-medium">Flows pending</div>
          <p className="mt-2 text-sm text-muted-foreground">
            This page is intentionally empty until the backend serves real flow records.
          </p>
        </Card>
      </div>
    </AppShell>
  );
}
