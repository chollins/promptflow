import { createFileRoute } from "@tanstack/react-router";
import { ShieldCheck } from "lucide-react";
import { AppShell, PageHeader } from "@/components/app-shell";
import { Card } from "@/components/ui-kit";

export const Route = createFileRoute("/admin/manage-flows")({
  component: ManageFlowsPage,
});

function ManageFlowsPage() {
  return (
    <AppShell>
      <PageHeader
        title="Manage Flow Access"
        description="Backend-managed flow access will appear here."
      />
      <Card className="p-5">
        <div className="flex items-start gap-3">
          <div className="h-9 w-9 rounded-lg border border-border flex items-center justify-center shrink-0">
            <ShieldCheck className="h-4 w-4" />
          </div>
          <div>
        <div className="font-medium">Access management pending</div>
            <p className="mt-2 text-sm text-muted-foreground">
              This screen will be wired to the backend once organization-flow assignments exist.
            </p>
          </div>
        </div>
      </Card>
    </AppShell>
  );
}
