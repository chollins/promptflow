import { createFileRoute } from "@tanstack/react-router";
import { AppShell, PageHeader } from "@/components/app-shell";
import { Card } from "@/components/ui-kit";

export const Route = createFileRoute("/organization")({
  component: OrganizationPage,
});

function OrganizationPage() {
  return (
    <AppShell>
      <PageHeader
        title="Organization"
        description="Organization settings will be loaded from the backend."
      />
      <Card className="p-5">
        <div className="font-medium">No backend data yet</div>
        <p className="mt-2 text-sm text-muted-foreground">
          This page is preserved for future organization management features.
        </p>
      </Card>
    </AppShell>
  );
}
