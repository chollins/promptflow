import { createFileRoute } from "@tanstack/react-router";
import { AppShell, PageHeader } from "@/components/app-shell";
import { Card } from "@/components/ui-kit";

export const Route = createFileRoute("/admin/organizations/$id")({
  component: OrganizationDetailPage,
});

function OrganizationDetailPage() {
  return (
    <AppShell>
      <PageHeader
        title="Organization details"
        description="This page will load real organization data from the backend."
      />
      <Card className="p-5">
        <div className="font-medium">Backend pending</div>
        <p className="mt-2 text-sm text-muted-foreground">
          Organization members, flow access, and metadata will appear here once the API is connected.
        </p>
      </Card>
    </AppShell>
  );
}
