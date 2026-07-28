import { createFileRoute } from "@tanstack/react-router";
import { Building2 } from "lucide-react";
import { AppShell, PageHeader } from "@/components/app-shell";
import { Card } from "@/components/ui-kit";

export const Route = createFileRoute("/admin/organizations")({
  component: OrganizationsAdmin,
});

function OrganizationsAdmin() {
  return (
    <AppShell>
      <PageHeader
        title="Organizations"
        description="Organizations will be loaded from the backend."
      />
      <Card className="p-5">
        <div className="flex items-start gap-3">
          <div className="h-9 w-9 rounded-lg border border-border flex items-center justify-center shrink-0">
            <Building2 className="h-4 w-4" />
          </div>
          <div>
        <div className="font-medium">Organizations pending</div>
            <p className="mt-2 text-sm text-muted-foreground">
              This page is ready for API integration once the backend exposes organization records.
            </p>
          </div>
        </div>
      </Card>
    </AppShell>
  );
}
