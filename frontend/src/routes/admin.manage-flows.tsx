import { createFileRoute } from "@tanstack/react-router";
import { useEffect, useState } from "react";
import { ShieldCheck } from "lucide-react";
import { AppShell, PageHeader } from "@/components/app-shell";
import { Card } from "@/components/ui-kit";
import { apiGet } from "@/lib/api";

type AccessItem = {
  organization_id: string;
  organization_name: string | null;
  flow_id: string;
  flow_name: string | null;
};

export const Route = createFileRoute("/admin/manage-flows")({
  component: ManageFlowsPage,
});

function ManageFlowsPage() {
  const [items, setItems] = useState<AccessItem[]>([]);

  useEffect(() => {
    apiGet<{ items: AccessItem[] }>("/admin/manage-flows")
      .then((data) => setItems(data.items))
      .catch(() => setItems([]));
  }, []);

  return (
    <AppShell>
      <PageHeader
        title="Manage Flow Access"
        description="Organization-flow assignments are loaded from the backend."
      />
      <div className="space-y-4">
        {items.map((item) => (
          <Card key={`${item.organization_id}-${item.flow_id}`} className="p-5">
            <div className="flex items-start gap-3">
              <div className="h-9 w-9 rounded-lg border border-border flex items-center justify-center shrink-0">
                <ShieldCheck className="h-4 w-4" />
              </div>
              <div>
                <div className="font-medium">
                  {item.organization_name || item.organization_id}
                </div>
                <p className="mt-2 text-sm text-muted-foreground">
                  Access to {item.flow_name || item.flow_id}
                </p>
              </div>
            </div>
          </Card>
        ))}
        {items.length === 0 && (
          <Card className="p-5">
            <div className="font-medium">No assignments yet</div>
            <p className="mt-2 text-sm text-muted-foreground">
              Assign flows to organizations in the backend to manage access here.
            </p>
          </Card>
        )}
      </div>
    </AppShell>
  );
}
