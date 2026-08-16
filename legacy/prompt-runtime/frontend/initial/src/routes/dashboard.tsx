import { createFileRoute, Link } from "@tanstack/react-router";
import { ArrowUpRight, Workflow } from "lucide-react";
import { AppShell, PageHeader } from "@/components/app-shell";
import { Card, Badge } from "@/components/ui-kit";
import { useMockUser } from "@/hooks/use-mock-user";
import { ORG, FLOWS } from "@/lib/mock-store";

export const Route = createFileRoute("/dashboard")({
  component: Dashboard,
});

function Dashboard() {
  const user = useMockUser();
  const enabled = FLOWS.filter((f) => f.enabled);

  return (
    <AppShell>
      <PageHeader
        title={`Welcome, ${user.firstName}`}
        description="Here's what's happening in your workspace today."
      />

      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 mb-10">
        <Card>
          <div className="text-xs text-muted-foreground mb-2">Organization</div>
          <div className="text-lg font-medium">{ORG.name}</div>
        </Card>
        <Card>
          <div className="text-xs text-muted-foreground mb-2">Role</div>
          <div className="text-lg font-medium capitalize">
            {user.role === "admin" ? "Administrator" : "Member"}
          </div>
        </Card>
      </div>

      <div className="flex items-center justify-between mb-4">
        <h2 className="text-sm font-medium text-muted-foreground uppercase tracking-wider">
          Accessible flows
        </h2>
        <Link to="/flows" className="text-xs text-muted-foreground hover:text-foreground inline-flex items-center gap-1">
          View all <ArrowUpRight className="h-3 w-3" />
        </Link>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {enabled.map((flow) => (
          <Card key={flow.name} className="group hover:border-foreground/30 transition-colors cursor-pointer">
            <div className="flex items-start justify-between mb-4">
              <div className="h-9 w-9 rounded-lg border border-border flex items-center justify-center">
                <Workflow className="h-4 w-4" />
              </div>
              <Badge tone="muted">Enabled</Badge>
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
