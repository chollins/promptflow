import { createFileRoute, Link } from "@tanstack/react-router";
import { ArrowRight } from "lucide-react";
import { AppShell, PageHeader } from "@/components/app-shell";
import { Button, Card } from "@/components/ui-kit";

export const Route = createFileRoute("/dashboard")({
  component: Dashboard,
});

function Dashboard() {
  return (
    <AppShell>
      <PageHeader
        title="Dashboard"
        description="Real workspace data will appear here after the backend is connected."
      />

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <Card className="p-5">
          <div className="text-xs text-muted-foreground mb-2">Status</div>
          <div className="text-lg font-medium">Waiting for backend</div>
          <p className="mt-2 text-sm text-muted-foreground">
            This area will later show organization, flow, and user activity.
          </p>
        </Card>

        <Card className="p-5">
          <div className="text-xs text-muted-foreground mb-2">Next step</div>
          <div className="text-lg font-medium">Connect API routes</div>
          <div className="mt-4">
            <Link to="/flows">
              <Button variant="secondary">
                Browse flows
                <ArrowRight className="h-4 w-4" />
              </Button>
            </Link>
          </div>
        </Card>
      </div>
    </AppShell>
  );
}
