import { createFileRoute, Link } from "@tanstack/react-router";
import { ArrowRight } from "lucide-react";
import { useEffect, useState } from "react";
import { AppShell, PageHeader } from "@/components/app-shell";
import { Button, Card } from "@/components/ui-kit";
import { apiGet } from "@/lib/api";

type DashboardStats = {
  flows: number;
  users: number;
  organizations: number;
  profileName: string | null;
};

export const Route = createFileRoute("/dashboard")({
  component: Dashboard,
});

function Dashboard() {
  const [stats, setStats] = useState<DashboardStats>({
    flows: 0,
    users: 0,
    organizations: 0,
    profileName: null,
  });
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let active = true;
    Promise.all([
      apiGet<{ count: number }>("/flows"),
      apiGet<{ count: number }>("/users"),
      apiGet<{ count: number }>("/organizations"),
      apiGet<{ item: { name: string } | null }>("/profile"),
    ])
      .then(([flows, users, organizations, profile]) => {
        if (!active) return;
        setStats({
          flows: flows.count,
          users: users.count,
          organizations: organizations.count,
          profileName: profile.item?.name ?? null,
        });
      })
      .catch(() => {
        if (!active) return;
        setStats({ flows: 0, users: 0, organizations: 0, profileName: null });
      })
      .finally(() => {
        if (active) setLoading(false);
      });

    return () => {
      active = false;
    };
  }, []);

  return (
    <AppShell>
      <PageHeader
        title="Dashboard"
        description="Workspace data is now pulled from the backend API."
      />

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <Card className="p-5">
          <div className="text-xs text-muted-foreground mb-2">Status</div>
          <div className="text-lg font-medium">{loading ? "Loading..." : "Connected"}</div>
          <p className="mt-2 text-sm text-muted-foreground">
            {stats.profileName
              ? `Signed in as ${stats.profileName}.`
              : "No profile data available yet."}
          </p>
        </Card>

        <Card className="p-5">
          <div className="text-xs text-muted-foreground mb-2">Next step</div>
          <div className="text-lg font-medium">
            {stats.flows} flows, {stats.users} users, {stats.organizations} organizations
          </div>
          <div className="mt-4">
            <Link to="/flows">
              <Button variant="secondary">
                View flows
                <ArrowRight className="h-4 w-4" />
              </Button>
            </Link>
          </div>
        </Card>
      </div>
    </AppShell>
  );
}
