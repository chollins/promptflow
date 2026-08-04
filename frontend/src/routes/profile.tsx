import { createFileRoute } from "@tanstack/react-router";
import { useEffect, useState } from "react";
import { AppShell, PageHeader } from "@/components/app-shell";
import { Card } from "@/components/ui-kit";
import { apiGet } from "@/lib/api";

type Profile = {
  id: string;
  name: string;
  email: string;
  role: string | null;
  organization_name: string | null;
  is_active: boolean;
};

export const Route = createFileRoute("/profile")({
  component: ProfilePage,
});

function ProfilePage() {
  const [item, setItem] = useState<Profile | null>(null);

  useEffect(() => {
    apiGet<{ item: Profile | null }>("/profile")
      .then((data) => setItem(data.item))
      .catch(() => setItem(null));
  }, []);

  return (
    <AppShell>
      <PageHeader
        title="Profile"
        description="Profile data is loaded from the backend."
      />
      <Card className="p-5">
        {item ? (
          <div className="space-y-2">
            <div className="font-medium">{item.name}</div>
            <p className="text-sm text-muted-foreground">{item.email}</p>
            <p className="text-sm text-muted-foreground">
              {item.organization_name || "No organization"} · {item.role || "No role"}
            </p>
          </div>
        ) : (
          <div className="font-medium">No profile data yet</div>
        )}
      </Card>
    </AppShell>
  );
}
