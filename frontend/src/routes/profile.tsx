import { createFileRoute } from "@tanstack/react-router";
import { AppShell, PageHeader } from "@/components/app-shell";
import { Card } from "@/components/ui-kit";

export const Route = createFileRoute("/profile")({
  component: ProfilePage,
});

function ProfilePage() {
  return (
    <AppShell>
      <PageHeader
        title="Profile"
        description="Profile data will come from the backend once auth is real."
      />
      <Card className="p-5">
        <div className="font-medium">Profile pending</div>
        <p className="mt-2 text-sm text-muted-foreground">
          This page is now a placeholder until user accounts are implemented on the server.
        </p>
      </Card>
    </AppShell>
  );
}
