import { createFileRoute } from "@tanstack/react-router";
import { AppShell, PageHeader } from "@/components/app-shell";
import { Card } from "@/components/ui-kit";

export const Route = createFileRoute("/users")({
  component: UsersPage,
});

function UsersPage() {
  return (
    <AppShell>
      <PageHeader
        title="Users"
        description="User management will be powered by the backend later."
      />
      <Card className="p-5">
        <div className="font-medium">Users pending</div>
        <p className="mt-2 text-sm text-muted-foreground">
          This screen is temporarily empty until real user records exist in the database.
        </p>
      </Card>
    </AppShell>
  );
}
