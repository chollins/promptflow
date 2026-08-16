import { createFileRoute } from "@tanstack/react-router";
import { AppShell, PageHeader } from "@/components/app-shell";
import { Card, Field, Input, Button } from "@/components/ui-kit";
import { useMockUser } from "@/hooks/use-mock-user";
import { ORG } from "@/lib/mock-store";

export const Route = createFileRoute("/settings")({
  component: ProfilePage,
});

function ProfilePage() {
  const user = useMockUser();
  return (
    <AppShell>
      <PageHeader title="Profile" description="Your account information." />

      <div className="flex items-center gap-4 mb-8">
        <div className="h-16 w-16 rounded-full bg-foreground text-background flex items-center justify-center text-xl font-medium">
          {user.firstName[0]}
          {user.lastName[0]}
        </div>
        <div>
          <div className="text-lg font-medium">
            {user.firstName} {user.lastName}
          </div>
          <div className="text-sm text-muted-foreground">
            {user.role === "admin" ? "Administrator" : "Member"} · {ORG.name}
          </div>
        </div>
      </div>

      <Card>
        <form className="space-y-5" onSubmit={(e) => e.preventDefault()}>
          <div className="grid grid-cols-2 gap-4">
            <Field label="First Name">
              <Input defaultValue={user.firstName} />
            </Field>
            <Field label="Last Name">
              <Input defaultValue={user.lastName} />
            </Field>
          </div>
          <Field label="Email">
            <Input defaultValue={user.email} disabled />
          </Field>
          <div className="flex justify-end pt-2">
            <Button>Save changes</Button>
          </div>
        </form>
      </Card>
    </AppShell>
  );
}
