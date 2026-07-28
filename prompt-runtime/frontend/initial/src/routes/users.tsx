import { createFileRoute } from "@tanstack/react-router";
import { useState } from "react";
import { Plus, X, Copy, Check } from "lucide-react";
import { AppShell, PageHeader } from "@/components/app-shell";
import { Button, Input, Field, Badge } from "@/components/ui-kit";

export const Route = createFileRoute("/users")({
  component: UsersPage,
});

const MOCK_USERS = [
  { name: "John Doe", email: "john@email.com", role: "Admin", status: "Active" },
  { name: "Mary Jane", email: "mary@email.com", role: "User", status: "Active" },
];

function UsersPage() {
  const [open, setOpen] = useState(false);

  return (
    <AppShell>
      <PageHeader
        title="Users"
        description="Manage members of your organization."
        actions={
          <Button onClick={() => setOpen(true)}>
            <Plus className="h-4 w-4" />
            Invite User
          </Button>
        }
      />

      <div className="rounded-xl border border-border overflow-hidden bg-card">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-border bg-muted/50">
              <th className="text-left font-medium text-xs uppercase tracking-wider text-muted-foreground px-6 py-3">Name</th>
              <th className="text-left font-medium text-xs uppercase tracking-wider text-muted-foreground px-6 py-3">Email</th>
              <th className="text-left font-medium text-xs uppercase tracking-wider text-muted-foreground px-6 py-3">Role</th>
              <th className="text-left font-medium text-xs uppercase tracking-wider text-muted-foreground px-6 py-3">Status</th>
            </tr>
          </thead>
          <tbody>
            {MOCK_USERS.map((u, i) => (
              <tr key={u.email} className={i > 0 ? "border-t border-border" : ""}>
                <td className="px-6 py-4">
                  <div className="flex items-center gap-3">
                    <div className="h-8 w-8 rounded-full bg-muted flex items-center justify-center text-xs font-medium">
                      {u.name.split(" ").map((n) => n[0]).join("")}
                    </div>
                    <span className="font-medium">{u.name}</span>
                  </div>
                </td>
                <td className="px-6 py-4 text-muted-foreground">{u.email}</td>
                <td className="px-6 py-4">
                  <Badge tone={u.role === "Admin" ? "neutral" : "muted"}>{u.role}</Badge>
                </td>
                <td className="px-6 py-4">
                  <span className="inline-flex items-center gap-1.5 text-muted-foreground text-xs">
                    <span className="h-1.5 w-1.5 rounded-full bg-foreground" />
                    {u.status}
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {open && <InviteModal onClose={() => setOpen(false)} />}
    </AppShell>
  );
}

function InviteModal({ onClose }: { onClose: () => void }) {
  const [role, setRole] = useState<"User" | "Admin">("User");
  const [link, setLink] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);

  const copy = () => {
    if (!link) return;
    navigator.clipboard?.writeText(link);
    setCopied(true);
    setTimeout(() => setCopied(false), 1500);
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-foreground/20 backdrop-blur-sm p-4">
      <div className="w-full max-w-md rounded-xl border border-border bg-background shadow-xl">
        <div className="flex items-center justify-between px-6 py-4 border-b border-border">
          <div className="font-medium">Invite user</div>
          <button onClick={onClose} className="text-muted-foreground hover:text-foreground">
            <X className="h-4 w-4" />
          </button>
        </div>

        {!link ? (
          <form
            onSubmit={(e) => {
              e.preventDefault();
              setLink(
                `https://promptflow.ai/signup?token=${Math.random().toString(36).slice(2, 12)}`,
              );
            }}
            className="p-6 space-y-5"
          >
            <Field label="Email">
              <Input type="email" required placeholder="teammate@company.com" />
            </Field>
            <Field label="Role">
              <div className="flex rounded-md border border-border p-0.5">
                {(["User", "Admin"] as const).map((r) => (
                  <button
                    type="button"
                    key={r}
                    onClick={() => setRole(r)}
                    className={
                      "flex-1 text-sm py-1.5 rounded-sm transition-colors " +
                      (role === r
                        ? "bg-foreground text-background"
                        : "text-muted-foreground hover:text-foreground")
                    }
                  >
                    {r}
                  </button>
                ))}
              </div>
            </Field>
            <div className="flex gap-2 justify-end pt-2">
              <Button type="button" variant="secondary" onClick={onClose}>Cancel</Button>
              <Button type="submit">Send Invitation</Button>
            </div>
          </form>
        ) : (
          <div className="p-6 space-y-4">
            <div>
              <div className="text-sm font-medium">Invitation sent</div>
              <div className="text-xs text-muted-foreground mt-1">
                Share this link with your teammate.
              </div>
            </div>
            <div className="flex items-center gap-2 rounded-md border border-border bg-muted/50 px-3 py-2">
              <div className="text-xs font-mono truncate flex-1">{link}</div>
              <button
                onClick={copy}
                className="text-muted-foreground hover:text-foreground shrink-0"
              >
                {copied ? <Check className="h-3.5 w-3.5" /> : <Copy className="h-3.5 w-3.5" />}
              </button>
            </div>
            <div className="flex justify-end">
              <Button onClick={onClose}>Done</Button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
