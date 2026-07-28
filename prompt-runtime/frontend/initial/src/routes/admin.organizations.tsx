import { createFileRoute, Link } from "@tanstack/react-router";
import { useState } from "react";
import { Plus, X, ArrowUpRight, Building2 } from "lucide-react";
import { AppShell, PageHeader } from "@/components/app-shell";
import { Button, Card, Input, Field, Badge } from "@/components/ui-kit";
import { ORGANIZATIONS, type Organization } from "@/lib/mock-store";

export const Route = createFileRoute("/admin/organizations")({
  component: OrganizationsAdmin,
});

function OrganizationsAdmin() {
  const [orgs, setOrgs] = useState<Organization[]>(ORGANIZATIONS);
  const [open, setOpen] = useState(false);

  const addOrg = (name: string, code: string, plan: Organization["plan"]) => {
    setOrgs((prev) => [
      ...prev,
      {
        id: `org_${Math.random().toString(36).slice(2, 8)}`,
        name,
        code: code.toUpperCase(),
        plan,
        createdAt: new Date().toISOString().slice(0, 10),
        members: [],
        flowIds: [],
      },
    ]);
  };

  return (
    <AppShell>
      <PageHeader
        title="Organizations"
        description="All tenants on the platform."
        actions={
          <Button onClick={() => setOpen(true)}>
            <Plus className="h-4 w-4" />
            New Organization
          </Button>
        }
      />

      <div className="rounded-xl border border-border overflow-hidden bg-card">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-border bg-muted/50">
              <th className="text-left font-medium text-xs uppercase tracking-wider text-muted-foreground px-6 py-3">Organization</th>
              <th className="text-left font-medium text-xs uppercase tracking-wider text-muted-foreground px-6 py-3">Code</th>
              <th className="text-left font-medium text-xs uppercase tracking-wider text-muted-foreground px-6 py-3">Plan</th>
              <th className="text-left font-medium text-xs uppercase tracking-wider text-muted-foreground px-6 py-3">Members</th>
              <th className="text-left font-medium text-xs uppercase tracking-wider text-muted-foreground px-6 py-3">Created</th>
              <th className="px-6 py-3" />
            </tr>
          </thead>
          <tbody>
            {orgs.map((o, i) => (
              <tr key={o.id} className={i > 0 ? "border-t border-border" : ""}>
                <td className="px-6 py-4">
                  <div className="flex items-center gap-3">
                    <div className="h-8 w-8 rounded-md border border-border flex items-center justify-center">
                      <Building2 className="h-4 w-4" />
                    </div>
                    <span className="font-medium">{o.name}</span>
                  </div>
                </td>
                <td className="px-6 py-4 font-mono text-xs text-muted-foreground">{o.code}</td>
                <td className="px-6 py-4">
                  <Badge tone={o.plan === "Enterprise" ? "neutral" : "muted"}>{o.plan}</Badge>
                </td>
                <td className="px-6 py-4 text-muted-foreground">{o.members.length}</td>
                <td className="px-6 py-4 text-muted-foreground">{o.createdAt}</td>
                <td className="px-6 py-4 text-right">
                  <Link
                    to="/admin/organizations/$id"
                    params={{ id: o.id }}
                    className="text-xs text-muted-foreground hover:text-foreground inline-flex items-center gap-1"
                  >
                    Manage <ArrowUpRight className="h-3 w-3" />
                  </Link>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {open && (
        <NewOrgModal
          onClose={() => setOpen(false)}
          onCreate={(n, c, p) => {
            addOrg(n, c, p);
            setOpen(false);
          }}
        />
      )}
    </AppShell>
  );
}

function NewOrgModal({
  onClose,
  onCreate,
}: {
  onClose: () => void;
  onCreate: (name: string, code: string, plan: Organization["plan"]) => void;
}) {
  const [name, setName] = useState("");
  const [code, setCode] = useState("");
  const [plan, setPlan] = useState<Organization["plan"]>("Free");

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-foreground/20 backdrop-blur-sm p-4">
      <div className="w-full max-w-md rounded-xl border border-border bg-background shadow-xl">
        <div className="flex items-center justify-between px-6 py-4 border-b border-border">
          <div className="font-medium">New organization</div>
          <button onClick={onClose} className="text-muted-foreground hover:text-foreground">
            <X className="h-4 w-4" />
          </button>
        </div>
        <form
          onSubmit={(e) => {
            e.preventDefault();
            if (!name || !code) return;
            onCreate(name, code, plan);
          }}
          className="p-6 space-y-5"
        >
          <Field label="Name">
            <Input value={name} onChange={(e) => setName(e.target.value)} placeholder="Acme Inc." required />
          </Field>
          <Field label="Code" hint="Short identifier used in signup links.">
            <Input value={code} onChange={(e) => setCode(e.target.value)} placeholder="ACM" required />
          </Field>
          <Field label="Plan">
            <div className="flex rounded-md border border-border p-0.5">
              {(["Free", "Team", "Enterprise"] as const).map((p) => (
                <button
                  type="button"
                  key={p}
                  onClick={() => setPlan(p)}
                  className={
                    "flex-1 text-sm py-1.5 rounded-sm transition-colors " +
                    (plan === p ? "bg-foreground text-background" : "text-muted-foreground hover:text-foreground")
                  }
                >
                  {p}
                </button>
              ))}
            </div>
          </Field>
          <div className="flex gap-2 justify-end pt-2">
            <Button type="button" variant="secondary" onClick={onClose}>Cancel</Button>
            <Button type="submit">Create</Button>
          </div>
        </form>
      </div>
    </div>
  );
}
