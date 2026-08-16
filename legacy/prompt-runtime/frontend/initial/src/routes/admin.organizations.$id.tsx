import { createFileRoute, Link, notFound } from "@tanstack/react-router";
import { useState } from "react";
import { Plus, X, Trash2, ArrowLeft, Check, Minus } from "lucide-react";
import { AppShell, PageHeader } from "@/components/app-shell";
import { Button, Card, Input, Field, Badge } from "@/components/ui-kit";
import { ORGANIZATIONS, FLOWS, type OrgMember, type Organization } from "@/lib/mock-store";

export const Route = createFileRoute("/admin/organizations/$id")({
  loader: ({ params }) => {
    const org = ORGANIZATIONS.find((o) => o.id === params.id);
    if (!org) throw notFound();
    return { org };
  },
  component: OrgDetail,
  notFoundComponent: () => (
    <AppShell>
      <PageHeader title="Organization not found" />
      <Link to="/admin/organizations" className="text-sm underline">Back to organizations</Link>
    </AppShell>
  ),
});

function OrgDetail() {
  const { org: initial } = Route.useLoaderData();
  const [org, setOrg] = useState<Organization>(initial);
  const [inviteOpen, setInviteOpen] = useState(false);

  const removeMember = (email: string) =>
    setOrg((o) => ({ ...o, members: o.members.filter((m) => m.email !== email) }));

  const addMember = (m: OrgMember) =>
    setOrg((o) => ({ ...o, members: [...o.members, m] }));

  const toggleFlow = (flowId: string) =>
    setOrg((o) => ({
      ...o,
      flowIds: o.flowIds.includes(flowId)
        ? o.flowIds.filter((f) => f !== flowId)
        : [...o.flowIds, flowId],
    }));

  return (
    <AppShell>
      <Link
        to="/admin/organizations"
        className="inline-flex items-center gap-1.5 text-xs text-muted-foreground hover:text-foreground mb-4"
      >
        <ArrowLeft className="h-3 w-3" /> Organizations
      </Link>

      <PageHeader
        title={org.name}
        description={`Code ${org.code} · ${org.plan} · Created ${org.createdAt}`}
      />

      <Card className="mb-6">
        <div className="flex items-center justify-between mb-4">
          <div>
            <div className="font-medium">Members</div>
            <div className="text-xs text-muted-foreground mt-0.5">{org.members.length} in this workspace</div>
          </div>
          <Button size="sm" onClick={() => setInviteOpen(true)}>
            <Plus className="h-3.5 w-3.5" />
            Add member
          </Button>
        </div>
        {org.members.length === 0 ? (
          <div className="text-sm text-muted-foreground py-6 text-center border border-dashed border-border rounded-md">
            No members yet.
          </div>
        ) : (
          <div className="divide-y divide-border">
            {org.members.map((m) => (
              <div key={m.email} className="flex items-center justify-between py-3">
                <div className="flex items-center gap-3">
                  <div className="h-8 w-8 rounded-full bg-muted flex items-center justify-center text-xs font-medium">
                    {m.name.split(" ").map((n) => n[0]).join("")}
                  </div>
                  <div>
                    <div className="text-sm font-medium">{m.name}</div>
                    <div className="text-xs text-muted-foreground">{m.email}</div>
                  </div>
                </div>
                <div className="flex items-center gap-3">
                  <Badge tone={m.role === "Admin" ? "neutral" : "muted"}>{m.role}</Badge>
                  <Badge tone="outline">{m.status}</Badge>
                  <button
                    onClick={() => removeMember(m.email)}
                    className="text-muted-foreground hover:text-foreground"
                    aria-label="Remove member"
                  >
                    <Trash2 className="h-3.5 w-3.5" />
                  </button>
                </div>
              </div>
            ))}
          </div>
        )}
      </Card>

      <Card>
        <div className="flex items-start justify-between mb-4 gap-4">
          <div>
            <div className="font-medium">Accessed flows</div>
            <div className="text-xs text-muted-foreground mt-0.5">
              {org.flowIds.length} of {FLOWS.length} flows assigned to this organization.
            </div>
          </div>
          <div className="flex gap-2">
            <Button
              size="sm"
              variant="secondary"
              onClick={() => setOrg((o) => ({ ...o, flowIds: [] }))}
              disabled={org.flowIds.length === 0}
            >
              Remove all
            </Button>
            <Button
              size="sm"
              onClick={() => setOrg((o) => ({ ...o, flowIds: FLOWS.map((f) => f.id) }))}
              disabled={org.flowIds.length === FLOWS.length}
            >
              Assign all
            </Button>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <div className="text-xs uppercase tracking-wider text-muted-foreground mb-2">
              Assigned ({org.flowIds.length})
            </div>
            {org.flowIds.length === 0 ? (
              <div className="text-sm text-muted-foreground py-6 text-center border border-dashed border-border rounded-md">
                No flows assigned.
              </div>
            ) : (
              <div className="space-y-2">
                {FLOWS.filter((f) => org.flowIds.includes(f.id)).map((f) => (
                  <div key={f.id} className="flex items-center justify-between rounded-md border border-border p-3">
                    <div>
                      <div className="text-sm font-medium flex items-center gap-2">
                        <Check className="h-3.5 w-3.5" />
                        {f.name}
                      </div>
                      <div className="text-xs text-muted-foreground mt-0.5">{f.description}</div>
                    </div>
                    <Button size="sm" variant="secondary" onClick={() => toggleFlow(f.id)}>
                      <Minus className="h-3.5 w-3.5" />
                      Remove
                    </Button>
                  </div>
                ))}
              </div>
            )}
          </div>

          <div>
            <div className="text-xs uppercase tracking-wider text-muted-foreground mb-2">
              Available ({FLOWS.length - org.flowIds.length})
            </div>
            {FLOWS.length - org.flowIds.length === 0 ? (
              <div className="text-sm text-muted-foreground py-6 text-center border border-dashed border-border rounded-md">
                All flows assigned.
              </div>
            ) : (
              <div className="space-y-2">
                {FLOWS.filter((f) => !org.flowIds.includes(f.id)).map((f) => (
                  <div key={f.id} className="flex items-center justify-between rounded-md border border-border p-3">
                    <div>
                      <div className="text-sm font-medium">{f.name}</div>
                      <div className="text-xs text-muted-foreground mt-0.5">{f.description}</div>
                    </div>
                    <Button size="sm" onClick={() => toggleFlow(f.id)}>
                      <Plus className="h-3.5 w-3.5" />
                      Assign
                    </Button>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </Card>

      {inviteOpen && (
        <AddMemberModal
          onClose={() => setInviteOpen(false)}
          onAdd={(m) => {
            addMember(m);
            setInviteOpen(false);
          }}
        />
      )}
    </AppShell>
  );
}

function AddMemberModal({ onClose, onAdd }: { onClose: () => void; onAdd: (m: OrgMember) => void }) {
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [role, setRole] = useState<"User" | "Admin">("User");

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-foreground/20 backdrop-blur-sm p-4">
      <div className="w-full max-w-md rounded-xl border border-border bg-background shadow-xl">
        <div className="flex items-center justify-between px-6 py-4 border-b border-border">
          <div className="font-medium">Add member</div>
          <button onClick={onClose} className="text-muted-foreground hover:text-foreground">
            <X className="h-4 w-4" />
          </button>
        </div>
        <form
          onSubmit={(e) => {
            e.preventDefault();
            if (!email) return;
            onAdd({ name: name || email.split("@")[0], email, role, status: "Invited" });
          }}
          className="p-6 space-y-5"
        >
          <Field label="Name">
            <Input value={name} onChange={(e) => setName(e.target.value)} placeholder="Jane Doe" />
          </Field>
          <Field label="Email">
            <Input type="email" value={email} onChange={(e) => setEmail(e.target.value)} placeholder="jane@company.com" required />
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
                    (role === r ? "bg-foreground text-background" : "text-muted-foreground hover:text-foreground")
                  }
                >
                  {r}
                </button>
              ))}
            </div>
          </Field>
          <div className="flex gap-2 justify-end pt-2">
            <Button type="button" variant="secondary" onClick={onClose}>Cancel</Button>
            <Button type="submit">Add</Button>
          </div>
        </form>
      </div>
    </div>
  );
}
