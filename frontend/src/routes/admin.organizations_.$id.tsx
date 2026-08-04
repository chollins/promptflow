import { createFileRoute, Link } from "@tanstack/react-router";
import { useEffect, useMemo, useState } from "react";
import { ArrowLeft, Plus, Trash2, UserPlus } from "lucide-react";
import { AppShell, PageHeader } from "@/components/app-shell";
import { Badge, Button, Card, Field, Input } from "@/components/ui-kit";
import { apiDelete, apiGet, apiPost } from "@/lib/api";

type OrganizationDetail = {
  id: string;
  name: string;
  slug: string;
  code: string;
  is_active: boolean;
  users: Array<{ id: string; name: string; email: string; role: string | null; is_active: boolean }>;
  flows: Array<{ flow_id: string; flow_name: string | null; flow_slug: string | null }>;
};

type FlowItem = {
  id: string;
  name: string;
  slug: string;
  is_active: boolean;
};

type AdminInviteForm = {
  name: string;
  email: string;
  password: string;
  confirmPassword: string;
};

const EMPTY_ADMIN_FORM: AdminInviteForm = {
  name: "",
  email: "",
  password: "",
  confirmPassword: "",
};

export const Route = createFileRoute("/admin/organizations_/$id")({
  component: OrganizationDetailPage,
});

function OrganizationDetailPage() {
  const { id } = Route.useParams();
  const [item, setItem] = useState<OrganizationDetail | null>(null);
  const [flows, setFlows] = useState<FlowItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [flowLoading, setFlowLoading] = useState(true);
  const [savingAdmin, setSavingAdmin] = useState(false);
  const [savingFlowId, setSavingFlowId] = useState<string | null>(null);
  const [adminError, setAdminError] = useState<string | null>(null);
  const [flowError, setFlowError] = useState<string | null>(null);
  const [adminForm, setAdminForm] = useState<AdminInviteForm>(EMPTY_ADMIN_FORM);

  const assignedFlowIds = useMemo(
    () => new Set((item?.flows || []).map((flow) => flow.flow_id)),
    [item],
  );

  function refresh() {
    setLoading(true);
    return apiGet<OrganizationDetail>(`/admin/organizations/${id}`)
      .then((data) => setItem(data))
      .catch(() => setItem(null))
      .finally(() => setLoading(false));
  }

  useEffect(() => {
    void refresh();
    setFlowLoading(true);
    apiGet<{ items: FlowItem[] }>("/admin/flows")
      .then((data) => setFlows(data.items))
      .catch(() => setFlows([]))
      .finally(() => setFlowLoading(false));
  }, [id]);

  async function handleAddAdmin() {
    setSavingAdmin(true);
    setAdminError(null);
    try {
      const payload = {
        name: adminForm.name.trim(),
        email: adminForm.email.trim(),
        password: adminForm.password,
        role: "admin",
      };

      if (!payload.name) throw new Error("Admin name is required.");
      if (!payload.email) throw new Error("Admin email is required.");
      if (!payload.password) throw new Error("Admin password is required.");
      if (payload.password !== adminForm.confirmPassword) throw new Error("Passwords do not match.");

      await apiPost(`/admin/organizations/${id}/admins`, payload);
      setAdminForm(EMPTY_ADMIN_FORM);
      await refresh();
    } catch (err) {
      setAdminError(err instanceof Error ? err.message : "Failed to add admin");
    } finally {
      setSavingAdmin(false);
    }
  }

  async function toggleFlow(flow: FlowItem) {
    setSavingFlowId(flow.id);
    setFlowError(null);
    try {
      if (assignedFlowIds.has(flow.id)) {
        await apiDelete(`/admin/organizations/${id}/flows/${flow.id}`);
      } else {
        await apiPost(`/admin/organizations/${id}/flows`, { flow_id: flow.id });
      }
      await refresh();
    } catch (err) {
      setFlowError(err instanceof Error ? err.message : "Failed to update flow access");
    } finally {
      setSavingFlowId(null);
    }
  }

  return (
    <AppShell>
      <Link
        to="/admin/organizations"
        className="inline-flex items-center gap-1.5 text-xs text-muted-foreground hover:text-foreground mb-4"
      >
        <ArrowLeft className="h-3 w-3" /> Organizations
      </Link>
      <PageHeader
        title={item?.name || "Organization details"}
        description={item ? `${item.code} · ${item.slug}` : "Loading organization data from the backend."}
      />
      <div className="space-y-4">
        <Card className="p-5">
          <div className="flex items-center justify-between gap-4">
            <div>
              <div className="font-medium">Organization status</div>
              <p className="mt-1 text-sm text-muted-foreground">
                {item ? "Loaded from backend." : "No organization data available."}
              </p>
            </div>
            {item && (
              <Badge tone={item.is_active ? "neutral" : "muted"}>
                {item.is_active ? "Active" : "Inactive"}
              </Badge>
            )}
          </div>
        </Card>

        <Card className="p-5 space-y-4">
          <div className="flex items-center gap-2 font-medium">
            <UserPlus className="h-4 w-4" />
            Add admin
          </div>
          <p className="text-sm text-muted-foreground">
            Create the organization admin here. This submits the user record and assigns it to this organization.
          </p>

          {adminError && (
            <div className="rounded-lg border border-red-200 bg-red-50 p-3 text-sm text-red-700">
              {adminError}
            </div>
          )}

          <div className="grid gap-4 md:grid-cols-2">
            <Field label="Admin Name">
              <Input
                value={adminForm.name}
                onChange={(e) => setAdminForm((prev) => ({ ...prev, name: e.target.value }))}
                placeholder="Jane Doe"
              />
            </Field>
            <Field label="Admin Email">
              <Input
                type="email"
                value={adminForm.email}
                onChange={(e) => setAdminForm((prev) => ({ ...prev, email: e.target.value }))}
                placeholder="jane@company.com"
              />
            </Field>
            <Field label="Password">
              <Input
                type="password"
                value={adminForm.password}
                onChange={(e) => setAdminForm((prev) => ({ ...prev, password: e.target.value }))}
              />
            </Field>
            <Field label="Confirm Password">
              <Input
                type="password"
                value={adminForm.confirmPassword}
                onChange={(e) =>
                  setAdminForm((prev) => ({ ...prev, confirmPassword: e.target.value }))
                }
              />
            </Field>
          </div>

          <div className="flex gap-2">
            <Button onClick={() => void handleAddAdmin()} disabled={savingAdmin || loading || !item}>
              {savingAdmin ? "Creating admin..." : "Add admin"}
            </Button>
            <Button variant="secondary" onClick={() => setAdminForm(EMPTY_ADMIN_FORM)}>
              Reset
            </Button>
          </div>
        </Card>

        <Card className="p-5">
          <div className="font-medium">Members</div>
          <div className="mt-4 space-y-3">
            {(item?.users || []).map((user) => (
              <div key={user.id} className="flex items-center justify-between rounded-md border border-border p-3">
                <div>
                  <div className="text-sm font-medium">{user.name}</div>
                  <div className="text-xs text-muted-foreground">{user.email}</div>
                </div>
                <div className="text-xs text-muted-foreground">{user.role || "Member"}</div>
              </div>
            ))}
            {item && item.users.length === 0 && (
              <p className="text-sm text-muted-foreground">No members assigned.</p>
            )}
          </div>
        </Card>

        <Card className="p-5">
          <div className="flex items-center justify-between gap-3">
            <div>
              <div className="font-medium">Assigned flows</div>
              <p className="mt-1 text-sm text-muted-foreground">
                Manage which flows this organization can access.
              </p>
            </div>
            <Badge tone="outline">{item?.flows.length || 0} assigned</Badge>
          </div>

          {flowError && (
            <div className="mt-4 rounded-lg border border-red-200 bg-red-50 p-3 text-sm text-red-700">
              {flowError}
            </div>
          )}

          <div className="mt-4 space-y-3">
            {flowLoading ? (
              <p className="text-sm text-muted-foreground">Loading flows...</p>
            ) : flows.length === 0 ? (
              <p className="text-sm text-muted-foreground">No flows available.</p>
            ) : (
              flows.map((flow) => {
                const assigned = assignedFlowIds.has(flow.id);
                return (
                  <div
                    key={flow.id}
                    className="flex items-center justify-between rounded-md border border-border p-3"
                  >
                    <div>
                      <div className="text-sm font-medium">{flow.name}</div>
                      <div className="text-xs text-muted-foreground">{flow.slug}</div>
                    </div>
                    <Button
                      variant={assigned ? "secondary" : "primary"}
                      size="sm"
                      onClick={() => void toggleFlow(flow)}
                      disabled={savingFlowId === flow.id}
                    >
                      {assigned ? (
                        <>
                          <Trash2 className="h-4 w-4" />
                          Remove
                        </>
                      ) : (
                        <>
                          <Plus className="h-4 w-4" />
                          Add
                        </>
                      )}
                    </Button>
                  </div>
                );
              })
            )}
          </div>
        </Card>
      </div>
    </AppShell>
  );
}
