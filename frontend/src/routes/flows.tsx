import { Link, useNavigate } from "react-router-dom";
import { useEffect, useMemo, useState } from "react";
import {
  Workflow,
  Plus,
  FileText,
  Play,
  Edit3,
  ShieldCheck,
  Trash2,
  Search,
  Check,
  Building2,
  Loader2,
  X,
  Layers,
} from "lucide-react";
import { AppShell, PageHeader } from "@/components/app-shell";
import { Badge, Button, Card, Input } from "@/components/ui-kit";
import { apiDelete, apiGet, apiPost } from "@/lib/api";
import { authService, type AuthUser } from "@/lib/auth";
import { logActivity } from "@/lib/activity";
import { toast } from "sonner";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";

export type UnifiedFlowItem = {
  id: string; // slug or UUID
  dbId?: string; // Database UUID
  slug?: string;
  name: string;
  description: string | null;
  version?: string;
  runtime?: { mode: "guided" | "automatic"; default_review_required: boolean };
  steps?: Array<{ id: string; name: string; sequence: number }>;
  is_active?: boolean;
};

type OrganizationItem = {
  id: string;
  name: string;
  slug: string;
  code: string;
};

type AccessItem = {
  organization_id: string;
  organization_name: string | null;
  flow_id: string;
  flow_name: string | null;
};

type FlowFormState = {
  name: string;
  description: string;
  content_json: string;
  is_active: boolean;
};

const EMPTY_FORM: FlowFormState = {
  name: "",
  description: "",
  content_json:
    '{\n  "id": "",\n  "version": "1.0",\n  "name": "",\n  "description": "",\n  "runtime": { "mode": "guided", "default_review_required": true },\n  "steps": []\n}',
  is_active: true,
};

export default function FlowsPage() {
  const navigate = useNavigate();
  const [currentUser, setCurrentUser] = useState<AuthUser | null>(null);
  const [flows, setFlows] = useState<UnifiedFlowItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [searchQuery, setSearchQuery] = useState("");

  // Create Flow Modal
  const [createDialogOpen, setCreateDialogOpen] = useState(false);
  const [saving, setSaving] = useState(false);
  const [form, setForm] = useState<FlowFormState>(EMPTY_FORM);

  // Delete Confirmation Dialog
  const [deleteTarget, setDeleteTarget] = useState<UnifiedFlowItem | null>(null);
  const [deleting, setDeleting] = useState(false);

  // Access Management Modal
  const [accessTarget, setAccessTarget] = useState<UnifiedFlowItem | null>(null);
  const [organizations, setOrganizations] = useState<OrganizationItem[]>([]);
  const [assignedOrgIds, setAssignedOrgIds] = useState<Set<string>>(new Set());
  const [loadingAccess, setLoadingAccess] = useState(false);
  const [togglingOrgId, setTogglingOrgId] = useState<string | null>(null);
  const [orgSearchQuery, setOrgSearchQuery] = useState("");

  const isSuperadmin = currentUser?.role === "superadmin";

  async function loadFlows(userRole?: string | null) {
    const role = userRole !== undefined ? userRole : currentUser?.role;
    if (role === "superadmin") {
      try {
        const [adminRes, runtimeRes] = await Promise.all([
          apiGet<{ items: Array<{ id: string; name: string; slug: string; description: string | null; is_active: boolean }> }>("/admin/flows"),
          apiGet<{ items: Array<{ id: string; name: string; version: string; description: string; steps: any[]; runtime?: any }> }>("/flows").catch(() => ({ items: [] })),
        ]);

        const runtimeMap = new Map((runtimeRes.items || []).map((f) => [f.id, f]));
        const merged: UnifiedFlowItem[] = (adminRes.items || []).map((af) => {
          const rf = runtimeMap.get(af.slug) || runtimeMap.get(af.id);
          return {
            id: af.slug || af.id,
            dbId: af.id,
            slug: af.slug,
            name: af.name,
            description: af.description,
            is_active: af.is_active,
            version: rf?.version,
            runtime: rf?.runtime,
            steps: rf?.steps,
          };
        });

        for (const rf of (runtimeRes.items || [])) {
          if (!merged.some((m) => m.slug === rf.id || m.id === rf.id)) {
            merged.push({
              id: rf.id,
              name: rf.name,
              description: rf.description,
              version: rf.version,
              runtime: rf.runtime,
              steps: rf.steps,
              is_active: true,
            });
          }
        }
        setFlows(merged);
      } catch {
        const res = await apiGet<{ items: UnifiedFlowItem[] }>("/flows");
        setFlows(res.items || []);
      }
    } else {
      const res = await apiGet<{ items: UnifiedFlowItem[] }>("/flows");
      setFlows(res.items || []);
    }
  }

  useEffect(() => {
    let active = true;
    setLoading(true);

    authService
      .getMe()
      .then((user) => {
        if (!active) return;
        setCurrentUser(user);
        return loadFlows(user.role);
      })
      .catch(() => {
        if (!active) return;
        return loadFlows(null);
      })
      .finally(() => {
        if (active) setLoading(false);
      });

    return () => {
      active = false;
    };
  }, []);

  async function handleCreateSave() {
    if (!form.name.trim()) {
      toast.error("Please enter a flow name.");
      return;
    }
    setSaving(true);
    try {
      const created = await apiPost<{ item: { id: string } }>("/admin/flows", {
        name: form.name.trim(),
        description: form.description.trim(),
        content_json: form.content_json,
        is_active: form.is_active,
      });
      await loadFlows(currentUser?.role);
      setCreateDialogOpen(false);
      setForm(EMPTY_FORM);
      toast.success("Flow created successfully");
      logActivity("created flow", form.name);
      void navigate(`/admin/flows/${created.item.id}`);
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Failed to create flow");
    } finally {
      setSaving(false);
    }
  }

  async function handleDelete() {
    if (!deleteTarget) return;
    setDeleting(true);
    try {
      const targetId = deleteTarget.dbId || deleteTarget.id;
      await apiDelete(`/admin/flows/${targetId}`);
      await loadFlows(currentUser?.role);
      toast.success(`Flow "${deleteTarget.name}" deleted`);
      logActivity("deleted flow", deleteTarget.name);
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Failed to delete flow");
    } finally {
      setDeleting(false);
      setDeleteTarget(null);
    }
  }

  async function handleMarkdownImport(file: File | null) {
    if (!file) return;
    try {
      const text = await file.text();
      const res = await apiPost<{ items: Array<{ id: string; name: string }>; count: number }>(
        "/admin/forms/import-markdown",
        { content: text }
      );
      toast.success(`Imported ${res.count} form(s) from Markdown into forms catalog`);
      await loadFlows(currentUser?.role);
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Failed to import Markdown");
    }
  }

  async function handleOpenAccess(flow: UnifiedFlowItem) {
    setAccessTarget(flow);
    setOrgSearchQuery("");
    setLoadingAccess(true);
    try {
      const [orgsRes, accessRes] = await Promise.all([
        apiGet<{ items: OrganizationItem[] }>("/admin/organizations"),
        apiGet<{ items: AccessItem[] }>("/admin/manage-flows"),
      ]);
      setOrganizations(orgsRes.items || []);

      const matching = new Set<string>();
      for (const a of (accessRes.items || [])) {
        if (
          (flow.dbId && a.flow_id === flow.dbId) ||
          (flow.id && a.flow_id === flow.id) ||
          (flow.slug && a.flow_id === flow.slug) ||
          (flow.name && a.flow_name === flow.name)
        ) {
          matching.add(a.organization_id);
        }
      }
      setAssignedOrgIds(matching);
    } catch (err) {
      toast.error("Failed to load organization access permissions");
    } finally {
      setLoadingAccess(false);
    }
  }

  async function handleToggleOrgAccess(orgId: string, orgName: string) {
    if (!accessTarget) return;
    const flowKey = accessTarget.dbId || accessTarget.id;
    const isAssigned = assignedOrgIds.has(orgId);
    setTogglingOrgId(orgId);
    try {
      if (isAssigned) {
        await apiDelete(`/admin/organizations/${orgId}/flows/${flowKey}`);
        setAssignedOrgIds((prev) => {
          const next = new Set(prev);
          next.delete(orgId);
          return next;
        });
        toast.success(`Access revoked from ${orgName}`);
      } else {
        await apiPost(`/admin/organizations/${orgId}/flows`, { flow_id: flowKey });
        setAssignedOrgIds((prev) => {
          const next = new Set(prev);
          next.add(orgId);
          return next;
        });
        toast.success(`Access granted to ${orgName}`);
      }
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Failed to update organization access");
    } finally {
      setTogglingOrgId(null);
    }
  }

  const filteredFlows = useMemo(() => {
    if (!searchQuery.trim()) return flows;
    const q = searchQuery.toLowerCase().trim();
    return flows.filter(
      (f) =>
        f.name.toLowerCase().includes(q) ||
        (f.description && f.description.toLowerCase().includes(q)) ||
        f.id.toLowerCase().includes(q)
    );
  }, [flows, searchQuery]);

  const filteredOrganizations = useMemo(() => {
    if (!orgSearchQuery.trim()) return organizations;
    const q = orgSearchQuery.toLowerCase().trim();
    return organizations.filter(
      (org) =>
        org.name.toLowerCase().includes(q) ||
        org.code.toLowerCase().includes(q) ||
        org.slug.toLowerCase().includes(q)
    );
  }, [organizations, orgSearchQuery]);

  return (
    <AppShell>
      <div className="w-full space-y-6">
        <PageHeader
          title="Flows"
          description={
            isSuperadmin
              ? "Manage, compose, and execute step-by-step prompt flows across organizations."
              : "Prompt flows assigned to your organization, executed step by step."
          }
          actions={
            isSuperadmin ? (
              <div className="flex items-center gap-2">
                <label className="inline-flex cursor-pointer items-center gap-1.5 rounded-md border border-border bg-background px-3 py-1.5 text-xs font-medium text-foreground transition-colors hover:bg-muted">
                  <FileText className="h-4 w-4" />
                  Import Markdown
                  <input
                    type="file"
                    accept=".md,.markdown,text/markdown"
                    className="hidden"
                    onChange={(e) => {
                      const file = e.target.files?.[0] ?? null;
                      void handleMarkdownImport(file);
                      e.target.value = "";
                    }}
                  />
                </label>
                <Button
                  onClick={() => {
                    setForm(EMPTY_FORM);
                    setCreateDialogOpen(true);
                  }}
                  className="gap-1.5 bg-foreground text-background hover:bg-foreground/90"
                >
                  <Plus className="h-4 w-4" />
                  Create new flow
                </Button>
              </div>
            ) : undefined
          }
        />

        {/* Search bar */}
        <div className="relative max-w-md">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
          <Input
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Search flows by name or description..."
            className="pl-9 pr-8"
          />
          {searchQuery && (
            <button
              onClick={() => setSearchQuery("")}
              className="absolute right-2.5 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground"
            >
              <X className="h-3.5 w-3.5" />
            </button>
          )}
        </div>

        {/* Flows List */}
        {loading ? (
          <div className="py-12 text-center text-sm text-muted-foreground">
            <Loader2 className="mx-auto h-6 w-6 animate-spin mb-2" />
            Loading flows...
          </div>
        ) : filteredFlows.length === 0 ? (
          <Card className="p-8 text-center">
            <Workflow className="mx-auto h-8 w-8 text-muted-foreground mb-3 opacity-60" />
            <div className="font-medium text-foreground">
              {searchQuery ? "No matching flows found" : "No flows available"}
            </div>
            <p className="mt-1 text-sm text-muted-foreground max-w-sm mx-auto">
              {searchQuery
                ? `No flows matched "${searchQuery}". Clear your search to see all flows.`
                : isSuperadmin
                  ? "Get started by creating a new flow or importing forms from markdown."
                  : "No flows are currently assigned to your organization."}
            </p>
            {isSuperadmin && !searchQuery && (
              <Button
                onClick={() => {
                  setForm(EMPTY_FORM);
                  setCreateDialogOpen(true);
                }}
                className="mt-4 gap-1.5"
              >
                <Plus className="h-4 w-4" />
                Create new flow
              </Button>
            )}
          </Card>
        ) : (
          <div className="grid grid-cols-1 gap-3">
            {filteredFlows.map((flow) => {
              const stepCount = flow.steps?.length;
              return (
                <Card
                  key={flow.id}
                  className="p-5 transition-colors hover:border-foreground/20"
                >
                  <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
                    <div className="min-w-0 flex-1 space-y-1.5">
                      <div className="flex items-center gap-2">
                        <span className="font-semibold text-foreground text-base tracking-tight">
                          {flow.name}
                        </span>
                        {isSuperadmin && flow.is_active === false && (
                          <Badge variant="outline" className="text-[10px] text-muted-foreground">
                            Inactive
                          </Badge>
                        )}
                        {flow.version && (
                          <span className="text-[11px] text-muted-foreground">
                            v{flow.version}
                          </span>
                        )}
                      </div>

                      <p className="text-sm text-muted-foreground line-clamp-2">
                        {flow.description || "No description provided."}
                      </p>

                      <div className="flex items-center gap-3 text-xs text-muted-foreground pt-1">
                        {typeof stepCount === "number" && (
                          <span className="flex items-center gap-1">
                            <Layers className="h-3.5 w-3.5 opacity-70" />
                            {stepCount} {stepCount === 1 ? "step" : "steps"}
                          </span>
                        )}
                        {flow.runtime?.mode && (
                          <span className="capitalize">
                            Mode: {flow.runtime.mode}
                          </span>
                        )}
                      </div>
                    </div>

                    {/* Actions: Open | Edit | Access | Delete */}
                    <div className="flex items-center gap-2 shrink-0 pt-2 sm:pt-0">
                      {/* Open - Available to ALL roles */}
                      <Button
                        asChild
                        variant="outline"
                        size="sm"
                        className="gap-1.5 border-border hover:bg-muted text-foreground font-medium"
                      >
                        <Link to={`/flows/${flow.id}`}>
                          <Play className="h-3.5 w-3.5 fill-current" />
                          <span>Open</span>
                        </Link>
                      </Button>

                      {/* Superadmin actions: Edit | Access | Delete */}
                      {isSuperadmin && (
                        <>
                          <Button
                            asChild
                            variant="secondary"
                            size="sm"
                            className="gap-1.5 font-medium"
                          >
                            <Link to={`/admin/flows/${flow.dbId || flow.id}`}>
                              <Edit3 className="h-3.5 w-3.5" />
                              <span>Edit</span>
                            </Link>
                          </Button>

                          <Button
                            variant="secondary"
                            size="sm"
                            onClick={() => void handleOpenAccess(flow)}
                            className="gap-1.5 font-medium"
                          >
                            <ShieldCheck className="h-3.5 w-3.5" />
                            <span>Access</span>
                          </Button>

                          <Button
                            variant="ghost"
                            size="sm"
                            onClick={() => setDeleteTarget(flow)}
                            className="text-muted-foreground hover:text-destructive hover:bg-destructive/10 px-2"
                            title="Delete flow"
                          >
                            <Trash2 className="h-4 w-4" />
                          </Button>
                        </>
                      )}
                    </div>
                  </div>
                </Card>
              );
            })}
          </div>
        )}

        {/* Create Flow Modal (Superadmin) */}
        <Dialog open={createDialogOpen} onOpenChange={setCreateDialogOpen}>
          <DialogContent className="max-w-md">
            <DialogHeader>
              <DialogTitle>Create new flow</DialogTitle>
              <DialogDescription>
                Define the new flow metadata. You will be able to compose steps immediately after creating.
              </DialogDescription>
            </DialogHeader>
            <div className="grid gap-4 mt-2">
              <div className="space-y-1.5">
                <label className="text-xs font-medium text-foreground/80">Flow Name</label>
                <Input
                  value={form.name}
                  onChange={(e) => setForm((prev) => ({ ...prev, name: e.target.value }))}
                  placeholder="e.g. Client Assessment Flow"
                  autoFocus
                />
              </div>
              <div className="space-y-1.5">
                <label className="text-xs font-medium text-foreground/80">Description</label>
                <Input
                  value={form.description}
                  onChange={(e) => setForm((prev) => ({ ...prev, description: e.target.value }))}
                  placeholder="Brief description of what this flow does..."
                />
              </div>
              <label className="flex items-center gap-2 text-sm text-foreground cursor-pointer pt-1">
                <input
                  type="checkbox"
                  checked={form.is_active}
                  onChange={(e) => setForm((prev) => ({ ...prev, is_active: e.target.checked }))}
                  className="rounded border-border text-foreground"
                />
                Active
              </label>
              <div className="flex justify-end gap-2 pt-3 border-t border-border mt-2">
                <Button
                  variant="secondary"
                  onClick={() => setCreateDialogOpen(false)}
                  disabled={saving}
                >
                  Cancel
                </Button>
                <Button
                  onClick={() => void handleCreateSave()}
                  disabled={saving || !form.name.trim()}
                  className="bg-foreground text-background hover:bg-foreground/90"
                >
                  {saving ? "Creating..." : "Create & Compose"}
                </Button>
              </div>
            </div>
          </DialogContent>
        </Dialog>

        {/* Access Management Modal (Superadmin) */}
        <Dialog open={!!accessTarget} onOpenChange={(open) => !open && setAccessTarget(null)}>
          <DialogContent className="max-w-lg">
            <DialogHeader>
              <DialogTitle className="flex items-center gap-2">
                <ShieldCheck className="h-5 w-5 text-foreground" />
                Manage Flow Access
              </DialogTitle>
              <DialogDescription>
                Configure which organizations can see and run{" "}
                <span className="font-semibold text-foreground">"{accessTarget?.name}"</span>.
              </DialogDescription>
            </DialogHeader>

            {loadingAccess ? (
              <div className="py-8 text-center text-sm text-muted-foreground">
                <Loader2 className="mx-auto h-5 w-5 animate-spin mb-2" />
                Loading organization permissions...
              </div>
            ) : (
              <div className="space-y-4 pt-2">
                {organizations.length > 3 && (
                  <div className="relative">
                    <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-muted-foreground" />
                    <Input
                      value={orgSearchQuery}
                      onChange={(e) => setOrgSearchQuery(e.target.value)}
                      placeholder="Search organizations..."
                      className="pl-8 text-xs h-9"
                    />
                  </div>
                )}

                <div className="max-h-72 overflow-y-auto space-y-2 pr-1">
                  {filteredOrganizations.length === 0 ? (
                    <div className="py-6 text-center text-xs text-muted-foreground">
                      No organizations found.
                    </div>
                  ) : (
                    filteredOrganizations.map((org) => {
                      const hasAccess = assignedOrgIds.has(org.id);
                      const isToggling = togglingOrgId === org.id;

                      return (
                        <div
                          key={org.id}
                          className="flex items-center justify-between gap-3 p-3 rounded-lg border border-border bg-card/50 hover:bg-card transition-colors"
                        >
                          <div className="min-w-0 flex-1">
                            <div className="flex items-center gap-1.5 font-medium text-xs text-foreground">
                              <Building2 className="h-3.5 w-3.5 text-muted-foreground shrink-0" />
                              <span className="truncate">{org.name}</span>
                            </div>
                            <div className="text-[10px] text-muted-foreground mt-0.5">
                              Code: {org.code || org.slug}
                            </div>
                          </div>

                          <Button
                            size="sm"
                            variant={hasAccess ? "outline" : "secondary"}
                            disabled={isToggling}
                            onClick={() => void handleToggleOrgAccess(org.id, org.name)}
                            className={`h-8 text-xs gap-1.5 shrink-0 ${hasAccess
                              ? "border-foreground/30 hover:bg-destructive/10 hover:text-destructive hover:border-destructive/30"
                              : "bg-foreground text-white hover:bg-foreground/90 hover:text-black"
                              }`}
                          >
                            {isToggling ? (
                              <Loader2 className="h-3 w-3 animate-spin" />
                            ) : hasAccess ? (
                              <>
                                <Check className="h-3 w-3" />
                                <span>Granted</span>
                              </>
                            ) : (
                              <span>Grant</span>
                            )}
                          </Button>
                        </div>
                      );
                    })
                  )}
                </div>

                <div className="flex justify-between items-center pt-3 border-t border-border text-xs text-muted-foreground">
                  <span>
                    {assignedOrgIds.size} of {organizations.length} organizations granted
                  </span>
                  <Button variant="secondary" size="sm" onClick={() => setAccessTarget(null)}>
                    Done
                  </Button>
                </div>
              </div>
            )}
          </DialogContent>
        </Dialog>

        {/* Delete Confirmation Alert Dialog (Superadmin) */}
        <AlertDialog open={!!deleteTarget} onOpenChange={(open) => !open && setDeleteTarget(null)}>
          <AlertDialogContent>
            <AlertDialogHeader>
              <AlertDialogTitle>Delete flow "{deleteTarget?.name}"?</AlertDialogTitle>
              <AlertDialogDescription>
                This action cannot be undone. All configured steps and history associated with this flow
                will be permanently deleted.
              </AlertDialogDescription>
            </AlertDialogHeader>
            <AlertDialogFooter>
              <AlertDialogCancel disabled={deleting}>Cancel</AlertDialogCancel>
              <AlertDialogAction
                onClick={() => void handleDelete()}
                disabled={deleting}
                className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
              >
                {deleting ? "Deleting..." : "Delete"}
              </AlertDialogAction>
            </AlertDialogFooter>
          </AlertDialogContent>
        </AlertDialog>
      </div>
    </AppShell>
  );
}
