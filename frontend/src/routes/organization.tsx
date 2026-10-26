import { useEffect, useMemo, useState } from "react";
import {
  Building2,
  Copy,
  Check,
  Users,
  Tag,
  Pencil,
  Trash2,
  Search,
  Plus,
  X,
  Loader2,
  Sparkles,
} from "lucide-react";
import { AppShell, PageHeader } from "@/components/app-shell";
import { Badge, Button, Card, Input } from "@/components/ui-kit";
import { apiDelete, apiGet, apiPost } from "@/lib/api";
import { authService } from "@/lib/auth";
import { toast } from "sonner";

type Organization = {
  id: string;
  name: string;
  slug: string;
  code: string;
  is_active: boolean;
  created_at?: string | null;
  updated_at?: string | null;
  users?: Array<{
    id: string;
    name: string;
    email: string;
    role: string | null;
    is_active: boolean;
  }>;
};

type AuthUser = {
  id: string;
  name: string;
  email: string;
  role: string | null;
  organization_id: string;
};

type ModelItem = {
  id: string;
  name: string;
  provider: string;
  category: string;
  description?: string | null;
  is_active: boolean;
  recommended?: boolean;
  custom_label?: string | null;
};

export default function OrganizationPage() {
  const [user, setUser] = useState<AuthUser | null>(null);
  const [organization, setOrganization] = useState<Organization | null>(null);
  const [loading, setLoading] = useState(true);
  const [showLink, setShowLink] = useState(false);
  const [copied, setCopied] = useState(false);

  // Model Labels state (Exclusive for Org Admins)
  const [models, setModels] = useState<ModelItem[]>([]);
  const [modelLabels, setModelLabels] = useState<Record<string, string>>({});
  const [modelsLoading, setModelsLoading] = useState(true);
  const [searchQuery, setSearchQuery] = useState("");
  const [categoryFilter, setCategoryFilter] = useState<string>("all");
  const [statusFilter, setStatusFilter] = useState<"all" | "labeled" | "unlabeled">("all");

  // Inline editing state
  const [editingModelId, setEditingModelId] = useState<string | null>(null);
  const [editingValue, setEditingValue] = useState("");
  const [savingModelId, setSavingModelId] = useState<string | null>(null);

  const registrationLink = useMemo(() => {
    if (!organization) return "";
    if (typeof window === "undefined") return "";
    return `${window.location.origin}/signup?oc=${organization.code}`;
  }, [organization]);

  const loadModelsAndLabels = async () => {
    setModelsLoading(true);
    try {
      const [modelsRes, labelsRes] = await Promise.all([
        apiGet<{ models: ModelItem[] }>("/models"),
        apiGet<{ labels: Record<string, string> }>("/organization/model-labels").catch(() => ({
          labels: {},
        })),
      ]);
      setModels(modelsRes.models || []);
      setModelLabels(labelsRes.labels || {});
    } catch (err) {
      console.error("Failed to load models or labels:", err);
    } finally {
      setModelsLoading(false);
    }
  };

  useEffect(() => {
    let active = true;

    async function load() {
      setLoading(true);
      try {
        const me = await authService.getMe();
        if (!active) return;
        setUser(me);

        if (!me.organization_id) {
          setOrganization(null);
          return;
        }

        const org = await apiGet<Organization>(`/organizations/${me.organization_id}`);
        if (!active) return;
        setOrganization(org);
      } catch {
        if (active) {
          setUser(null);
          setOrganization(null);
        }
      } finally {
        if (active) setLoading(false);
      }
    }

    void load();
    void loadModelsAndLabels();

    return () => {
      active = false;
    };
  }, []);

  function copyRegistrationLink() {
    if (!registrationLink) return;
    navigator.clipboard?.writeText(registrationLink);
    setCopied(true);
    window.setTimeout(() => setCopied(false), 1500);
  }

  const startEdit = (modelId: string, currentLabel: string) => {
    setEditingModelId(modelId);
    setEditingValue(currentLabel || "");
  };

  const cancelEdit = () => {
    setEditingModelId(null);
    setEditingValue("");
  };

  const saveLabel = async (modelId: string) => {
    const cleanLabel = editingValue.trim();
    setSavingModelId(modelId);
    try {
      if (!cleanLabel) {
        // Clear label if empty
        await apiDelete(`/organization/model-labels/${modelId}`);
        setModelLabels((prev) => {
          const next = { ...prev };
          delete next[modelId];
          return next;
        });
        toast.success("Model nickname removed");
      } else {
        await apiPost<{ ok: boolean; label: string }>("/organization/model-labels", {
          model_id: modelId,
          label: cleanLabel,
        });
        setModelLabels((prev) => ({
          ...prev,
          [modelId]: cleanLabel,
        }));
        toast.success(`Nickname set to "${cleanLabel}"`);
      }
      setEditingModelId(null);
      setEditingValue("");
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err);
      toast.error(`Failed to save nickname: ${msg}`);
    } finally {
      setSavingModelId(null);
    }
  };

  const clearLabel = async (modelId: string, modelName: string) => {
    setSavingModelId(modelId);
    try {
      await apiDelete(`/organization/model-labels/${modelId}`);
      setModelLabels((prev) => {
        const next = { ...prev };
        delete next[modelId];
        return next;
      });
      toast.success(`Removed nickname for ${modelName}`);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err);
      toast.error(`Failed to remove nickname: ${msg}`);
    } finally {
      setSavingModelId(null);
    }
  };

  // Filter models
  const filteredModels = useMemo(() => {
    return models.filter((m) => {
      if (categoryFilter !== "all" && m.category !== categoryFilter) {
        return false;
      }
      const label = modelLabels[m.id];
      if (statusFilter === "labeled" && !label) {
        return false;
      }
      if (statusFilter === "unlabeled" && label) {
        return false;
      }
      if (searchQuery.trim()) {
        const q = searchQuery.toLowerCase();
        const matchName = m.name.toLowerCase().includes(q);
        const matchId = m.id.toLowerCase().includes(q);
        const matchProvider = m.provider.toLowerCase().includes(q);
        const matchLabel = (label || "").toLowerCase().includes(q);
        if (!matchName && !matchId && !matchProvider && !matchLabel) {
          return false;
        }
      }
      return true;
    });
  }, [models, modelLabels, categoryFilter, statusFilter, searchQuery]);

  const labeledCount = useMemo(() => {
    return Object.keys(modelLabels).length;
  }, [modelLabels]);

  return (
    <AppShell>
      <PageHeader
        title="Organization"
        description="Manage your organization scope and configure custom AI model nicknames for your members."
      />

      <div className="space-y-6">
        {/* Organization details card */}
        <Card className="p-5">
          <div className="flex items-start justify-between gap-4">
            <div>
              <div className="flex items-center gap-2 text-sm font-medium">
                <Building2 className="h-4 w-4" />
                Organization details
              </div>
              <p className="mt-1 text-sm text-muted-foreground">
                {loading
                  ? "Loading your organization..."
                  : organization
                    ? "Loaded from your account membership."
                    : "No organization assigned to this account."}
              </p>
            </div>
            {organization && (
              <Badge tone={organization.is_active ? "neutral" : "muted"}>
                {organization.is_active ? "Active" : "Inactive"}
              </Badge>
            )}
          </div>

          {organization && (
            <div className="mt-6 grid gap-4 md:grid-cols-2">
              <div>
                <div className="text-xs text-muted-foreground mb-1.5">Organization Name</div>
                <div className="text-base font-medium">{organization.name}</div>
              </div>
              <div>
                <div className="text-xs text-muted-foreground mb-1.5">Organization Code</div>
                <div className="text-base font-mono font-medium">{organization.code}</div>
              </div>
              <div>
                <div className="text-xs text-muted-foreground mb-1.5">Slug</div>
                <div className="text-sm font-mono">{organization.slug}</div>
              </div>
              <div>
                <div className="text-xs text-muted-foreground mb-1.5">Signed-in user</div>
                <div className="text-sm">
                  {user?.name || "Unknown"} {user?.role ? `(${user.role})` : ""}
                </div>
              </div>
            </div>
          )}
        </Card>

        {/* AI Model Custom Labels & Nicknames Card (Exclusive for Org Admins) */}
        {organization && (
          <Card className="overflow-hidden p-0 border border-border shadow-xs">
            <div className="flex flex-col md:flex-row md:items-center justify-between border-b border-border px-5 py-4 bg-muted/10 gap-3">
              <div>
                <div className="flex items-center gap-2">
                  <Tag className="h-4 w-4 text-primary" />
                  <span className="font-semibold text-sm text-foreground">AI Model Nicknames</span>
                  <span className="inline-flex items-center px-2 py-0.5 rounded-full text-[10px] font-semibold bg-emerald-500/10 text-emerald-700 dark:text-emerald-400 border border-emerald-500/20">
                    Org Admin Exclusive
                  </span>
                </div>
                <p className="mt-1 text-xs text-muted-foreground">
                  Give AI models intuitive nicknames for your organization. All members will see these labels when selecting models.
                </p>
              </div>
              <div className="flex items-center gap-2">
                <Badge tone="muted" className="text-xs font-medium">
                  {labeledCount} of {models.length} labeled
                </Badge>
              </div>
            </div>

            {/* Filter toolbar */}
            <div className="p-4 border-b border-border/60 bg-muted/5 flex flex-col md:flex-row gap-3 items-stretch md:items-center justify-between">
              <div className="relative flex-1 max-w-sm">
                <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-muted-foreground" />
                <Input
                  placeholder="Filter models or nicknames..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="pl-8 h-8 text-xs"
                />
              </div>

              <div className="flex items-center gap-2 flex-wrap">
                {/* Category Pills */}
                <div className="inline-flex rounded-lg border border-border bg-background p-0.5 text-xs">
                  {(["all", "text", "image", "audio"] as const).map((cat) => (
                    <button
                      key={cat}
                      type="button"
                      onClick={() => setCategoryFilter(cat)}
                      className={`px-2.5 py-1 rounded-md text-xs font-medium capitalize transition-colors ${categoryFilter === cat
                          ? "bg-muted text-foreground font-semibold"
                          : "text-muted-foreground hover:text-foreground"
                        }`}
                    >
                      {cat}
                    </button>
                  ))}
                </div>

                {/* Status Pills */}
                <div className="inline-flex rounded-lg border border-border bg-background p-0.5 text-xs">
                  {(["all", "labeled", "unlabeled"] as const).map((st) => (
                    <button
                      key={st}
                      type="button"
                      onClick={() => setStatusFilter(st)}
                      className={`px-2.5 py-1 rounded-md text-xs font-medium capitalize transition-colors ${statusFilter === st
                          ? "bg-muted text-foreground font-semibold"
                          : "text-muted-foreground hover:text-foreground"
                        }`}
                    >
                      {st}
                    </button>
                  ))}
                </div>
              </div>
            </div>

            {/* Models Table */}
            <div className="overflow-x-auto">
              {modelsLoading ? (
                <div className="py-12 text-center text-xs text-muted-foreground">
                  <Loader2 className="h-6 w-6 animate-spin mx-auto mb-2 opacity-50" />
                  Loading models...
                </div>
              ) : filteredModels.length === 0 ? (
                <div className="py-12 text-center text-xs text-muted-foreground">
                  No AI models match the current filter.
                </div>
              ) : (
                <table className="w-full text-sm">
                  <thead className="border-b border-border bg-muted/30 text-left">
                    <tr>
                      <th className="px-5 py-3 text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                        Model &amp; Provider
                      </th>
                      <th className="px-5 py-3 text-xs font-semibold uppercase tracking-wider text-muted-foreground w-28">
                        Category
                      </th>
                      <th className="px-5 py-3 text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                        Organization Nickname / Label
                      </th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-border">
                    {filteredModels.map((model) => {
                      const currentLabel = modelLabels[model.id] || "";
                      const isEditing = editingModelId === model.id;
                      const isSaving = savingModelId === model.id;

                      return (
                        <tr key={model.id} className="hover:bg-muted/20 transition-colors">
                          <td className="px-5 py-3">
                            <div className="flex items-center gap-2">
                              <span className="font-medium text-foreground">{model.name}</span>
                              <span className="inline-flex items-center px-1.5 py-0.5 rounded text-[10px] font-mono uppercase bg-muted text-muted-foreground border border-border">
                                {model.provider}
                              </span>
                              {model.recommended && (
                                <span className="inline-flex items-center gap-0.5 px-1.5 py-0.5 rounded-full text-[9px] font-medium bg-muted text-muted-foreground border border-border">
                                  <Sparkles className="h-2 w-2" />
                                  Rec
                                </span>
                              )}
                            </div>
                            <div className="text-xs text-muted-foreground font-mono mt-0.5">
                              {model.id}
                            </div>
                          </td>

                          <td className="px-5 py-3">
                            <span className="capitalize text-xs font-medium text-muted-foreground px-2 py-0.5 rounded bg-muted/60 border border-border">
                              {model.category}
                            </span>
                          </td>

                          <td className="px-5 py-3">
                            {isEditing ? (
                              <div className="flex items-center gap-2 max-w-md">
                                <Input
                                  value={editingValue}
                                  onChange={(e) => setEditingValue(e.target.value)}
                                  placeholder="e.g. Fast Draft Assistant"
                                  className="h-8 text-xs flex-1"
                                  autoFocus
                                  disabled={isSaving}
                                  onKeyDown={(e) => {
                                    if (e.key === "Enter") {
                                      e.preventDefault();
                                      void saveLabel(model.id);
                                    } else if (e.key === "Escape") {
                                      cancelEdit();
                                    }
                                  }}
                                />
                                <Button
                                  size="sm"
                                  onClick={() => void saveLabel(model.id)}
                                  disabled={isSaving}
                                  className="h-8 px-2.5 text-xs gap-1"
                                >
                                  {isSaving ? (
                                    <Loader2 className="h-3 w-3 animate-spin" />
                                  ) : (
                                    <Check className="h-3 w-3" />
                                  )}
                                  Save
                                </Button>
                                <Button
                                  variant="ghost"
                                  size="sm"
                                  onClick={cancelEdit}
                                  disabled={isSaving}
                                  className="h-8 px-2 text-xs"
                                >
                                  <X className="h-3 w-3" />
                                </Button>
                              </div>
                            ) : currentLabel ? (
                              <div className="flex items-center gap-2">
                                <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md text-xs font-semibold bg-emerald-500/10 text-emerald-800 dark:text-emerald-300 border border-emerald-500/20">
                                  <Tag className="h-3 w-3 text-emerald-600 dark:text-emerald-400" />
                                  {currentLabel}
                                </span>
                                <button
                                  type="button"
                                  onClick={() => startEdit(model.id, currentLabel)}
                                  disabled={isSaving}
                                  title="Edit nickname"
                                  className="p-1 rounded text-muted-foreground hover:text-foreground hover:bg-muted cursor-pointer transition-colors"
                                >
                                  <Pencil className="h-3.5 w-3.5" />
                                </button>
                                <button
                                  type="button"
                                  onClick={() => void clearLabel(model.id, model.name)}
                                  disabled={isSaving}
                                  title="Remove nickname"
                                  className="p-1 rounded text-muted-foreground hover:text-destructive hover:bg-destructive/10 cursor-pointer transition-colors"
                                >
                                  <Trash2 className="h-3.5 w-3.5" />
                                </button>
                              </div>
                            ) : (
                              <Button
                                variant="outline"
                                size="sm"
                                onClick={() => startEdit(model.id, "")}
                                className="h-7 text-xs gap-1 border-dashed text-muted-foreground hover:text-foreground"
                              >
                                <Plus className="h-3 w-3" />
                                Set nickname
                              </Button>
                            )}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              )}
            </div>
          </Card>
        )}

        {/* Organization Members Card */}
        {organization && (
          <Card className="overflow-hidden p-0 border border-border shadow-xs">
            <div className="flex items-center justify-between border-b border-border px-5 py-4 bg-muted/10">
              <div className="flex items-center gap-2">
                <Users className="h-4 w-4 text-muted-foreground" />
                <span className="font-medium text-sm text-foreground">Members</span>
              </div>
              <Badge tone="muted">
                {(organization.users || []).length} {organization.users?.length === 1 ? "member" : "members"}
              </Badge>
            </div>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead className="border-b border-border bg-muted/30 text-left">
                  <tr>
                    <th className="px-5 py-3 text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                      Name
                    </th>
                    <th className="px-5 py-3 text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                      Email
                    </th>
                    <th className="px-5 py-3 text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                      Role
                    </th>
                    <th className="px-5 py-3 text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                      Status
                    </th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-border">
                  {(organization.users || []).length === 0 ? (
                    <tr>
                      <td colSpan={4} className="px-5 py-8 text-center text-muted-foreground">
                        No members assigned to this organization.
                      </td>
                    </tr>
                  ) : (
                    (organization.users || []).map((member) => (
                      <tr key={member.id} className="hover:bg-muted/20 transition-colors">
                        <td className="px-5 py-3 font-medium text-foreground">
                          {member.name || "—"}
                        </td>
                        <td className="px-5 py-3 text-muted-foreground text-xs">
                          {member.email}
                        </td>
                        <td className="px-5 py-3">
                          <span className="capitalize text-xs font-medium">
                            {member.role || "Member"}
                          </span>
                        </td>
                        <td className="px-5 py-3">
                          <Badge tone={member.is_active ? "neutral" : "outline"}>
                            {member.is_active ? "Active" : "Inactive"}
                          </Badge>
                        </td>
                      </tr>
                    ))
                  )}
                </tbody>
              </table>
            </div>
          </Card>
        )}
      </div>

      {showLink && organization && (
        <LinkModal
          link={registrationLink}
          organizationName={organization.name}
          copied={copied}
          onCopy={copyRegistrationLink}
          onClose={() => setShowLink(false)}
        />
      )}
    </AppShell>
  );
}

function LinkModal({
  link,
  organizationName,
  copied,
  onCopy,
  onClose,
}: {
  link: string;
  organizationName: string;
  copied: boolean;
  onCopy: () => void;
  onClose: () => void;
}) {
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-foreground/20 backdrop-blur-sm p-4">
      <div className="w-full max-w-md rounded-xl border border-border bg-background shadow-xl">
        <div className="flex items-center justify-between border-b border-border px-6 py-4">
          <div className="font-medium">Registration link</div>
          <button onClick={onClose} className="text-muted-foreground hover:text-foreground">
            <Check className="h-4 w-4 opacity-0" />
          </button>
        </div>
        <div className="space-y-4 p-6">
          <div className="text-xs text-muted-foreground">
            Anyone with this link can register into{" "}
            <b className="text-foreground">{organizationName}</b>.
          </div>
          <div className="flex items-center gap-2 rounded-md border border-border bg-muted/50 px-3 py-2">
            <div className="min-w-0 flex-1 truncate font-mono text-xs">{link}</div>
            <button
              onClick={onCopy}
              className="shrink-0 text-muted-foreground hover:text-foreground"
            >
              {copied ? <Check className="h-3.5 w-3.5" /> : <Copy className="h-3.5 w-3.5" />}
            </button>
          </div>
          <div className="flex justify-end">
            <Button onClick={onClose}>Done</Button>
          </div>
        </div>
      </div>
    </div>
  );
}

