import { Link, useNavigate } from "react-router-dom";
import { useEffect, useMemo, useState } from "react";
import { Building2, PencilLine, Plus, RefreshCw, Trash2, EyeIcon, X, Search, CheckCircle2 } from "lucide-react";
import { AppShell, PageHeader } from "@/components/app-shell";
import { Badge, Button, Card, Field, Input } from "@/components/ui-kit";
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
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
} from "@/components/ui/dialog";
import { apiDelete, apiGet, apiPost, apiPut } from "@/lib/api";

type OrganizationItem = {
  id: string;
  name: string;
  slug: string;
  code: string;
  is_active: boolean;
};

type OrganizationFormState = {
  name: string;
  slug: string;
  code: string;
  is_active: boolean;
  admin_name: string;
  admin_email: string;
  admin_password: string;
  admin_confirm_password: string;
};

const EMPTY_FORM: OrganizationFormState = {
  name: "",
  slug: "",
  code: "",
  is_active: true,
  admin_name: "",
  admin_email: "",
  admin_password: "",
  admin_confirm_password: "",
};

function slugify(value: string) {
  return value
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "");
}

function codeify(value: string) {
  return value
    .trim()
    .toUpperCase()
    .replace(/[^A-Z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "");
}

export default function OrganizationsAdmin() {
  const navigate = useNavigate();
  const [items, setItems] = useState<OrganizationItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [deleteId, setDeleteId] = useState<string | null>(null);

  // Modal state
  const [modalOpen, setModalOpen] = useState(false);
  const [editingId, setEditingId] = useState<string | null>(null);
  const [form, setForm] = useState<OrganizationFormState>(EMPTY_FORM);

  const editingItem = useMemo(
    () => items.find((item) => item.id === editingId) ?? null,
    [items, editingId],
  );

  async function refresh() {
    const data = await apiGet<{ items: OrganizationItem[] }>("/admin/organizations");
    setItems(data.items);
  }

  useEffect(() => {
    let active = true;
    setLoading(true);
    refresh()
      .catch(() => active && setItems([]))
      .finally(() => active && setLoading(false));
    return () => {
      active = false;
    };
  }, []);

  function openCreateModal() {
    setError(null);
    setEditingId(null);
    setForm(EMPTY_FORM);
    setModalOpen(true);
  }

  function openEditModal(item: OrganizationItem) {
    setError(null);
    setEditingId(item.id);
    setForm({
      name: item.name,
      slug: item.slug,
      code: item.code,
      is_active: item.is_active,
      admin_name: "",
      admin_email: "",
      admin_password: "",
      admin_confirm_password: "",
    });
    setModalOpen(true);
  }

  function closeModal() {
    setModalOpen(false);
    setEditingId(null);
    setForm(EMPTY_FORM);
    setError(null);
  }

  async function handleSave() {
    setSaving(true);
    setError(null);
    try {
      const payload = {
        name: form.name.trim(),
        slug: slugify(form.slug || form.name),
        code: codeify(form.code || form.name),
        is_active: form.is_active,
      };

      if (!payload.name) {
        throw new Error("Name is required.");
      }

      if (editingId) {
        await apiPut(`/admin/organizations/${editingId}`, payload);
      } else {
        if (!form.admin_name.trim()) {
          throw new Error("Admin name is required.");
        }
        if (!form.admin_email.trim()) {
          throw new Error("Admin email is required.");
        }
        if (!form.admin_password) {
          throw new Error("Admin password is required.");
        }
        if (form.admin_password !== form.admin_confirm_password) {
          throw new Error("Admin passwords do not match.");
        }

        await apiPost("/admin/organizations", {
          ...payload,
          admin: {
            name: form.admin_name.trim(),
            email: form.admin_email.trim(),
            password: form.admin_password,
            role: "admin",
          },
        });
      }

      await refresh();
      closeModal();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to save organization");
    } finally {
      setSaving(false);
    }
  }

  async function handleDelete(id: string) {
    await apiDelete(`/admin/organizations/${id}`);
    await refresh();
  }

  const [searchQuery, setSearchQuery] = useState("");
  const [statusFilter, setStatusFilter] = useState<"all" | "active" | "inactive">("all");

  const filteredItems = useMemo(() => {
    return items.filter((item) => {
      const q = searchQuery.toLowerCase().trim();
      const matchesSearch =
        !q ||
        item.name.toLowerCase().includes(q) ||
        item.slug.toLowerCase().includes(q) ||
        item.code.toLowerCase().includes(q);

      const matchesStatus =
        statusFilter === "all" ||
        (statusFilter === "active" && item.is_active) ||
        (statusFilter === "inactive" && !item.is_active);

      return matchesSearch && matchesStatus;
    });
  }, [items, searchQuery, statusFilter]);

  const activeCount = useMemo(() => items.filter((i) => i.is_active).length, [items]);
  const inactiveCount = items.length - activeCount;

  return (
    <AppShell>
      <div className="space-y-6">
        <PageHeader
          title="Organizations"
          description="Create, edit, and manage organizations across the system."
          actions={
            <div className="flex items-center gap-2">
              <Button variant="secondary" onClick={() => void refresh()} disabled={loading} className="gap-1.5">
                <RefreshCw className="h-4 w-4" />
                Refresh
              </Button>
              <Button onClick={openCreateModal} className="gap-1.5 bg-foreground text-background hover:bg-foreground/90">
                <Plus className="h-4 w-4" />
                New organization
              </Button>
            </div>
          }
        />

        {/* Stats Summary Bar */}
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
          <Card className="p-4 bg-card border-border shadow-xs">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-xs font-semibold text-muted-foreground uppercase tracking-wider">Total Organizations</p>
                <p className="text-2xl font-bold text-foreground mt-1">{items.length}</p>
              </div>
              <div className="p-2.5 bg-muted text-foreground rounded-lg border border-border">
                <Building2 className="w-5 h-5" />
              </div>
            </div>
          </Card>

          <Card className="p-4 bg-card border-border shadow-xs">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-xs font-semibold text-muted-foreground uppercase tracking-wider">Active</p>
                <p className="text-2xl font-bold text-foreground mt-1">{activeCount}</p>
              </div>
              <div className="p-2.5 bg-muted text-foreground rounded-lg border border-border">
                <CheckCircle2 className="w-5 h-5 text-emerald-600 dark:text-emerald-400" />
              </div>
            </div>
          </Card>

          <Card className="p-4 bg-card border-border shadow-xs">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-xs font-semibold text-muted-foreground uppercase tracking-wider">Inactive</p>
                <p className="text-2xl font-bold text-foreground mt-1">{inactiveCount}</p>
              </div>
              <div className="p-2.5 bg-muted text-foreground rounded-lg border border-border">
                <X className="w-5 h-5 text-muted-foreground" />
              </div>
            </div>
          </Card>
        </div>

        {/* Search & Filters */}
        <Card className="p-4 bg-card border-border shadow-xs">
          <div className="flex flex-col sm:flex-row items-stretch sm:items-center justify-between gap-3">
            <div className="relative flex-1 max-w-md">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
              <Input
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                placeholder="Search organizations by name, code, slug..."
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

            {/* Status Tabs */}
            <div className="flex items-center gap-1 bg-muted/60 p-1 rounded-lg border border-border text-xs shrink-0">
              <span className="text-muted-foreground px-2 font-medium">Status:</span>
              {(["all", "active", "inactive"] as const).map((st) => (
                <button
                  key={st}
                  onClick={() => setStatusFilter(st)}
                  className={`px-3 py-1.5 rounded-md font-medium capitalize transition-all cursor-pointer ${
                    statusFilter === st
                      ? "bg-foreground text-background shadow-xs"
                      : "text-muted-foreground hover:text-foreground hover:bg-muted"
                  }`}
                >
                  {st}
                </button>
              ))}
            </div>
          </div>
        </Card>

        {/* Organizations Table (Mobile-Responsive) */}
        <Card className="border-border bg-card overflow-hidden shadow-xs">
          {loading ? (
            <div className="p-12 text-center text-muted-foreground flex flex-col items-center gap-3">
              <span className="h-6 w-6 animate-spin rounded-full border-2 border-foreground border-t-transparent" />
              <span>Loading organizations...</span>
            </div>
          ) : filteredItems.length === 0 ? (
            <div className="p-12 text-center text-muted-foreground">
              <Building2 className="h-10 w-10 mx-auto mb-3 text-muted-foreground opacity-50" />
              <p className="font-semibold text-foreground">No organizations found</p>
              <p className="text-xs text-muted-foreground mt-1 mb-4">
                {searchQuery || statusFilter !== "all"
                  ? "Try adjusting your search query or status filter."
                  : "Get started by creating your first organization."}
              </p>
              {!searchQuery && statusFilter === "all" && (
                <Button onClick={openCreateModal} className="gap-1.5 text-xs inline-flex bg-foreground text-background hover:bg-foreground/90">
                  <Plus className="h-4 w-4" /> New organization
                </Button>
              )}
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-left text-sm text-foreground min-w-[640px]">
                <thead className="bg-muted/50 text-xs font-semibold uppercase tracking-wider text-muted-foreground border-b border-border">
                  <tr>
                    <th className="px-6 py-3.5">Organization</th>
                    <th className="px-6 py-3.5">Code</th>
                    <th className="px-6 py-3.5">Status</th>
                    <th className="px-6 py-3.5 text-right">Actions</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-border">
                  {filteredItems.map((item) => (
                    <tr key={item.id} className="hover:bg-muted/40 transition-colors">
                      {/* Organization Name & Slug */}
                      <td className="px-6 py-4">
                        <div className="flex items-center gap-3">
                          <div className="p-2.5 rounded-lg bg-muted border border-border text-foreground shrink-0">
                            <Building2 className="w-4 h-4" />
                          </div>
                          <div className="min-w-0">
                            <div className="font-semibold text-foreground truncate">{item.name}</div>
                            <div className="text-xs text-muted-foreground font-mono truncate">{item.slug}</div>
                          </div>
                        </div>
                      </td>

                      {/* Code */}
                      <td className="px-6 py-4">
                        <span className="inline-flex items-center px-2.5 py-0.5 rounded-md font-mono text-xs font-semibold bg-muted text-foreground border border-border">
                          {item.code}
                        </span>
                      </td>

                      {/* Status */}
                      <td className="px-6 py-4">
                        <span
                          className={`inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-md text-xs font-medium border ${
                            item.is_active
                              ? "bg-zinc-100 text-zinc-900 border-zinc-200 dark:bg-zinc-800 dark:text-zinc-100 dark:border-zinc-700"
                              : "bg-muted text-muted-foreground border-border"
                          }`}
                        >
                          <span
                            className={`w-1.5 h-1.5 rounded-full ${
                              item.is_active ? "bg-emerald-500" : "bg-muted-foreground/50"
                            }`}
                          />
                          {item.is_active ? "Active" : "Inactive"}
                        </span>
                      </td>

                      {/* Actions */}
                      <td className="px-6 py-4 text-right">
                        <div className="flex items-center justify-end gap-1.5">
                          <Button
                            variant="outline"
                            size="sm"
                            onClick={() => void navigate(`/admin/organizations/${item.id}`)}
                            className="h-8 w-8 p-0 border-border text-foreground hover:bg-muted"
                            title="View details"
                          >
                            <EyeIcon className="h-4 w-4" />
                          </Button>
                          <Button
                            variant="outline"
                            size="sm"
                            onClick={() => openEditModal(item)}
                            className="h-8 w-8 p-0 border-border text-foreground hover:bg-muted"
                            title="Edit organization"
                          >
                            <PencilLine className="h-4 w-4" />
                          </Button>
                          <Button
                            variant="outline"
                            size="sm"
                            onClick={() => setDeleteId(item.id)}
                            className="h-8 w-8 p-0 border-border text-muted-foreground hover:text-destructive hover:bg-destructive/10"
                            title="Delete organization"
                          >
                            <Trash2 className="h-4 w-4" />
                          </Button>
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </Card>
      </div>

      {/* Create / Edit Modal */}
      <Dialog open={modalOpen} onOpenChange={(open) => { if (!open) closeModal(); }}>
        <DialogContent className="w-full max-w-lg">
          <DialogHeader>
            <DialogTitle>
              {editingItem ? `Edit ${editingItem.name}` : "New organization"}
            </DialogTitle>
            <DialogDescription>
              {editingItem
                ? "Update the organization's details below."
                : "Fill in the details to create a new organization."}
            </DialogDescription>
          </DialogHeader>

          {error && (
            <div className="rounded-lg border border-red-200 bg-red-50 p-3 text-sm text-red-700">
              {error}
            </div>
          )}

          <div className="grid gap-4 mt-2">
            <Field
              label="Name"
              hint="This drives the instant slug and code unless you override them."
            >
              <Input
                value={form.name}
                onChange={(e) => {
                  const name = e.target.value;
                  setForm((prev) => ({
                    ...prev,
                    name,
                    slug: prev.slug ? prev.slug : slugify(name),
                    code: prev.code ? prev.code : codeify(name),
                  }));
                }}
                placeholder="Acme Corporation"
              />
            </Field>

            <Field label="Slug" hint="Used in URLs and should stay unique.">
              <Input
                value={form.slug}
                onChange={(e) => setForm((prev) => ({ ...prev, slug: slugify(e.target.value) }))}
                placeholder="acme-corporation"
              />
            </Field>

            <Field label="Code" hint="A short uppercase identifier for internal use.">
              <Input
                value={form.code}
                onChange={(e) => setForm((prev) => ({ ...prev, code: codeify(e.target.value) }))}
                placeholder="ACME"
              />
            </Field>

            <label className="flex items-center gap-2 text-sm text-foreground">
              <input
                type="checkbox"
                checked={form.is_active}
                onChange={(e) => setForm((prev) => ({ ...prev, is_active: e.target.checked }))}
              />
              Active
            </label>

            {!editingId && (
              <div className="space-y-4 rounded-xl border border-border bg-muted/20 p-4">
                <div>
                  <div className="text-sm font-medium">First admin</div>
                  <p className="mt-1 text-xs text-muted-foreground">
                    These details create the first user account for this organization.
                  </p>
                </div>

                <Field label="Admin Name" hint="The person who will manage this organization.">
                  <Input
                    value={form.admin_name}
                    onChange={(e) => setForm((prev) => ({ ...prev, admin_name: e.target.value }))}
                    placeholder="Jane Doe"
                  />
                </Field>

                <Field label="Admin Email" hint="Used for login and account recovery.">
                  <Input
                    type="email"
                    value={form.admin_email}
                    onChange={(e) => setForm((prev) => ({ ...prev, admin_email: e.target.value }))}
                    placeholder="jane@company.com"
                  />
                </Field>

                <div className="grid gap-4 md:grid-cols-2">
                  <Field label="Admin Password">
                    <Input
                      type="password"
                      value={form.admin_password}
                      onChange={(e) =>
                        setForm((prev) => ({ ...prev, admin_password: e.target.value }))
                      }
                    />
                  </Field>
                  <Field label="Confirm Password">
                    <Input
                      type="password"
                      value={form.admin_confirm_password}
                      onChange={(e) =>
                        setForm((prev) => ({
                          ...prev,
                          admin_confirm_password: e.target.value,
                        }))
                      }
                    />
                  </Field>
                </div>
              </div>
            )}

            <div className="flex gap-2 pt-2">
              <Button onClick={() => void handleSave()} disabled={saving}>
                {saving ? "Saving..." : editingId ? "Update organization" : "Create organization"}
              </Button>
              <Button variant="secondary" onClick={closeModal} disabled={saving}>
                Cancel
              </Button>
            </div>
          </div>
        </DialogContent>
      </Dialog>

      {/* Delete Confirmation */}
      <AlertDialog open={Boolean(deleteId)} onOpenChange={(open) => !open && setDeleteId(null)}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Delete organization?</AlertDialogTitle>
            <AlertDialogDescription>
              This will permanently remove the organization and its admin view. This cannot be
              undone.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel onClick={() => setDeleteId(null)}>Cancel</AlertDialogCancel>
            <AlertDialogAction
              onClick={() => {
                if (!deleteId) return;
                void handleDelete(deleteId).finally(() => setDeleteId(null));
              }}
            >
              Delete
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </AppShell>
  );
}

