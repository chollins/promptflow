import { createFileRoute, Link, useNavigate } from "@tanstack/react-router";
import { useEffect, useMemo, useState } from "react";
import { Building2, PencilLine, Plus, RefreshCw, Trash2, EyeIcon, X } from "lucide-react";
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

export const Route = createFileRoute("/admin/organizations")({
  component: OrganizationsAdmin,
});

function OrganizationsAdmin() {
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

  return (
    <AppShell>
      <PageHeader
        title="Organizations"
        description="Create, edit, and remove organizations for superadmins."
        actions={
          <>
            <Button variant="secondary" onClick={() => void refresh()} disabled={loading}>
              <RefreshCw className="h-4 w-4" />
              Refresh
            </Button>
            <Button onClick={openCreateModal}>
              <Plus className="h-4 w-4" />
              New organization
            </Button>
          </>
        }
      />

      <Card className="p-5">
        <div className="mb-4 flex items-center gap-2 text-sm font-medium">
          <Building2 className="h-4 w-4" />
          Organization list
        </div>
        {loading ? (
          <div className="text-sm text-muted-foreground">Loading organizations...</div>
        ) : items.length === 0 ? (
          <div className="text-sm text-muted-foreground">No organizations found.</div>
        ) : (
          <div className="space-y-3">
            {items.map((item) => (
              <div
                key={item.id}
                className="rounded-xl border border-border bg-background p-4 transition-colors hover:bg-muted/20"
              >
                <div className="flex items-start justify-between gap-3">
                  <div className="text-left">
                    <div className="flex items-center gap-2">
                      <div className="font-medium">{item.name}</div>
                      <Badge tone={item.is_active ? "neutral" : "muted"}>
                        {item.is_active ? "Active" : "Inactive"}
                      </Badge>
                    </div>
                    <div className="mt-1 text-xs text-muted-foreground">
                      {item.code} · {item.slug}
                    </div>
                  </div>
                  <div className="flex items-center gap-2 shrink-0">
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() =>
                        void navigate({
                          to: "/admin/organizations/$id",
                          params: { id: item.id },
                        })
                      }
                      title="View details"
                    >
                      <EyeIcon className="h-4 w-4" />
                    </Button>
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => openEditModal(item)}
                      title="Edit"
                    >
                      <PencilLine className="h-4 w-4" />
                    </Button>
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => setDeleteId(item.id)}
                      title="Delete"
                    >
                      <Trash2 className="h-4 w-4" />
                    </Button>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </Card>

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
