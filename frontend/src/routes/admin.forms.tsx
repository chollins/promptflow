import { createFileRoute, useNavigate } from "@tanstack/react-router";
import { useEffect, useMemo, useState } from "react";
import { Database, PencilLine, Plus, Trash2, Upload, FormInput, Eye } from "lucide-react";
import { AppShell, PageHeader } from "@/components/app-shell";
import { Button, Card, Input } from "@/components/ui-kit";
import { Textarea } from "@/components/ui/textarea";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
} from "@/components/ui/dialog";
import { apiDelete, apiGet, apiPost, apiPut } from "@/lib/api";
import { toast } from "sonner";

type FormItem = {
  id: string;
  name: string;
  slug: string;
  description: string | null;
  file_path: string;
  content_json?: string | null;
  is_active: boolean;
  created_at?: string | null;
  updated_at?: string | null;
};

type FormState = {
  name: string;
  description: string;
  content_json: string;
  is_active: boolean;
};

const EMPTY_FORM: FormState = {
  name: "",
  description: "",
  content_json:
    "{\n  \"id\": \"\",\n  \"name\": \"\",\n  \"description\": \"\",\n  \"version\": \"1.0\",\n  \"fields\": [],\n  \"prompt\": {\n    \"system\": \"\",\n    \"user\": \"\"\n  },\n  \"model\": {\n    \"provider\": \"openai\",\n    \"name\": \"gpt-4o-mini\",\n    \"temperature\": 0.7\n  }\n}",
  is_active: true,
};

export const Route = createFileRoute("/admin/forms")({
  component: AdminFormsPage,
});

function AdminFormsPage() {
  const navigate = useNavigate();
  const [items, setItems] = useState<FormItem[]>([]);
  const [editingId, setEditingId] = useState<string | null>(null);
  const [form, setForm] = useState<FormState>(EMPTY_FORM);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [modalOpen, setModalOpen] = useState(false);

  const editingItem = useMemo(
    () => items.find((item) => item.id === editingId) ?? null,
    [items, editingId],
  );

  async function refresh() {
    const data = await apiGet<{ items: FormItem[] }>("/admin/forms");
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
    setEditingId(null);
    setForm(EMPTY_FORM);
    setModalOpen(true);
  }

  function openEditModal(item: FormItem) {
    setEditingId(item.id);
    setModalOpen(true);
    apiGet<FormItem>(`/admin/forms/${item.id}`)
      .then((data) => {
        setForm({
          name: data.name,
          description: data.description || "",
          content_json: data.content_json || "",
          is_active: data.is_active,
        });
      })
      .catch((err: Error) => toast.error(err.message));
  }

  function closeModal() {
    setModalOpen(false);
    setEditingId(null);
    setForm(EMPTY_FORM);
  }

  async function handleUpload(file: File | null) {
    if (!file) return;
    const text = await file.text();
    setForm((prev) => ({ ...prev, content_json: text }));
  }

  async function handleSave() {
    setSaving(true);
    try {
      const payload = {
        name: form.name,
        description: form.description,
        content_json: form.content_json,
        is_active: form.is_active,
      };
      if (editingItem) {
        await apiPut(`/admin/forms/${editingItem.id}`, payload);
      } else {
        await apiPost("/admin/forms", payload);
      }
      await refresh();
      closeModal();
      toast.success("Form saved successfully");
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Failed to save form");
    } finally {
      setSaving(false);
    }
  }

  async function handleDelete(id: string) {
    if (!window.confirm("Delete this form?")) return;
    try {
      await apiDelete(`/admin/forms/${id}`);
      await refresh();
      toast.success("Form deleted");
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Failed to delete form");
    }
  }

  return (
    <AppShell>
      <PageHeader
        title="Forms Catalog"
        description="Create, edit, or delete reusable form definitions stored in the database."
        actions={
          <Button onClick={openCreateModal}>
            <Plus className="h-4 w-4" />
            New form
          </Button>
        }
      />

      <Card className="p-5">
        <div className="mb-4 flex items-center gap-2 text-sm font-medium">
          <FormInput className="h-4 w-4" />
          Form list
        </div>
        {loading ? (
          <div className="text-sm text-muted-foreground">Loading forms...</div>
        ) : items.length === 0 ? (
          <div className="text-sm text-muted-foreground">No forms yet.</div>
        ) : (
          <div className="space-y-3">
            {items.map((item) => (
              <div
                key={item.id}
                className="rounded-xl border border-border bg-background p-4 transition-colors hover:bg-muted/20"
              >
                <div className="flex items-start justify-between gap-3">
                  <div className="text-left">
                    <div className="font-medium">{item.name}</div>
                    <div className="mt-1 text-xs text-muted-foreground">{item.slug}</div>
                    <div className="mt-2 text-sm text-muted-foreground">
                      {item.description || "No description."}
                    </div>
                  </div>
                  <div className="flex items-center gap-2 shrink-0">
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => void navigate({ to: `/admin/forms/${item.id}` })}
                      title="View"
                    >
                      <Eye className="h-4 w-4" />
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
                      onClick={() => void handleDelete(item.id)}
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
        <DialogContent className="w-full max-w-2xl">
          <DialogHeader>
            <DialogTitle>{editingItem ? "Edit form" : "New form"}</DialogTitle>
            <DialogDescription>
              {editingItem
                ? "Update the form definition below."
                : "Fill in the details to create a new form definition."}
            </DialogDescription>
          </DialogHeader>

          <div className="grid gap-4 mt-2 max-h-[70vh] overflow-y-auto pr-1">
            <div className="space-y-2">
              <label className="text-xs font-medium text-foreground/80">Name</label>
              <Input
                value={form.name}
                onChange={(e) => setForm((prev) => ({ ...prev, name: e.target.value }))}
                placeholder="My Form"
              />
            </div>

            <div className="space-y-2">
              <label className="text-xs font-medium text-foreground/80">Description</label>
              <Input
                value={form.description}
                onChange={(e) => setForm((prev) => ({ ...prev, description: e.target.value }))}
                placeholder="Optional description"
              />
            </div>

            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <label className="text-xs font-medium text-foreground/80">Content JSON</label>
                <label className="inline-flex cursor-pointer items-center gap-2 text-xs text-muted-foreground hover:text-foreground">
                  <Upload className="h-4 w-4" />
                  Upload JSON
                  <input
                    type="file"
                    accept=".json,application/json"
                    className="hidden"
                    onChange={(e) => {
                      const file = e.target.files?.[0] ?? null;
                      void handleUpload(file);
                    }}
                  />
                </label>
              </div>
              <Textarea
                value={form.content_json}
                onChange={(e) => setForm((prev) => ({ ...prev, content_json: e.target.value }))}
                rows={12}
                className="font-mono text-sm"
                placeholder="Paste form JSON here"
              />
            </div>

            <label className="flex items-center gap-2 text-sm text-foreground">
              <input
                type="checkbox"
                checked={form.is_active}
                onChange={(e) => setForm((prev) => ({ ...prev, is_active: e.target.checked }))}
              />
              Active
            </label>
          </div>

          <div className="flex gap-2 pt-4 border-t border-border mt-2">
            <Button onClick={() => void handleSave()} disabled={saving}>
              {saving ? "Saving..." : editingItem ? "Update form" : "Create form"}
            </Button>
            <Button variant="secondary" onClick={closeModal} disabled={saving}>
              Cancel
            </Button>
          </div>
        </DialogContent>
      </Dialog>
    </AppShell>
  );
}
