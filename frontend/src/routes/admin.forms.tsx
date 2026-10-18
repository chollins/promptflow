import { useNavigate } from "react-router-dom";
import { useEffect, useMemo, useState } from "react";
import { Database, PencilLine, Plus, Trash2, Upload, FormInput, Eye, FileText, Search, X, CheckCircle2, Workflow } from "lucide-react";
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
import { logActivity } from "@/lib/activity";
import { toast } from "sonner";
import { AlertDialog, AlertDialogAction, AlertDialogCancel, AlertDialogContent, AlertDialogDescription, AlertDialogFooter, AlertDialogHeader, AlertDialogTitle } from "@/components/ui/alert-dialog";


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
  flows?: string[];
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

export default function AdminFormsPage() {
  const navigate = useNavigate();
  const [items, setItems] = useState<FormItem[]>([]);
  const [editingId, setEditingId] = useState<string | null>(null);
  const [form, setForm] = useState<FormState>(EMPTY_FORM);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [modalOpen, setModalOpen] = useState(false);
  const [deleteId, setDeleteId] = useState<string | null>(null);

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

  async function handleMarkdownImport(file: File | null) {
    if (!file) return;
    try {
      const text = await file.text();
      const res = await apiPost<{ items: Array<{ id: string; name: string }>; count: number }>(
        "/admin/forms/import-markdown",
        { content: text }
      );
      toast.success(`Imported ${res.count} form(s) from Markdown`);
      await refresh();
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Failed to import Markdown");
    }
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
        logActivity("updated form", payload.name);
      } else {
        await apiPost("/admin/forms", payload);
        logActivity("created form", payload.name);
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

  async function handleDelete() {
    if (!deleteId) return;
    try {
      await apiDelete(`/admin/forms/${deleteId}`);
      await refresh();
      toast.success("Form deleted");
      logActivity("deleted form", deleteId);
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Failed to delete form");
    } finally {
      setDeleteId(null);
    }
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
        (item.description && item.description.toLowerCase().includes(q)) ||
        (item.flows && item.flows.some((flow) => flow.toLowerCase().includes(q)));

      const matchesStatus =
        statusFilter === "all" ||
        (statusFilter === "active" && item.is_active) ||
        (statusFilter === "inactive" && !item.is_active);

      return matchesSearch && matchesStatus;
    });
  }, [items, searchQuery, statusFilter]);

  const activeCount = useMemo(() => items.filter((i) => i.is_active).length, [items]);
  const linkedCount = useMemo(() => items.filter((i) => i.flows && i.flows.length > 0).length, [items]);

  return (
    <AppShell>
      <div className="space-y-6">
        <PageHeader
          title="Forms Catalog"
          description="Create, edit, and manage reusable prompt form definitions stored in the database."
          actions={
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
              <Button onClick={openCreateModal} className="gap-1.5 bg-foreground text-background hover:bg-foreground/90">
                <Plus className="h-4 w-4" />
                New form
              </Button>
            </div>
          }
        />

        {/* Stats Summary Bar */}
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
          <Card className="p-4 bg-card border-border shadow-xs">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-xs font-semibold text-muted-foreground uppercase tracking-wider">Total Forms</p>
                <p className="text-2xl font-bold text-foreground mt-1">{items.length}</p>
              </div>
              <div className="p-2.5 bg-muted text-foreground rounded-lg border border-border">
                <FormInput className="w-5 h-5" />
              </div>
            </div>
          </Card>

          <Card className="p-4 bg-card border-border shadow-xs">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-xs font-semibold text-muted-foreground uppercase tracking-wider">Active Status</p>
                <p className="text-2xl font-bold text-foreground mt-1">{activeCount} / {items.length}</p>
              </div>
              <div className="p-2.5 bg-muted text-foreground rounded-lg border border-border">
                <CheckCircle2 className="w-5 h-5 text-emerald-600 dark:text-emerald-400" />
              </div>
            </div>
          </Card>

          <Card className="p-4 bg-card border-border shadow-xs">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-xs font-semibold text-muted-foreground uppercase tracking-wider">Linked to Flows</p>
                <p className="text-2xl font-bold text-foreground mt-1">{linkedCount}</p>
              </div>
              <div className="p-2.5 bg-muted text-foreground rounded-lg border border-border">
                <Workflow className="w-5 h-5 text-muted-foreground" />
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
                placeholder="Search forms by name, slug, description, or flow..."
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
                  className={`px-3 py-1.5 rounded-md font-medium capitalize transition-all cursor-pointer ${statusFilter === st
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

        {/* Forms Table (Mobile-Responsive) */}
        <Card className="border-border bg-card overflow-hidden shadow-xs">
          {loading ? (
            <div className="p-12 text-center text-muted-foreground flex flex-col items-center gap-3">
              <span className="h-6 w-6 animate-spin rounded-full border-2 border-foreground border-t-transparent" />
              <span>Loading forms...</span>
            </div>
          ) : filteredItems.length === 0 ? (
            <div className="p-12 text-center text-muted-foreground">
              <FormInput className="h-10 w-10 mx-auto mb-3 text-muted-foreground opacity-50" />
              <p className="font-semibold text-foreground">No forms found</p>
              <p className="text-xs text-muted-foreground mt-1 mb-4">
                {searchQuery || statusFilter !== "all"
                  ? "Try adjusting your search query or filters."
                  : "Get started by creating a new form definition or importing from Markdown."}
              </p>
              {!searchQuery && statusFilter === "all" && (
                <Button onClick={openCreateModal} className="gap-1.5 text-xs inline-flex bg-foreground text-background hover:bg-foreground/90">
                  <Plus className="h-4 w-4" /> New form
                </Button>
              )}
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-left text-sm text-foreground min-w-[700px]">
                <thead className="bg-muted/50 text-xs font-semibold uppercase tracking-wider text-muted-foreground border-b border-border">
                  <tr>
                    <th className="px-6 py-3.5">Form Name & ID</th>
                    <th className="px-6 py-3.5">Linked Flows</th>
                    <th className="px-6 py-3.5">Status</th>
                    <th className="px-6 py-3.5 text-right">Actions</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-border">
                  {filteredItems.map((item) => (
                    <tr key={item.id} className="hover:bg-muted/40 transition-colors">
                      {/* Name & Slug */}
                      <td className="px-6 py-4">
                        <div className="flex items-start gap-3">
                          <div className="p-2.5 rounded-lg bg-muted border border-border text-foreground shrink-0 mt-0.5">
                            <FormInput className="w-4 h-4" />
                          </div>
                          <div className="min-w-0 max-w-sm sm:max-w-md">
                            <div className="font-semibold text-foreground truncate">{item.name}</div>
                            <div className="text-xs text-muted-foreground font-mono truncate">{item.slug}</div>
                            {item.description && (
                              <div className="text-xs text-muted-foreground line-clamp-1 mt-1">
                                {item.description}
                              </div>
                            )}
                          </div>
                        </div>
                      </td>

                      {/* Linked Flows */}
                      <td className="px-6 py-4">
                        {item.flows && item.flows.length > 0 ? (
                          <div className="flex flex-wrap gap-1.5 max-w-xs">
                            {item.flows.map((flowName) => (
                              <span
                                key={flowName}
                                className="inline-flex items-center gap-1 rounded-md bg-muted px-2 py-0.5 text-xs font-medium text-foreground border border-border"
                              >
                                <Workflow className="h-3 w-3 text-muted-foreground" />
                                <span>{flowName}</span>
                              </span>
                            ))}
                          </div>
                        ) : (
                          <span className="text-xs text-muted-foreground italic">Standalone (not in flow)</span>
                        )}
                      </td>

                      {/* Status */}
                      <td className="px-6 py-4">
                        <span
                          className={`inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-md text-xs font-medium border ${item.is_active
                            ? "bg-zinc-100 text-zinc-900 border-zinc-200 dark:bg-zinc-800 dark:text-zinc-100 dark:border-zinc-700"
                            : "bg-muted text-muted-foreground border-border"
                            }`}
                        >
                          <span
                            className={`w-1.5 h-1.5 rounded-full ${item.is_active ? "bg-emerald-500" : "bg-muted-foreground/50"
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
                            onClick={() => void navigate(`/admin/forms/${item.id}`)}
                            className=" text-foreground hover:bg-muted"
                            title="View form builder"
                          >
                            <Eye className="h-4 w-4" />
                          </Button>
                          <Button
                            variant="outline"
                            size="sm"
                            onClick={() => openEditModal(item)}
                            className=" text-foreground hover:bg-muted"
                            title="Edit form"
                          >
                            <PencilLine className="h-4 w-4" />
                          </Button>
                          <Button
                            variant="outline"
                            size="sm"
                            onClick={() => setDeleteId(item.id)}
                            className=" text-red-700 hover:text-destructive hover:bg-destructive/10"
                            title="Delete form"
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
                <div className="flex items-center gap-3">
                  <label className="inline-flex cursor-pointer items-center gap-1.5 text-xs text-muted-foreground hover:text-foreground">
                    <Upload className="h-3.5 w-3.5" />
                    Upload JSON
                    <input
                      type="file"
                      accept=".json,application/json"
                      className="hidden"
                      onChange={(e) => {
                        const file = e.target.files?.[0] ?? null;
                        void handleUpload(file);
                        e.target.value = "";
                      }}
                    />
                  </label>
                  <label className="inline-flex cursor-pointer items-center gap-1.5 text-xs text-muted-foreground hover:text-foreground">
                    <FileText className="h-3.5 w-3.5" />
                    Upload Markdown
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
                </div>
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
      <AlertDialog open={!!deleteId} onOpenChange={(open) => !open && setDeleteId(null)}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Delete this form?</AlertDialogTitle>
            <AlertDialogDescription>This action cannot be undone.</AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction onClick={() => void handleDelete()} className="bg-destructive text-destructive-foreground hover:bg-destructive/90">
              Delete
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </AppShell>
  );
}

