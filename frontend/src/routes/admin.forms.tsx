import { createFileRoute } from "@tanstack/react-router";
import { useEffect, useMemo, useState } from "react";
import { Database, PencilLine, Plus, Trash2, Upload, FormInput } from "lucide-react";
import { AppShell, PageHeader } from "@/components/app-shell";
import { Button, Card, Input } from "@/components/ui-kit";
import { Textarea } from "@/components/ui/textarea";
import { apiDelete, apiGet, apiPost, apiPut } from "@/lib/api";

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
  const [items, setItems] = useState<FormItem[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [form, setForm] = useState<FormState>(EMPTY_FORM);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const selectedItem = useMemo(
    () => items.find((item) => item.id === selectedId) ?? null,
    [items, selectedId],
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

  useEffect(() => {
    if (!selectedItem) {
      setForm(EMPTY_FORM);
      return;
    }
    apiGet<FormItem>(`/admin/forms/${selectedItem.id}`)
      .then((data) => {
        setForm({
          name: data.name,
          description: data.description || "",
          content_json: data.content_json || "",
          is_active: data.is_active,
        });
      })
      .catch((err: Error) => setError(err.message));
  }, [selectedItem]);

  async function handleUpload(file: File | null) {
    if (!file) return;
    const text = await file.text();
    setForm((prev) => ({ ...prev, content_json: text }));
  }

  async function handleSave() {
    setSaving(true);
    setError(null);
    try {
      const payload = {
        name: form.name,
        description: form.description,
        content_json: form.content_json,
        is_active: form.is_active,
      };
      if (selectedItem) {
        await apiPut(`/admin/forms/${selectedItem.id}`, payload);
      } else {
        await apiPost("/admin/forms", payload);
      }
      await refresh();
      setSelectedId(null);
      setForm(EMPTY_FORM);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to save form");
    } finally {
      setSaving(false);
    }
  }

  async function handleDelete(id: string) {
    if (!window.confirm("Delete this form?")) return;
    await apiDelete(`/admin/forms/${id}`);
    await refresh();
    if (selectedId === id) {
      setSelectedId(null);
      setForm(EMPTY_FORM);
    }
  }

  return (
    <AppShell>
      <PageHeader
        title="Forms Catalog"
        description="Create, edit, or delete reusable form definitions stored in the database."
        actions={
          <Button
            onClick={() => {
              setSelectedId(null);
              setForm(EMPTY_FORM);
            }}
          >
            <Plus className="h-4 w-4" />
            New form
          </Button>
        }
      />

      {error && (
        <div className="mb-4 rounded-lg border border-red-200 bg-red-50 p-4 text-sm text-red-700">
          {error}
        </div>
      )}

      <div className="grid gap-6 lg:grid-cols-[1fr_1.1fr]">
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
                  className={
                    "rounded-xl border p-4 transition-colors " +
                    (selectedId === item.id ? "border-foreground bg-muted/40" : "border-border bg-background")
                  }
                >
                  <div className="flex items-start justify-between gap-3">
                    <button type="button" onClick={() => setSelectedId(item.id)} className="text-left">
                      <div className="font-medium">{item.name}</div>
                      <div className="mt-1 text-xs text-muted-foreground">{item.slug}</div>
                      <div className="mt-2 text-sm text-muted-foreground">
                        {item.description || "No description."}
                      </div>
                    </button>
                    <div className="flex items-center gap-2">
                      <Button variant="ghost" size="sm" onClick={() => setSelectedId(item.id)}>
                        <PencilLine className="h-4 w-4" />
                      </Button>
                      <Button variant="ghost" size="sm" onClick={() => handleDelete(item.id)}>
                        <Trash2 className="h-4 w-4" />
                      </Button>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </Card>

        <Card className="p-5 space-y-4">
          <div className="flex items-center gap-2 text-sm font-medium">
            <Database className="h-4 w-4" />
            {selectedItem ? "Edit form" : "New form"}
          </div>

          <div className="grid gap-4">
            <div className="space-y-2">
              <label className="text-xs font-medium text-foreground/80">Name</label>
              <Input
                value={form.name}
                onChange={(e) => setForm((prev) => ({ ...prev, name: e.target.value }))}
              />
            </div>

            <div className="space-y-2">
              <label className="text-xs font-medium text-foreground/80">Description</label>
              <Input
                value={form.description}
                onChange={(e) => setForm((prev) => ({ ...prev, description: e.target.value }))}
              />
            </div>

            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <label className="text-xs font-medium text-foreground/80">Content JSON</label>
                <label className="inline-flex cursor-pointer items-center gap-2 text-xs text-muted-foreground">
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
                rows={18}
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

            <div className="flex gap-2">
              <Button onClick={() => void handleSave()} disabled={saving}>
                {saving ? "Saving..." : selectedItem ? "Update form" : "Create form"}
              </Button>
              <Button
                variant="secondary"
                onClick={() => {
                  setSelectedId(null);
                  setForm(EMPTY_FORM);
                }}
              >
                Reset
              </Button>
            </div>
          </div>
        </Card>
      </div>
    </AppShell>
  );
}
