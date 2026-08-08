import { createFileRoute } from "@tanstack/react-router";
import { useEffect, useMemo, useState } from "react";
import { Database, Plus, Trash2, Workflow } from "lucide-react";
import { AppShell, PageHeader } from "@/components/app-shell";
import { Button, Card, Input } from "@/components/ui-kit";
import { Textarea } from "@/components/ui/textarea";
import { apiDelete, apiGet, apiPost, apiPut } from "@/lib/api";

type FlowItem = {
  id: string;
  name: string;
  slug: string;
  description: string | null;
  content_json?: string | null;
  file_path: string;
  is_active: boolean;
};

type FormItem = {
  id: string;
  name: string;
  slug: string;
  description: string | null;
};

type FlowStepItem = {
  flow_id: string;
  form_id: string;
  form_name: string | null;
  form_slug: string | null;
  step_number: number;
  is_required: boolean;
};

type FlowDetail = FlowItem & {
  steps: FlowStepItem[];
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
    "{\n  \"id\": \"\",\n  \"version\": \"1.0\",\n  \"name\": \"\",\n  \"description\": \"\",\n  \"runtime\": { \"mode\": \"guided\", \"default_review_required\": true },\n  \"steps\": []\n}",
  is_active: true,
};

export const Route = createFileRoute("/admin/flows")({
  component: FlowsCatalog,
});

function FlowsCatalog() {
  const [items, setItems] = useState<FlowItem[]>([]);
  const [forms, setForms] = useState<FormItem[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [form, setForm] = useState<FlowFormState>(EMPTY_FORM);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [stepNumber, setStepNumber] = useState<number>(1);
  const [stepFormId, setStepFormId] = useState<string>("");
  const [stepRequired, setStepRequired] = useState(true);
  const [selectedDetail, setSelectedDetail] = useState<FlowDetail | null>(null);

  const selectedItem = useMemo(
    () => items.find((item) => item.id === selectedId) ?? null,
    [items, selectedId],
  );

  async function refresh() {
    const [flowData, formData] = await Promise.all([
      apiGet<{ items: FlowItem[] }>("/admin/flows"),
      apiGet<{ items: FormItem[] }>("/admin/forms"),
    ]);
    setItems(flowData.items);
    setForms(formData.items);
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
      setStepNumber(1);
      setStepFormId("");
      setStepRequired(true);
      setSelectedDetail(null);
      return;
    }
    apiGet<FlowItem>(`/admin/flows/${selectedItem.id}`)
      .then((data) => {
        const withSteps = data as FlowDetail;
        setSelectedDetail(withSteps);
        setForm({
          name: data.name,
          description: data.description || "",
          content_json: data.content_json || "",
          is_active: data.is_active,
        });
      })
      .catch((err: Error) => setError(err.message));
  }, [selectedItem]);

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
        await apiPut(`/admin/flows/${selectedItem.id}`, payload);
      } else {
        await apiPost("/admin/flows", payload);
      }
      await refresh();
      setSelectedId(null);
      setForm(EMPTY_FORM);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to save flow");
    } finally {
      setSaving(false);
    }
  }

  async function handleDelete(id: string) {
    if (!window.confirm("Delete this flow?")) return;
    await apiDelete(`/admin/flows/${id}`);
    await refresh();
    if (selectedId === id) {
      setSelectedId(null);
      setForm(EMPTY_FORM);
    }
  }

  async function handleAddStep() {
    if (!selectedItem || !stepFormId) return;
    await apiPost(`/admin/flows/${selectedItem.id}/steps`, {
      form_id: stepFormId,
      step_number: stepNumber,
      is_required: stepRequired,
    });
    const updated = await apiGet<FlowDetail>(`/admin/flows/${selectedItem.id}`);
    setSelectedDetail(updated);
    await refresh();
  }

  async function handleRemoveStep(formId: string) {
    if (!selectedItem) return;
    await apiDelete(`/admin/flows/${selectedItem.id}/steps/${formId}`);
    const updated = await apiGet<FlowDetail>(`/admin/flows/${selectedItem.id}`);
    setSelectedDetail(updated);
    await refresh();
  }

  return (
    <AppShell>
      <div className="mx-auto w-full max-w-7xl">
        <PageHeader
          title="Flow Composer"
          description="Create flows, then arrange reusable forms into ordered flow steps."
          actions={
            <Button
              onClick={() => {
                setSelectedId(null);
                setForm(EMPTY_FORM);
                setStepNumber(1);
                setStepFormId("");
                setStepRequired(true);
              }}
            >
              <Plus className="h-4 w-4" />
              New flow
            </Button>
          }
        />

        {error && (
          <div className="mb-4 rounded-lg border border-red-200 bg-red-50 p-4 text-sm text-red-700">
            {error}
          </div>
        )}

        <div className="grid gap-8 xl:grid-cols-[0.95fr_1.1fr_1.25fr]">
          <Card className="p-6">
            <div className="mb-4 flex items-center gap-2 text-sm font-medium">
              <Workflow className="h-4 w-4" />
              Flows
            </div>
            {loading ? (
              <div className="text-sm text-muted-foreground">Loading flows...</div>
            ) : items.length === 0 ? (
              <div className="text-sm text-muted-foreground">No flows yet.</div>
            ) : (
              <div className="space-y-3">
                {items.map((item) => (
                  <button
                    key={item.id}
                    type="button"
                    onClick={() => setSelectedId(item.id)}
                    className={
                      "w-full rounded-xl border p-4 text-left transition-colors " +
                      (selectedId === item.id
                        ? "border-foreground bg-muted/40"
                        : "border-border bg-background")
                    }
                  >
                    <div className="flex items-start justify-between gap-2">
                      <div>
                        <div className="font-medium">{item.name}</div>
                        <div className="mt-1 text-xs text-muted-foreground">{item.slug}</div>
                        <div className="mt-2 text-sm text-muted-foreground">
                          {item.description || "No description."}
                        </div>
                      </div>
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={(e) => {
                          e.preventDefault();
                          e.stopPropagation();
                          void handleDelete(item.id);
                        }}
                      >
                        <Trash2 className="h-4 w-4" />
                      </Button>
                    </div>
                  </button>
                ))}
              </div>
            )}
          </Card>

          <Card className="p-6 space-y-4">
            <div className="flex items-center gap-2 text-sm font-medium">
              <Database className="h-4 w-4" />
              {selectedItem ? "Edit flow metadata" : "New flow"}
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
                <label className="text-xs font-medium text-foreground/80">Flow JSON</label>
                <Textarea
                  value={form.content_json}
                  onChange={(e) => setForm((prev) => ({ ...prev, content_json: e.target.value }))}
                  rows={18}
                  className="font-mono text-sm"
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
                  {saving ? "Saving..." : selectedItem ? "Update flow" : "Create flow"}
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

          <Card className="p-6 space-y-4">
            <div className="text-sm font-medium">Flow steps</div>
            {!selectedItem ? (
              <div className="text-sm text-muted-foreground">
                Select a flow to manage its ordered forms.
              </div>
            ) : (
              <>
                <div className="rounded-lg border border-border p-4 space-y-3">
                  <div className="grid gap-3 md:grid-cols-[140px_1fr_140px]">
                    <div className="space-y-2">
                      <label className="text-xs font-medium text-foreground/80">Step #</label>
                      <Input
                        type="number"
                        min={1}
                        value={stepNumber}
                        onChange={(e) => setStepNumber(Number(e.target.value) || 1)}
                      />
                    </div>
                    <div className="space-y-2">
                      <label className="text-xs font-medium text-foreground/80">Form</label>
                      <select
                        className="h-10 w-full rounded-md border border-input bg-background px-3 text-sm"
                        value={stepFormId}
                        onChange={(e) => setStepFormId(e.target.value)}
                      >
                        <option value="">Select a form</option>
                        {forms.map((formItem) => (
                          <option key={formItem.id} value={formItem.id}>
                            {formItem.name}
                          </option>
                        ))}
                      </select>
                    </div>
                    <div className="flex items-end">
                      <label className="flex h-10 items-center gap-2 text-sm">
                        <input
                          type="checkbox"
                          checked={stepRequired}
                          onChange={(e) => setStepRequired(e.target.checked)}
                        />
                        Required
                      </label>
                    </div>
                  </div>
                  <Button onClick={() => void handleAddStep()} disabled={!stepFormId}>
                    Add step
                  </Button>
                </div>

                <div className="overflow-hidden rounded-lg border border-border">
                  <table className="min-w-full divide-y divide-border text-sm">
                    <thead className="bg-muted/40">
                      <tr>
                        <th className="px-3 py-2 text-left font-medium">Step</th>
                        <th className="px-3 py-2 text-left font-medium">Form</th>
                        <th className="px-3 py-2 text-left font-medium">Required</th>
                        <th className="px-3 py-2 text-left font-medium">Actions</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-border">
                      {selectedDetail?.steps?.length ? (
                        selectedDetail.steps.map((step) => (
                          <tr key={step.form_id}>
                            <td className="px-3 py-2">{step.step_number}</td>
                            <td className="px-3 py-2">{step.form_name || step.form_slug || step.form_id}</td>
                            <td className="px-3 py-2">{step.is_required ? "Yes" : "No"}</td>
                            <td className="px-3 py-2">
                              <Button variant="ghost" size="sm" onClick={() => void handleRemoveStep(step.form_id)}>
                                <Trash2 className="h-4 w-4" />
                              </Button>
                            </td>
                          </tr>
                        ))
                      ) : (
                        <tr>
                          <td className="px-3 py-4 text-muted-foreground" colSpan={4}>
                            No steps assigned yet.
                          </td>
                        </tr>
                      )}
                    </tbody>
                  </table>
                </div>
              </>
            )}
          </Card>
        </div>
      </div>
    </AppShell>
  );
}
