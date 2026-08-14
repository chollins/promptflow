import { createFileRoute, Link } from "@tanstack/react-router";
import { useEffect, useMemo, useState } from "react";
import { ArrowLeft, Database, Plus, Trash2, Workflow, GripVertical, Edit } from "lucide-react";
import { AppShell, PageHeader } from "@/components/app-shell";
import { Button, Card, Input } from "@/components/ui-kit";
import { Textarea } from "@/components/ui/textarea";
import { apiDelete, apiGet, apiPost, apiPut } from "@/lib/api";
import { toast } from "sonner";

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
  const [stepFormId, setStepFormId] = useState<string>("");
  const [stepRequired, setStepRequired] = useState(true);
  const [selectedDetail, setSelectedDetail] = useState<FlowDetail | null>(null);
  const [draggedIndex, setDraggedIndex] = useState<number | null>(null);

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
      .catch((err: Error) => toast.error(err.message));
  }, [selectedItem]);

  async function handleSave() {
    setSaving(true);
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
      toast.success("Flow saved successfully");
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Failed to save flow");
    } finally {
      setSaving(false);
    }
  }

  async function handleDelete(id: string) {
    if (!window.confirm("Delete this flow?")) return;
    try {
      await apiDelete(`/admin/flows/${id}`);
      await refresh();
      if (selectedId === id) {
        setSelectedId(null);
        setForm(EMPTY_FORM);
      }
      toast.success("Flow deleted");
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Failed to delete flow");
    }
  }

  async function handleAddStep() {
    if (!selectedItem || !stepFormId) return;
    try {
      const nextStepNumber = (selectedDetail?.steps?.length || 0) + 1;
      await apiPost(`/admin/flows/${selectedItem.id}/steps`, {
        form_id: stepFormId,
        step_number: nextStepNumber,
        is_required: stepRequired,
      });
      const updated = await apiGet<FlowDetail>(`/admin/flows/${selectedItem.id}`);
      setSelectedDetail(updated);
      setStepFormId("");
      toast.success("Step added");
      await refresh();
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Failed to add step");
    }
  }

  async function handleRemoveStep(formId: string) {
    if (!selectedItem) return;
    try {
      await apiDelete(`/admin/flows/${selectedItem.id}/steps/${formId}`);
      const updated = await apiGet<FlowDetail>(`/admin/flows/${selectedItem.id}`);
      setSelectedDetail(updated);
      toast.success("Step removed");
      await refresh();
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Failed to remove step");
    }
  }

  const handleDragStart = (e: React.DragEvent, index: number) => {
    setDraggedIndex(index);
    e.dataTransfer.effectAllowed = "move";
    e.dataTransfer.setData("text/plain", index.toString());
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    e.dataTransfer.dropEffect = "move";
  };

  const handleDragEnd = () => {
    setDraggedIndex(null);
  };

  const handleDrop = async (e: React.DragEvent, dropIndex: number) => {
    e.preventDefault();
    if (draggedIndex === null || draggedIndex === dropIndex) {
      setDraggedIndex(null);
      return;
    }

    if (!selectedItem || !selectedDetail || !selectedDetail.steps) return;
    const newSteps = [...selectedDetail.steps];

    // Swap or insert? Insert at position is better for drag and drop
    const item = newSteps.splice(draggedIndex, 1)[0];
    newSteps.splice(dropIndex, 0, item);

    // Optimistic update
    setSelectedDetail({ ...selectedDetail, steps: newSteps });

    try {
      await apiPut(`/admin/flows/${selectedItem.id}/steps/reorder`, {
        steps: newSteps.map(s => ({ form_id: s.form_id, is_required: s.is_required }))
      });
      toast.success("Steps reordered");
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Failed to reorder steps");
      // Revert
      const reverted = await apiGet<FlowDetail>(`/admin/flows/${selectedItem.id}`);
      setSelectedDetail(reverted);
    }
    setDraggedIndex(null);
  };

  if (selectedItem && selectedDetail) {
    return (
      <AppShell>
        <div className="mx-auto w-full max-w-7xl">
          <div className="mb-6">
            <Button variant="ghost" onClick={() => { setSelectedId(null); setForm(EMPTY_FORM); }} className="gap-2 -ml-3 text-muted-foreground hover:text-foreground">
              <ArrowLeft className="h-4 w-4" /> Back to flows
            </Button>
          </div>
          <PageHeader
            title={form.name || "Edit Flow"}
            description={form.slug || "Manage flow metadata and arrange reusable forms into ordered flow steps."}
          />
          <div className="grid gap-8 lg:grid-cols-[1fr_1.5fr] items-start">
            <Card className="p-6 flex flex-col sticky top-4 h-[calc(100vh-14rem)]">
              <div className="flex items-center gap-2 text-sm font-medium shrink-0 mb-4">
                <Database className="h-4 w-4" />
                Edit flow metadata
              </div>
              <div className="grid gap-4 overflow-y-auto flex-1 pr-2 pb-4">
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
                    rows={12}
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
              </div>
              <div className="flex gap-2 shrink-0 pt-4 border-t border-border mt-auto">
                <Button onClick={() => void handleSave()} disabled={saving}>
                  {saving ? "Saving..." : "Update flow"}
                </Button>
              </div>
            </Card>

            <Card className="p-6 space-y-4">
              <div className="text-sm font-medium">Flow steps</div>
              <div className="rounded-lg border border-border p-4 space-y-3 bg-muted/20">
                <div className="grid gap-3 md:grid-cols-[1fr_140px]">
                  <div className="space-y-2">
                    <label className="text-xs font-medium text-foreground/80">Form</label>
                    <select
                      className="h-10 w-full rounded-md border border-input bg-background px-3 text-sm focus:outline-none focus:ring-2 focus:ring-primary"
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
                        className="rounded border-input focus:ring-primary h-4 w-4"
                      />
                      Required
                    </label>
                  </div>
                </div>
                <Button onClick={() => void handleAddStep()} disabled={!stepFormId} variant="secondary" className="w-full sm:w-auto">
                  <Plus className="h-4 w-4 mr-2" />
                  Add to Flow
                </Button>
              </div>

              <div className="overflow-hidden rounded-lg border border-border">
                <table className="min-w-full divide-y divide-border text-sm">
                  <thead className="bg-muted/40">
                    <tr>
                      <th className="px-3 py-3 text-left font-medium text-muted-foreground">Step</th>
                      <th className="px-3 py-3 text-left font-medium text-muted-foreground">Form</th>
                      <th className="px-3 py-3 text-left font-medium text-muted-foreground">Required</th>
                      <th className="px-3 py-3 text-left font-medium text-muted-foreground">Actions</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-border bg-background">
                    {selectedDetail?.steps?.length ? (
                      selectedDetail.steps.map((step, index) => (
                        <tr
                          key={step.form_id}
                          className={`hover:bg-muted/30 transition-colors ${draggedIndex === index ? 'opacity-50 bg-muted' : ''}`}
                          draggable
                          onDragStart={(e) => handleDragStart(e, index)}
                          onDragOver={handleDragOver}
                          onDrop={(e) => handleDrop(e, index)}
                          onDragEnd={handleDragEnd}
                        >
                          <td className="px-3 py-2 text-muted-foreground w-12 text-center cursor-grab active:cursor-grabbing">
                            <GripVertical className="h-4 w-4 inline text-muted-foreground/50 hover:text-foreground" />
                          </td>
                          <td className="px-3 py-2 font-medium">{step.form_name || step.form_slug || step.form_id}</td>
                          <td className="px-3 py-2">{step.is_required ? "Yes" : "No"}</td>
                          <td className="px-3 py-2">
                            <div className="flex items-center gap-1">
                              <Button
                                variant="outline"
                                size="sm"
                                className="h-8 px-2 gap-1 text-xs"
                                asChild
                              >
                                <Link to={`/admin/forms/${step.form_id}`}>
                                  <Edit className="h-3.5 w-3.5" />
                                </Link>
                              </Button>
                              <Button
                                variant="ghost"
                                size="sm"
                                onClick={() => void handleRemoveStep(step.form_id)}
                                className="h-10 w-10 p-0 text-red-500 hover:text-red-600 hover:bg-red-50 ml-1"
                              >
                                <Trash2 className="h-5 w-5" />
                              </Button>
                            </div>
                          </td>
                        </tr>
                      ))
                    ) : (
                      <tr>
                        <td className="px-3 py-8 text-muted-foreground text-center" colSpan={4}>
                          No steps assigned yet.
                        </td>
                      </tr>
                    )}
                  </tbody>
                </table>
              </div>
            </Card>
          </div>
        </div>
      </AppShell>
    );
  }

  return (
    <AppShell>
      <div className="mx-auto w-full max-w-7xl">
        <PageHeader
          title="Flow Composer"
          description="Create flows and arrange reusable forms into ordered flow steps."
        />

        <div className="grid gap-8 lg:grid-cols-[1.5fr_1fr] items-start">
          <Card className="p-6">
            <div className="mb-4 flex items-center gap-2 text-sm font-medium">
              <Workflow className="h-4 w-4" />
              Flows Catalog
            </div>
            {loading ? (
              <div className="text-sm text-muted-foreground">Loading flows...</div>
            ) : items.length === 0 ? (
              <div className="text-sm text-muted-foreground">No flows yet.</div>
            ) : (
              <div className="grid gap-3">
                {items.map((item) => (
                  <div
                    key={item.id}
                    className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 rounded-xl border border-border bg-background p-4 transition-colors hover:bg-muted/40"
                  >
                    <div className="min-w-0 flex-1">
                      <div className="font-medium text-foreground">{item.name}</div>
                      <div className="mt-1 text-xs text-muted-foreground">{item.slug}</div>
                      {item.description && (
                        <div className="mt-2 text-sm text-muted-foreground line-clamp-1">
                          {item.description}
                        </div>
                      )}
                    </div>
                    <div className="flex shrink-0 items-center gap-2">
                      <Button
                        variant="secondary"
                        size="sm"
                        onClick={() => setSelectedId(item.id)}
                      >
                        Edit flow
                      </Button>
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => void handleDelete(item.id)}
                        className="text-red-500 hover:bg-red-50"
                      >
                        <Trash2 className="h-4 w-4" />
                      </Button>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </Card>

          <Card className="p-6 flex flex-col sticky top-4 h-fit">
            <div className="flex items-center gap-2 text-sm font-medium shrink-0 mb-4">
              <Plus className="h-4 w-4" />
              Create new flow
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
              {/* <div className="space-y-2">
                <label className="text-xs font-medium text-foreground/80">Flow JSON</label>
                <Textarea
                  value={form.content_json}
                  onChange={(e) => setForm((prev) => ({ ...prev, content_json: e.target.value }))}
                  rows={8}
                  className="font-mono text-sm"
                />
              </div> */}
              <label className="flex items-center gap-2 text-sm text-foreground">
                <input
                  type="checkbox"
                  checked={form.is_active}
                  onChange={(e) => setForm((prev) => ({ ...prev, is_active: e.target.checked }))}
                />
                Active
              </label>
              <div className="flex gap-2 shrink-0 pt-4 border-t border-border mt-4">
                <Button onClick={() => void handleSave()} disabled={saving}>
                  {saving ? "Creating..." : "Create flow"}
                </Button>
                <Button variant="secondary" onClick={() => setForm(EMPTY_FORM)}>
                  Reset
                </Button>
              </div>
            </div>
          </Card>
        </div>
      </div>
    </AppShell>
  );
}
