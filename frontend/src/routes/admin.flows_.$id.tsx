import { useNavigate, useParams } from "react-router-dom";
import { useEffect, useState } from "react";
import { ArrowLeft, Database, Edit, GripVertical, Plus, Trash2, Upload } from "lucide-react";
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

type FormItem = { id: string; name: string; slug: string };

type FlowStepItem = {
  flow_id: string;
  form_id: string;
  form_name: string | null;
  form_slug: string | null;
  step_number: number;
  is_required: boolean;
};

type FlowDetail = {
  id: string;
  name: string;
  slug: string;
  description: string | null;
  content_json?: string | null;
  is_active: boolean;
  steps: FlowStepItem[];
};

export default function EditFlowPage() {
  const { id } = useParams();
  const navigate = useNavigate();

  const [detail, setDetail] = useState<FlowDetail | null>(null);
  const [forms, setForms] = useState<FormItem[]>([]);
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [contentJson, setContentJson] = useState("");
  const [isActive, setIsActive] = useState(true);
  const [saving, setSaving] = useState(false);
  const [stepFormId, setStepFormId] = useState("");
  const [stepRequired, setStepRequired] = useState(true);
  const [draggedIndex, setDraggedIndex] = useState<number | null>(null);

  const [modalOpen, setModalOpen] = useState(false);
  const [newFormName, setNewFormName] = useState("");
  const [newFormDesc, setNewFormDesc] = useState("");
  const [newFormJson, setNewFormJson] = useState("{\n  \"id\": \"\",\n  \"name\": \"\",\n  \"description\": \"\",\n  \"version\": \"1.0\",\n  \"fields\": [],\n  \"prompt\": {\n    \"system\": \"\",\n    \"user\": \"\"\n  },\n  \"model\": {\n    \"provider\": \"openai\",\n    \"name\": \"gpt-4o-mini\",\n    \"temperature\": 0.7\n  }\n}");
  const [newFormActive, setNewFormActive] = useState(true);
  const [creatingForm, setCreatingForm] = useState(false);

  async function loadDetail() {
    const [data, formData] = await Promise.all([
      apiGet<FlowDetail>(`/admin/flows/${id}`),
      apiGet<{ items: FormItem[] }>("/admin/forms"),
    ]);
    setDetail(data);
    setName(data.name);
    setDescription(data.description || "");
    setContentJson(data.content_json || "");
    setIsActive(data.is_active);
    setForms(formData.items);
  }

  useEffect(() => {
    loadDetail().catch((err: Error) => toast.error(err.message));
  }, [id]);

  async function handleSave() {
    setSaving(true);
    try {
      await apiPut(`/admin/flows/${id}`, {
        name,
        description,
        content_json: contentJson,
        is_active: isActive,
      });
      toast.success("Flow updated");
      logActivity("updated flow", name);
      await loadDetail();
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Failed to save");
    } finally {
      setSaving(false);
    }
  }

  async function handleAddStep() {
    if (!stepFormId || !detail) return;
    try {
      await apiPost(`/admin/flows/${id}/steps`, {
        form_id: stepFormId,
        step_number: (detail.steps?.length || 0) + 1,
        is_required: stepRequired,
      });
      setStepFormId("");
      toast.success("Step added");
      logActivity("added form to flow", `${detail.name} - ${stepFormId}`);
      await loadDetail();
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Failed to add step");
    }
  }

  async function handleRemoveStep(formId: string) {
    try {
      await apiDelete(`/admin/flows/${id}/steps/${formId}`);
      toast.success("Step removed");
      logActivity("removed form from flow", `${detail.name} - ${formId}`);
      await loadDetail();
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Failed to remove step");
    }
  }

  async function handleUpload(file: File | null) {
    if (!file) return;
    const text = await file.text();
    setNewFormJson(text);
  }

  async function handleCreateForm() {
    setCreatingForm(true);
    try {
      const payload = {
        name: newFormName,
        description: newFormDesc,
        content_json: newFormJson,
        is_active: newFormActive,
      };
      const data = await apiPost<{ item: FormItem }>("/admin/forms", payload);
      logActivity("created form", payload.name);
      await loadDetail();
      setStepFormId(data.item.id);
      setModalOpen(false);
      setNewFormName("");
      setNewFormDesc("");
      toast.success("Form created and selected");
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Failed to create form");
    } finally {
      setCreatingForm(false);
    }
  }

  const handleDragStart = (e: React.DragEvent, index: number) => {
    setDraggedIndex(index);
    e.dataTransfer.effectAllowed = "move";
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    e.dataTransfer.dropEffect = "move";
  };

  const handleDrop = async (e: React.DragEvent, dropIndex: number) => {
    e.preventDefault();
    if (draggedIndex === null || draggedIndex === dropIndex || !detail) {
      setDraggedIndex(null);
      return;
    }
    const newSteps = [...detail.steps];
    const [moved] = newSteps.splice(draggedIndex, 1);
    newSteps.splice(dropIndex, 0, moved);
    setDetail({ ...detail, steps: newSteps });
    setDraggedIndex(null);
    try {
      await apiPut(`/admin/flows/${id}/steps/reorder`, {
        steps: newSteps.map((s) => ({ form_id: s.form_id, is_required: s.is_required })),
      });
      toast.success("Steps reordered");
      logActivity("reordered flow steps", detail.name);
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Failed to reorder");
      await loadDetail();
    }
  };

  return (
    <AppShell>
      <div className="w-full">
        <div className="mb-6">
          <Button
            variant="ghost"
            onClick={() => void navigate("/admin/flows")}
            className="gap-2 -ml-3 text-muted-foreground hover:text-foreground"
          >
            <ArrowLeft className="h-4 w-4" /> Back to flows
          </Button>
        </div>

        <PageHeader
          title={name || "Edit Flow"}
          description="Manage flow metadata and arrange reusable forms into ordered steps."
        />

        <div className="grid gap-8 lg:grid-cols-[1fr_1.5fr] items-start">
          <Card className="p-6 flex flex-col sticky top-4 h-[calc(100vh-14rem)]">
            <div className="flex items-center gap-2 text-sm font-medium shrink-0 mb-4">
              <Database className="h-4 w-4" />
              Flow metadata
            </div>
            <div className="grid gap-4 overflow-y-auto flex-1 pr-2 pb-4">
              <div className="space-y-2">
                <label className="text-xs font-medium text-foreground/80">Name</label>
                <Input value={name} onChange={(e) => setName(e.target.value)} />
              </div>
              <div className="space-y-2">
                <label className="text-xs font-medium text-foreground/80">Description</label>
                <Input value={description} onChange={(e) => setDescription(e.target.value)} />
              </div>
              <div className="space-y-2">
                <label className="text-xs font-medium text-foreground/80">Flow JSON</label>
                <Textarea
                  value={contentJson}
                  onChange={(e) => setContentJson(e.target.value)}
                  rows={12}
                  className="font-mono text-sm"
                />
              </div>
              <label className="flex items-center gap-2 text-sm">
                <input
                  type="checkbox"
                  checked={isActive}
                  onChange={(e) => setIsActive(e.target.checked)}
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
                  <div className="flex gap-2">
                    <select
                      className="h-10 w-full rounded-md border border-input bg-background px-3 text-sm focus:outline-none focus:ring-2 focus:ring-primary"
                      value={stepFormId}
                      onChange={(e) => setStepFormId(e.target.value)}
                    >
                      <option value="">Select a form</option>
                      {forms.map((f) => (
                        <option key={f.id} value={f.id}>{f.name}</option>
                      ))}
                    </select>
                    <Button variant="outline" className="px-3 shrink-0" onClick={() => setModalOpen(true)}>
                      <Plus className="h-4 w-4 mr-1" /> New
                    </Button>
                  </div>
                </div>
                <div className="flex items-end">
                  <label className="flex h-10 items-center gap-2 text-sm">
                    <input
                      type="checkbox"
                      checked={stepRequired}
                      onChange={(e) => setStepRequired(e.target.checked)}
                      className="h-4 w-4"
                    />
                    Required
                  </label>
                </div>
              </div>
              <Button onClick={() => void handleAddStep()} disabled={!stepFormId} variant="secondary">
                <Plus className="h-4 w-4 mr-2" /> Add to Flow
              </Button>
            </div>

            <div className="overflow-hidden rounded-lg border border-border">
              <table className="min-w-full divide-y divide-border text-sm">
                <thead className="bg-muted/40">
                  <tr>
                    <th className="px-3 py-3 w-10"></th>
                    <th className="px-3 py-3 text-left font-medium text-muted-foreground">Form</th>
                    <th className="px-3 py-3 text-left font-medium text-muted-foreground">Required</th>
                    <th className="px-3 py-3 text-left font-medium text-muted-foreground">Actions</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-border bg-background">
                  {detail?.steps?.length ? (
                    detail.steps.map((step, index) => (
                      <tr
                        key={step.form_id}
                        draggable
                        onDragStart={(e) => handleDragStart(e, index)}
                        onDragOver={handleDragOver}
                        onDrop={(e) => void handleDrop(e, index)}
                        onDragEnd={() => setDraggedIndex(null)}
                        className={`hover:bg-muted/30 transition-colors ${draggedIndex === index ? "opacity-50 bg-muted" : ""}`}
                      >
                        <td className="px-3 py-2 cursor-grab active:cursor-grabbing">
                          <GripVertical className="h-4 w-4 text-muted-foreground/50" />
                        </td>
                        <td className="px-3 py-2 font-medium">{step.form_name || step.form_slug || step.form_id}</td>
                        <td className="px-3 py-2">{step.is_required ? "Yes" : "No"}</td>
                        <td className="px-3 py-2">
                          <div className="flex items-center gap-1">
                            <Button
                              variant="ghost"
                              size="sm"
                              onClick={() => void navigate(`/admin/forms/${step.form_id}`)}
                            >
                              <Edit className="h-3.5 w-3.5" />
                            </Button>
                            <Button
                              variant="ghost"
                              size="sm"
                              onClick={() => void handleRemoveStep(step.form_id)}
                              className="text-red-500 hover:bg-red-50"
                            >
                              <Trash2 className="h-4 w-4" />
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
      {/* Create Form Modal */}
      <Dialog open={modalOpen} onOpenChange={setModalOpen}>
        <DialogContent className="w-full max-w-2xl">
          <DialogHeader>
            <DialogTitle>New form</DialogTitle>
            <DialogDescription>
              Fill in the details to create a new form definition.
            </DialogDescription>
          </DialogHeader>

          <div className="grid gap-4 mt-2 max-h-[70vh] overflow-y-auto pr-1">
            <div className="space-y-2">
              <label className="text-xs font-medium text-foreground/80">Name</label>
              <Input
                value={newFormName}
                onChange={(e) => setNewFormName(e.target.value)}
                placeholder="My Form"
              />
            </div>

            <div className="space-y-2">
              <label className="text-xs font-medium text-foreground/80">Description</label>
              <Input
                value={newFormDesc}
                onChange={(e) => setNewFormDesc(e.target.value)}
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
                value={newFormJson}
                onChange={(e) => setNewFormJson(e.target.value)}
                rows={12}
                className="font-mono text-sm"
                placeholder="Paste form JSON here"
              />
            </div>

            <label className="flex items-center gap-2 text-sm text-foreground">
              <input
                type="checkbox"
                checked={newFormActive}
                onChange={(e) => setNewFormActive(e.target.checked)}
              />
              Active
            </label>
          </div>

          <div className="flex gap-2 pt-4 border-t border-border mt-2">
            <Button onClick={() => void handleCreateForm()} disabled={creatingForm || !newFormName.trim()}>
              {creatingForm ? "Creating..." : "Create form"}
            </Button>
            <Button variant="secondary" onClick={() => setModalOpen(false)} disabled={creatingForm}>
              Cancel
            </Button>
          </div>
        </DialogContent>
      </Dialog>
    </AppShell>
  );
}

