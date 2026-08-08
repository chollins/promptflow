import { createFileRoute, Link, useNavigate } from "@tanstack/react-router";
import { useEffect, useMemo, useState } from "react";
import { AppShell, PageHeader } from "@/components/app-shell";
import { Button, Card, Input } from "@/components/ui-kit";
import { Textarea } from "@/components/ui/textarea";
import { FlowRunner } from "@/components/flow-runner";
import { authService } from "@/lib/auth";
import { apiDelete, apiGet, apiPut } from "@/lib/api";

type FlowStep = {
  id: string;
  sequence: number;
  name: string;
  prompt_form_id: string;
  review?: { required: boolean; editable: boolean };
};

type FlowRecord = {
  id: string;
  name: string;
  slug: string;
  description: string | null;
  content_json: string | null;
  is_active: boolean;
};

type FlowDefinition = {
  id: string;
  version: string;
  name: string;
  description?: string | null;
  runtime?: { mode: "guided" | "automatic"; default_review_required: boolean };
  steps: FlowStep[];
};

export const Route = createFileRoute("/flows_/$flowId")({
  component: FlowDetailPage,
});

function FlowDetailPage() {
  const { flowId } = Route.useParams();
  const navigate = useNavigate();
  const [role, setRole] = useState<string | null>(null);
  const [record, setRecord] = useState<FlowRecord | null>(null);
  const [definition, setDefinition] = useState<FlowDefinition | null>(null);
  const [editMode, setEditMode] = useState(false);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [flowName, setFlowName] = useState("");
  const [description, setDescription] = useState("");
  const [contentJson, setContentJson] = useState("");
  const [isActive, setIsActive] = useState(true);

  useEffect(() => {
    authService
      .getMe()
      .then((user) => setRole(user.role))
      .catch(() => setRole(null));

    apiGet<FlowRecord>(`/admin/flows/${flowId}`)
      .then((data) => {
        setRecord(data);
        setFlowName(data.name);
        setDescription(data.description || "");
        setContentJson(data.content_json || "");
        setIsActive(data.is_active);
        try {
          setDefinition(JSON.parse(data.content_json || "{}") as FlowDefinition);
        } catch {
          setDefinition(null);
        }
      })
      .catch((err) => setError(err instanceof Error ? err.message : "Failed to load flow"));
  }, [flowId]);

  const steps = useMemo(() => definition?.steps ?? [], [definition]);
  const isSuperadmin = role === "superadmin";

  async function handleSave() {
    setSaving(true);
    setError(null);
    try {
      const res = await apiPut<{ item: { id: string; slug: string } }>(`/admin/flows/${flowId}`, {
        name: flowName,
        description,
        content_json: contentJson,
        is_active: isActive,
      });
      if (res.item.slug && res.item.slug !== flowId) {
        await navigate({ to: "/flows/$flowId", params: { flowId: res.item.slug } });
      } else {
        const updated = await apiGet<FlowRecord>(`/admin/flows/${res.item.slug || flowId}`);
        setRecord(updated);
        try {
          setDefinition(JSON.parse(updated.content_json || "{}") as FlowDefinition);
        } catch {
          setDefinition(null);
        }
        setEditMode(false);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to save flow");
    } finally {
      setSaving(false);
    }
  }

  async function handleDelete() {
    if (!window.confirm("Delete this flow?")) return;
    await apiDelete(`/admin/flows/${flowId}`);
    await navigate({ to: "/flows" });
  }

  return (
    <AppShell>
      {/* {isSuperadmin ? (
        <>
          <PageHeader
            title={editMode ? "Edit flow" : record?.name || "Flow"}
            description="View the ordered flow steps, edit the JSON definition, or delete the flow."
            actions={
              record ? (
                <>
                  <Link
                    to="/flows"
                    className="inline-flex h-10 items-center justify-center rounded-md border border-border bg-background px-4 text-sm font-medium text-foreground transition-colors hover:bg-muted"
                  >
                    Back
                  </Link>
                  <Button variant="secondary" onClick={() => setEditMode((value) => !value)}>
                    {editMode ? "View" : "Edit"}
                  </Button>
                  <Button variant="destructive" onClick={() => void handleDelete()}>
                    Delete
                  </Button>
                </>
              ) : null
            }
          />

          {error && (
            <div className="mb-4 rounded-lg border border-red-200 bg-red-50 p-4 text-sm text-red-700">
              {error}
            </div>
          )}

          <div className="grid gap-6 lg:grid-cols-[1fr_1.1fr]">
            <Card className="p-5 space-y-4">
              <div className="text-sm font-medium">Flow metadata</div>
              <div className="grid gap-4">
                <div className="space-y-2">
                  <label className="text-xs font-medium text-foreground/80">Name</label>
                  <Input value={flowName} onChange={(e) => setFlowName(e.target.value)} disabled={!editMode} />
                </div>
                <div className="space-y-2">
                  <label className="text-xs font-medium text-foreground/80">Description</label>
                  <Input
                    value={description}
                    onChange={(e) => setDescription(e.target.value)}
                    disabled={!editMode}
                  />
                </div>
                <label className="flex items-center gap-2 text-sm">
                  <input
                    type="checkbox"
                    checked={isActive}
                    onChange={(e) => setIsActive(e.target.checked)}
                    disabled={!editMode}
                  />
                  Active
                </label>
                <div className="space-y-2">
                  <label className="text-xs font-medium text-foreground/80">Content JSON</label>
                  <Textarea
                    rows={20}
                    className="font-mono text-sm"
                    value={contentJson}
                    onChange={(e) => setContentJson(e.target.value)}
                    disabled={!editMode}
                  />
                </div>
                {editMode && (
                  <div className="flex gap-2">
                    <Button onClick={() => void handleSave()} disabled={saving}>
                      {saving ? "Saving..." : "Save changes"}
                    </Button>
                    <Button
                      variant="secondary"
                      onClick={() => {
                        if (!record) return;
                        setFlowName(record.name);
                        setDescription(record.description || "");
                        setContentJson(record.content_json || "");
                        setIsActive(record.is_active);
                        setEditMode(false);
                      }}
                    >
                      Cancel
                    </Button>
                  </div>
                )}
              </div>
            </Card>

            <Card className="p-5 space-y-4">
              <div className="text-sm font-medium">Flow steps</div>
              <div className="overflow-hidden rounded-lg border border-border">
                <table className="min-w-full divide-y divide-border text-sm">
                  <thead className="bg-muted/40">
                    <tr>
                      <th className="px-3 py-2 text-left font-medium">Step</th>
                      <th className="px-3 py-2 text-left font-medium">Form</th>
                      <th className="px-3 py-2 text-left font-medium">Required</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-border">
                    {steps.length === 0 ? (
                      <tr>
                        <td className="px-3 py-4 text-muted-foreground" colSpan={3}>
                          No steps available.
                        </td>
                      </tr>
                    ) : (
                      steps.map((step) => (
                        <tr key={step.id}>
                          <td className="px-3 py-2">{step.sequence}</td>
                          <td className="px-3 py-2">{step.name}</td>
                          <td className="px-3 py-2">{step.review?.required ? "Yes" : "No"}</td>
                        </tr>
                      ))
                    )}
                  </tbody>
                </table>
              </div>

              <div className="space-y-2">
                <div className="text-sm font-medium">Prompt preview</div>
                <div className="rounded-lg border border-border p-4 text-sm text-muted-foreground">
                  Use the full runner page to execute this flow after editing.
                </div>
              </div>
            </Card>
          </div>
        </>
      ) : ( */}
        <div className="space-y-6">
          <PageHeader title={record?.name || "Flow"} description="Execute this workflow." />
          <FlowRunner flowId={flowId} />
        </div>
      {/* )} */}
    </AppShell>
  );
}
