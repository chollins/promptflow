import { Link, useNavigate } from "react-router-dom";
import { useEffect, useState } from "react";
import { Database, Plus, Trash2, Workflow } from "lucide-react";
import { AppShell, PageHeader } from "@/components/app-shell";
import { Button, Card, Input } from "@/components/ui-kit";
import { apiDelete, apiGet, apiPost } from "@/lib/api";
import { logActivity } from "@/lib/activity";
import { toast } from "sonner";
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from "@/components/ui/dialog";

type FlowItem = {
  id: string;
  name: string;
  slug: string;
  description: string | null;
  content_json?: string | null;
  file_path: string;
  is_active: boolean;
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

export default function FlowsCatalog() {
  const navigate = useNavigate();
  const [items, setItems] = useState<FlowItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [createDialogOpen, setCreateDialogOpen] = useState(false);
  const [form, setForm] = useState<FlowFormState>(EMPTY_FORM);

  async function refresh() {
    const flowData = await apiGet<{ items: FlowItem[] }>("/admin/flows");
    setItems(flowData.items);
  }

  useEffect(() => {
    let active = true;
    setLoading(true);
    refresh()
      .catch(() => active && setItems([]))
      .finally(() => active && setLoading(false));
    return () => { active = false; };
  }, []);

  async function handleCreateSave() {
    setSaving(true);
    try {
      const created = await apiPost<{ item: { id: string } }>("/admin/flows", {
        name: form.name,
        description: form.description,
        content_json: form.content_json,
        is_active: form.is_active,
      });
      await refresh();
      setCreateDialogOpen(false);
      setForm(EMPTY_FORM);
      toast.success("Flow created");
      logActivity("created flow", form.name);
      void navigate(`/admin/flows/${created.item.id}`);
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Failed to create flow");
    } finally {
      setSaving(false);
    }
  }

  async function handleDelete(id: string) {
    if (!window.confirm("Delete this flow?")) return;
    try {
      await apiDelete(`/admin/flows/${id}`);
      await refresh();
      toast.success("Flow deleted");
      logActivity("deleted flow", id);
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Failed to delete flow");
    }
  }

  return (
    <AppShell>
      <div className="w-full">
        <PageHeader
          title="Flow Composer"
          description="Create flows and arrange reusable forms into ordered flow steps."
          actions={
            <Button onClick={() => { setForm(EMPTY_FORM); setCreateDialogOpen(true); }}>
              <Plus className="h-4 w-4" />
              Create new flow
            </Button>
          }
        />

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
                      <div className="mt-2 text-sm text-muted-foreground line-clamp-1">{item.description}</div>
                    )}
                  </div>
                  <div className="flex shrink-0 items-center gap-2">
                    <Button asChild variant="secondary" size="sm">
                      <Link to={`/admin/flows/${item.id}`} >
                        Edit flow
                      </Link>
                    </Button>
                    <Button variant="ghost" size="sm" onClick={() => void handleDelete(item.id)} className="text-red-500 hover:bg-red-50">
                      <Trash2 className="h-4 w-4" />
                    </Button>
                  </div>
                </div>
              ))}
            </div>
          )}
        </Card>
      </div>

      <Dialog open={createDialogOpen} onOpenChange={setCreateDialogOpen}>
        <DialogContent className="max-w-md">
          <DialogHeader>
            <DialogTitle>Create new flow</DialogTitle>
            <DialogDescription>Create a new flow to compose multiple forms.</DialogDescription>
          </DialogHeader>
          <div className="grid gap-4 mt-2">
            <div className="space-y-2">
              <label className="text-xs font-medium text-foreground/80">Name</label>
              <Input
                value={form.name}
                onChange={(e) => setForm((prev) => ({ ...prev, name: e.target.value }))}
                placeholder="e.g. Client Assessment"
              />
            </div>
            <div className="space-y-2">
              <label className="text-xs font-medium text-foreground/80">Description</label>
              <Input
                value={form.description}
                onChange={(e) => setForm((prev) => ({ ...prev, description: e.target.value }))}
                placeholder="Brief description of the flow"
              />
            </div>
            <label className="flex items-center gap-2 text-sm">
              <input
                type="checkbox"
                checked={form.is_active}
                onChange={(e) => setForm((prev) => ({ ...prev, is_active: e.target.checked }))}
              />
              Active
            </label>
            <div className="flex gap-2 pt-2">
              <Button onClick={() => void handleCreateSave()} disabled={saving}>
                {saving ? "Creating..." : "Create flow"}
              </Button>
              <Button variant="secondary" onClick={() => setCreateDialogOpen(false)} disabled={saving}>
                Cancel
              </Button>
            </div>
          </div>
        </DialogContent>
      </Dialog>
    </AppShell>
  );
}

