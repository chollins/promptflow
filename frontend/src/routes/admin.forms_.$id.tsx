import { createFileRoute, Link, useNavigate } from "@tanstack/react-router";
import { useEffect, useState } from "react";
import { ArrowLeft, Database, Trash2 } from "lucide-react";
import { AppShell, PageHeader } from "@/components/app-shell";
import { Button, Card, Input } from "@/components/ui-kit";
import { Textarea } from "@/components/ui/textarea";
import { apiDelete, apiGet, apiPut } from "@/lib/api";
import { toast } from "sonner";

type FormDetail = {
  id: string;
  name: string;
  slug: string;
  description: string | null;
  content_json: string | null;
  is_active: boolean;
};

export const Route = createFileRoute("/admin/forms_/$id")({
  component: FormDetailView,
});

function FormDetailView() {
  const { id } = Route.useParams();
  const navigate = useNavigate();

  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [form, setForm] = useState<FormDetail | null>(null);

  useEffect(() => {
    let active = true;
    setLoading(true);
    apiGet<FormDetail>(`/admin/forms/${id}`)
      .then((data) => {
        if (active) setForm(data);
      })
      .catch((err: Error) => {
        if (active) toast.error(err.message || "Failed to load form");
      })
      .finally(() => {
        if (active) setLoading(false);
      });
    return () => {
      active = false;
    };
  }, [id]);

  async function handleSave() {
    if (!form) return;
    setSaving(true);
    try {
      const payload = {
        name: form.name,
        description: form.description,
        content_json: form.content_json,
        is_active: form.is_active,
      };
      await apiPut(`/admin/forms/${form.id}`, payload);
      toast.success("Form saved successfully");
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Failed to save form");
    } finally {
      setSaving(false);
    }
  }

  async function handleDelete() {
    if (!form) return;
    if (!window.confirm("Delete this form?")) return;
    try {
      await apiDelete(`/admin/forms/${form.id}`);
      toast.success("Form deleted");
      void navigate({ to: "/admin/forms" });
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Failed to delete form");
    }
  }

  if (loading) {
    return (
      <AppShell>
        <div className="mx-auto w-full max-w-4xl p-8 text-center text-muted-foreground">
          Loading form details...
        </div>
      </AppShell>
    );
  }

  if (!form) {
    return (
      <AppShell>
        <div className="mx-auto w-full max-w-4xl p-8 text-center">
          <div className="text-destructive mb-4">Form not found.</div>
          <Button asChild variant="outline">
            <Link to="/admin/forms">Back to Forms</Link>
          </Button>
        </div>
      </AppShell>
    );
  }

  return (
    <AppShell>
      <div className="mx-auto w-full max-w-4xl">
        <div className="mb-6">
          <Button variant="ghost" onClick={() => window.history.back()} className="gap-2 -ml-3 text-muted-foreground hover:text-foreground">
            <ArrowLeft className="h-4 w-4" /> Back
          </Button>
        </div>
        <PageHeader
          title={form.name || "Edit Form"}
          description={form.slug || "Manage form properties and JSON schema."}
          actions={
            <Button variant="outline" className="text-red-500 hover:text-red-600 hover:bg-red-50 gap-2" onClick={handleDelete}>
              <Trash2 className="h-4 w-4" /> Delete Form
            </Button>
          }
        />

        <Card className="p-6 flex flex-col h-[calc(100vh-14rem)]">
          <div className="flex items-center gap-2 text-sm font-medium shrink-0 mb-4">
            <Database className="h-4 w-4" />
            Form Details
          </div>

          <div className="grid gap-6 overflow-y-auto flex-1 pr-2 pb-4">
            <div className="space-y-2">
              <label className="text-sm font-medium text-foreground/80">Name</label>
              <Input
                value={form.name}
                onChange={(e) => setForm({ ...form, name: e.target.value })}
                className="max-w-md"
              />
            </div>

            <div className="space-y-2">
              <label className="text-sm font-medium text-foreground/80">Description</label>
              <Input
                value={form.description || ""}
                onChange={(e) => setForm({ ...form, description: e.target.value })}
              />
            </div>

            <div className="space-y-2 flex-1 flex flex-col">
              <label className="text-sm font-medium text-foreground/80">Content JSON</label>
              <Textarea
                value={form.content_json || ""}
                onChange={(e) => setForm({ ...form, content_json: e.target.value })}
                className="font-mono text-sm flex-1 min-h-[300px]"
                placeholder="Paste form JSON here"
              />
            </div>

            <label className="flex items-center gap-2 text-sm text-foreground">
              <input
                type="checkbox"
                checked={form.is_active}
                onChange={(e) => setForm({ ...form, is_active: e.target.checked })}
                className="h-4 w-4 rounded border-input focus:ring-primary"
              />
              Active
            </label>
          </div>

          <div className="flex gap-2 pt-4 border-t border-border">
            <Button className="px-3 py-2" onClick={() => void handleSave()} disabled={saving} size="lg">
              {saving ? "Saving..." : "Save Changes"}
            </Button>
          </div>
        </Card>
      </div>
    </AppShell>
  );
}
