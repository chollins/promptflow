import { createFileRoute } from "@tanstack/react-router";
import { useState } from "react";
import { Plus, X, Trash2, Pencil, Workflow } from "lucide-react";
import { AppShell, PageHeader } from "@/components/app-shell";
import { Button, Card, Input, Field, Badge } from "@/components/ui-kit";
import { FLOWS, type Flow } from "@/lib/mock-store";

export const Route = createFileRoute("/admin/flows")({
  component: FlowsCatalog,
});

function FlowsCatalog() {
  const [flows, setFlows] = useState<Flow[]>(FLOWS);
  const [editing, setEditing] = useState<Flow | null>(null);
  const [creating, setCreating] = useState(false);

  const upsert = (f: Flow) =>
    setFlows((prev) => {
      const idx = prev.findIndex((x) => x.id === f.id);
      if (idx === -1) return [...prev, f];
      const copy = [...prev];
      copy[idx] = f;
      return copy;
    });

  const remove = (id: string) => setFlows((prev) => prev.filter((f) => f.id !== id));
  const toggle = (id: string) =>
    setFlows((prev) => prev.map((f) => (f.id === id ? { ...f, enabled: !f.enabled } : f)));

  return (
    <AppShell>
      <PageHeader
        title="Flows catalog"
        description="Global list of flows that organizations can enable."
        actions={
          <Button onClick={() => setCreating(true)}>
            <Plus className="h-4 w-4" />
            New Flow
          </Button>
        }
      />

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {flows.map((flow) => (
          <Card key={flow.id}>
            <div className="flex items-start justify-between mb-3">
              <div className="h-9 w-9 rounded-lg border border-border flex items-center justify-center">
                <Workflow className="h-4 w-4" />
              </div>
              <div className="flex items-center gap-2">
                <button
                  onClick={() => toggle(flow.id)}
                  className="text-xs text-muted-foreground hover:text-foreground"
                >
                  <Badge tone={flow.enabled ? "neutral" : "outline"}>
                    {flow.enabled ? "Enabled" : "Disabled"}
                  </Badge>
                </button>
                <button
                  onClick={() => setEditing(flow)}
                  className="text-muted-foreground hover:text-foreground"
                  aria-label="Edit"
                >
                  <Pencil className="h-3.5 w-3.5" />
                </button>
                <button
                  onClick={() => remove(flow.id)}
                  className="text-muted-foreground hover:text-foreground"
                  aria-label="Delete"
                >
                  <Trash2 className="h-3.5 w-3.5" />
                </button>
              </div>
            </div>
            <div className="font-medium">{flow.name}</div>
            <div className="text-xs text-muted-foreground mt-1 leading-relaxed">{flow.description}</div>
            <div className="text-[10px] font-mono text-muted-foreground mt-3">{flow.id}</div>
          </Card>
        ))}
      </div>

      {(editing || creating) && (
        <FlowModal
          initial={editing}
          onClose={() => {
            setEditing(null);
            setCreating(false);
          }}
          onSave={(f) => {
            upsert(f);
            setEditing(null);
            setCreating(false);
          }}
        />
      )}
    </AppShell>
  );
}

function FlowModal({
  initial,
  onClose,
  onSave,
}: {
  initial: Flow | null;
  onClose: () => void;
  onSave: (f: Flow) => void;
}) {
  const [name, setName] = useState(initial?.name ?? "");
  const [description, setDescription] = useState(initial?.description ?? "");
  const [enabled, setEnabled] = useState(initial?.enabled ?? true);

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-foreground/20 backdrop-blur-sm p-4">
      <div className="w-full max-w-md rounded-xl border border-border bg-background shadow-xl">
        <div className="flex items-center justify-between px-6 py-4 border-b border-border">
          <div className="font-medium">{initial ? "Edit flow" : "New flow"}</div>
          <button onClick={onClose} className="text-muted-foreground hover:text-foreground">
            <X className="h-4 w-4" />
          </button>
        </div>
        <form
          onSubmit={(e) => {
            e.preventDefault();
            if (!name) return;
            onSave({
              id: initial?.id ?? `f_${Math.random().toString(36).slice(2, 8)}`,
              name,
              description,
              enabled,
            });
          }}
          className="p-6 space-y-5"
        >
          <Field label="Name">
            <Input value={name} onChange={(e) => setName(e.target.value)} required />
          </Field>
          <Field label="Description">
            <Input value={description} onChange={(e) => setDescription(e.target.value)} />
          </Field>
          <Field label="Status">
            <div className="flex rounded-md border border-border p-0.5">
              {[
                { k: true, l: "Enabled" },
                { k: false, l: "Disabled" },
              ].map((o) => (
                <button
                  type="button"
                  key={o.l}
                  onClick={() => setEnabled(o.k)}
                  className={
                    "flex-1 text-sm py-1.5 rounded-sm transition-colors " +
                    (enabled === o.k ? "bg-foreground text-background" : "text-muted-foreground hover:text-foreground")
                  }
                >
                  {o.l}
                </button>
              ))}
            </div>
          </Field>
          <div className="flex gap-2 justify-end pt-2">
            <Button type="button" variant="secondary" onClick={onClose}>Cancel</Button>
            <Button type="submit">{initial ? "Save" : "Create"}</Button>
          </div>
        </form>
      </div>
    </div>
  );
}
