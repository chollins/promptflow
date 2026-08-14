import { createFileRoute } from "@tanstack/react-router";
import { useEffect, useState } from "react";
import { ShieldCheck, Plus, Trash2, Link2 } from "lucide-react";
import { AppShell, PageHeader } from "@/components/app-shell";
import { Button, Card } from "@/components/ui-kit";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
} from "@/components/ui/dialog";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import { apiDelete, apiGet, apiPost } from "@/lib/api";
import { toast } from "sonner";

type AccessItem = {
  organization_id: string;
  organization_name: string | null;
  flow_id: string;
  flow_name: string | null;
};

type OrganizationItem = {
  id: string;
  name: string;
  slug: string;
  code: string;
};

type FlowItem = {
  id: string;
  name: string;
  slug: string;
};

export const Route = createFileRoute("/admin/manage-flows")({
  component: ManageFlowsPage,
});

function ManageFlowsPage() {
  const [items, setItems] = useState<AccessItem[]>([]);
  const [organizations, setOrganizations] = useState<OrganizationItem[]>([]);
  const [flows, setFlows] = useState<FlowItem[]>([]);
  const [loading, setLoading] = useState(true);

  // Assign modal
  const [modalOpen, setModalOpen] = useState(false);
  const [selectedOrganizationId, setSelectedOrganizationId] = useState("");
  const [selectedFlowId, setSelectedFlowId] = useState("");
  const [assigning, setAssigning] = useState(false);

  // Delete confirmation
  const [deleteTarget, setDeleteTarget] = useState<AccessItem | null>(null);

  async function refresh() {
    const data = await apiGet<{ items: AccessItem[] }>("/admin/manage-flows");
    setItems(data.items);
  }

  useEffect(() => {
    setLoading(true);
    Promise.all([
      apiGet<{ items: AccessItem[] }>("/admin/manage-flows"),
      apiGet<{ items: OrganizationItem[] }>("/admin/organizations"),
      apiGet<{ items: FlowItem[] }>("/admin/flows"),
    ])
      .then(([accessData, orgData, flowData]) => {
        setItems(accessData.items);
        setOrganizations(orgData.items);
        setFlows(flowData.items);
      })
      .catch(() => {
        setItems([]);
      })
      .finally(() => setLoading(false));
  }, []);

  function openAssignModal() {
    setSelectedOrganizationId(organizations[0]?.id ?? "");
    setSelectedFlowId(flows[0]?.id ?? "");
    setModalOpen(true);
  }

  function closeModal() {
    setModalOpen(false);
    setSelectedOrganizationId("");
    setSelectedFlowId("");
  }

  async function handleAssign() {
    if (!selectedOrganizationId || !selectedFlowId) {
      toast.error("Please select both an organization and a flow.");
      return;
    }
    setAssigning(true);
    try {
      await apiPost(`/admin/organizations/${selectedOrganizationId}/flows`, {
        flow_id: selectedFlowId,
      });
      await refresh();
      closeModal();
      toast.success("Flow assigned successfully.");
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Failed to assign flow.");
    } finally {
      setAssigning(false);
    }
  }

  async function handleRemove(item: AccessItem) {
    try {
      await apiDelete(`/admin/organizations/${item.organization_id}/flows/${item.flow_id}`);
      await refresh();
      toast.success("Assignment removed.");
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Failed to remove assignment.");
    } finally {
      setDeleteTarget(null);
    }
  }

  return (
    <AppShell>
      <PageHeader
        title="Manage Flow Access"
        description="Assign and remove organization-flow access assignments."
        actions={
          <Button onClick={openAssignModal} disabled={loading}>
            <Plus className="h-4 w-4" />
            Assign Flow
          </Button>
        }
      />

      <Card className="p-5">
        <div className="mb-4 flex items-center gap-2 text-sm font-medium">
          <Link2 className="h-4 w-4" />
          Assignments
        </div>

        {loading ? (
          <div className="text-sm text-muted-foreground">Loading assignments...</div>
        ) : items.length === 0 ? (
          <div className="text-sm text-muted-foreground">
            No assignments yet. Click "Assign Flow" to get started.
          </div>
        ) : (
          <div className="space-y-3">
            {items.map((item) => (
              <div
                key={`${item.organization_id}-${item.flow_id}`}
                className="flex items-center justify-between rounded-xl border border-border bg-background p-4 hover:bg-muted/20 transition-colors"
              >
                <div className="flex items-start gap-3">
                  <div className="h-9 w-9 rounded-lg border border-border flex items-center justify-center shrink-0">
                    <ShieldCheck className="h-4 w-4" />
                  </div>
                  <div>
                    <div className="font-medium">
                      {item.organization_name || item.organization_id}
                    </div>
                    <p className="mt-1 text-sm text-muted-foreground">
                      Access to{" "}
                      <span className="font-medium text-foreground">
                        {item.flow_name || item.flow_id}
                      </span>
                    </p>
                  </div>
                </div>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => setDeleteTarget(item)}
                  title="Remove assignment"
                >
                  <Trash2 className="h-4 w-4" />
                </Button>
              </div>
            ))}
          </div>
        )}
      </Card>

      {/* Assign Flow Modal */}
      <Dialog open={modalOpen} onOpenChange={(open) => { if (!open) closeModal(); }}>
        <DialogContent className="w-full max-w-md">
          <DialogHeader>
            <DialogTitle>Assign Flow to Organization</DialogTitle>
            <DialogDescription>
              Select an organization and a flow to grant access.
            </DialogDescription>
          </DialogHeader>

          <div className="grid gap-4 mt-2">
            <div className="space-y-2">
              <label className="text-xs font-medium text-foreground/80">Organization</label>
              <select
                className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm shadow-sm focus:outline-none focus:ring-1 focus:ring-ring"
                value={selectedOrganizationId}
                onChange={(e) => setSelectedOrganizationId(e.target.value)}
              >
                <option value="" disabled>
                  Select organization…
                </option>
                {organizations.map((org) => (
                  <option key={org.id} value={org.id}>
                    {org.name} ({org.code})
                  </option>
                ))}
              </select>
            </div>

            <div className="space-y-2">
              <label className="text-xs font-medium text-foreground/80">Flow</label>
              <select
                className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm shadow-sm focus:outline-none focus:ring-1 focus:ring-ring"
                value={selectedFlowId}
                onChange={(e) => setSelectedFlowId(e.target.value)}
              >
                <option value="" disabled>
                  Select flow…
                </option>
                {flows.map((flow) => (
                  <option key={flow.id} value={flow.id}>
                    {flow.name}
                  </option>
                ))}
              </select>
            </div>

            <div className="flex gap-2 pt-2">
              <Button onClick={() => void handleAssign()} disabled={assigning}>
                {assigning ? "Assigning..." : "Assign"}
              </Button>
              <Button variant="secondary" onClick={closeModal} disabled={assigning}>
                Cancel
              </Button>
            </div>
          </div>
        </DialogContent>
      </Dialog>

      {/* Remove Confirmation */}
      <AlertDialog
        open={Boolean(deleteTarget)}
        onOpenChange={(open) => !open && setDeleteTarget(null)}
      >
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Remove assignment?</AlertDialogTitle>
            <AlertDialogDescription>
              This will remove{" "}
              <strong>{deleteTarget?.organization_name || deleteTarget?.organization_id}</strong>'s
              access to{" "}
              <strong>{deleteTarget?.flow_name || deleteTarget?.flow_id}</strong>.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel onClick={() => setDeleteTarget(null)}>Cancel</AlertDialogCancel>
            <AlertDialogAction
              onClick={() => {
                if (!deleteTarget) return;
                void handleRemove(deleteTarget);
              }}
            >
              Remove
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </AppShell>
  );
}
