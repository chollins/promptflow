import { } from "react-router-dom";
import { useEffect, useMemo, useState } from "react";
import { Mail, PencilLine, Plus, RefreshCw, Search, Send, Trash2 } from "lucide-react";
import { AppShell, PageHeader } from "@/components/app-shell";
import { Badge, Button, Card, Field, Input } from "@/components/ui-kit";
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
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { apiDelete, apiGet, apiPost, apiPut } from "@/lib/api";

type UserItem = {
  id: string;
  name: string | null;
  email: string;
  role: string | null;
  organization_name: string | null;
  organization_id: string | null;
  is_active?: boolean;
  status?: "active" | "pending" | "inactive";
  date_joined?: string | null;
  created_at?: string | null;
  expires_at?: string | null;
};

type UserForm = {
  name: string;
  role: "member" | "admin";
};

type InviteForm = {
  email: string;
  role: "member" | "admin";
};

const PAGE_SIZE = 8;

export default function UsersPage() {
  const [items, setItems] = useState<UserItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [search, setSearch] = useState("");
  const [roleFilter, setRoleFilter] = useState<string>("all");
  const [statusFilter, setStatusFilter] = useState<string>("all");
  const [page, setPage] = useState(1);
  const [inviteOpen, setInviteOpen] = useState(false);
  const [editItem, setEditItem] = useState<UserItem | null>(null);
  const [deleteItem, setDeleteItem] = useState<UserItem | null>(null);
  const [toast, setToast] = useState<string | null>(null);
  const [inviteForm, setInviteForm] = useState<InviteForm>({ email: "", role: "member" });
  const [editForm, setEditForm] = useState<UserForm>({ name: "", role: "member" });
  const [inviteLink, setInviteLink] = useState<string | null>(null);
  const [inviteCopied, setInviteCopied] = useState(false);

  async function refresh() {
    const data = await apiGet<{ items: UserItem[] }>("/users");
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

  const normalized = useMemo(
    () =>
      items.map((item) => ({
        ...item,
        status: item.status ?? (item.created_at || item.date_joined ? "active" : "pending"),
      })),
    [items],
  );

  const filtered = useMemo(() => {
    const term = search.trim().toLowerCase();
    return normalized.filter((item) => {
      const matchesSearch =
        !term || item.name?.toLowerCase().includes(term) || item.email.toLowerCase().includes(term);
      const matchesRole = roleFilter === "all" || (item.role || "member") === roleFilter;
      const matchesStatus = statusFilter === "all" || item.status === statusFilter;
      return matchesSearch && matchesRole && matchesStatus;
    });
  }, [normalized, search, roleFilter, statusFilter]);

  const totalPages = Math.max(1, Math.ceil(filtered.length / PAGE_SIZE));
  const pageItems = filtered.slice((page - 1) * PAGE_SIZE, page * PAGE_SIZE);

  useEffect(() => {
    setPage(1);
  }, [search, roleFilter, statusFilter]);

  function openInvite() {
    setInviteForm({ email: "", role: "member" });
    setInviteLink(null);
    setInviteCopied(false);
    setInviteOpen(true);
  }

  function openEdit(item: UserItem) {
    setEditItem(item);
    setEditForm({ name: item.name || "", role: (item.role as "member" | "admin") || "member" });
  }

  function showToast(message: string) {
    setToast(message);
    window.setTimeout(() => setToast(null), 2500);
  }

  async function handleInvite() {
    setSaving(true);
    setError(null);
    try {
      const result = await apiPost<{ ok: boolean; message: string; registration_link?: string }>(
        "/invitations",
        {
          email: inviteForm.email.trim(),
          role: inviteForm.role,
        },
      );
      setInviteLink(result.registration_link || null);
      setInviteCopied(false);
      showToast(result.message || "Invitation sent.");
      await refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to send invitation");
    } finally {
      setSaving(false);
    }
  }

  async function handleSaveEdit() {
    if (!editItem) return;
    setSaving(true);
    setError(null);
    try {
      await apiPut(`/users/${editItem.id}`, {
        name: editForm.name.trim(),
        role: editForm.role,
      });
      setEditItem(null);
      showToast("User updated.");
      await refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to update user");
    } finally {
      setSaving(false);
    }
  }

  async function handleDelete() {
    if (!deleteItem) return;
    setSaving(true);
    setError(null);
    try {
      if (deleteItem.status === "pending") {
        await apiDelete(`/invitations/${deleteItem.id}`);
        showToast("Invitation canceled.");
      } else {
        await apiDelete(`/users/${deleteItem.id}`);
        showToast("User deactivated.");
      }
      setDeleteItem(null);
      await refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to deactivate user");
    } finally {
      setSaving(false);
    }
  }

  async function handleResend(item: UserItem) {
    setSaving(true);
    setError(null);
    try {
      const result = await apiPost<{ ok: boolean; registration_link: string }>(
        `/invitations/${item.id}/resend`,
        {},
      );
      showToast("Invitation resent.");
      if (result.registration_link) {
        await navigator.clipboard?.writeText(result.registration_link);
      }
      await refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to resend invitation");
    } finally {
      setSaving(false);
    }
  }

  async function copyInviteLink() {
    if (!inviteLink) return;
    await navigator.clipboard?.writeText(inviteLink);
    setInviteCopied(true);
    window.setTimeout(() => setInviteCopied(false), 1500);
  }

  return (
    <AppShell>
      <PageHeader
        title="Users"
        description="Invite-only user management for your organization."
        actions={
          <>
            <Button variant="secondary" onClick={() => void refresh()} disabled={loading}>
              <RefreshCw className="h-4 w-4" />
              Refresh
            </Button>
            <Button onClick={openInvite}>
              <Plus className="h-4 w-4" />
              Invite User
            </Button>
          </>
        }
      />

      {toast && (
        <div className="mb-4 rounded-lg border border-border bg-background p-3 text-sm">
          {toast}
        </div>
      )}
      {error && (
        <div className="mb-4 rounded-lg border border-red-200 bg-red-50 p-4 text-sm text-red-700">
          {error}
        </div>
      )}

      <Card className="mb-4 p-4">
        <div className="grid gap-3 md:grid-cols-[1.5fr_0.7fr_0.7fr]">
          <div className="relative">
            <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
            <Input
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="Search by name or email"
              className="pl-9"
            />
          </div>
          <select
            value={roleFilter}
            onChange={(e) => setRoleFilter(e.target.value)}
            className="h-10 rounded-md border border-border bg-background px-3 text-sm"
          >
            <option value="all">All roles</option>
            <option value="member">User</option>
            <option value="admin">Admin</option>
          </select>
          <select
            value={statusFilter}
            onChange={(e) => setStatusFilter(e.target.value)}
            className="h-10 rounded-md border border-border bg-background px-3 text-sm"
          >
            <option value="all">All status</option>
            <option value="active">Active</option>
            <option value="pending">Pending Invitation</option>
            <option value="inactive">Inactive</option>
          </select>
        </div>
      </Card>

      <Card className="overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead className="border-b border-border bg-muted/30 text-left">
              <tr>
                <th className="px-4 py-3 font-medium">Name</th>
                <th className="px-4 py-3 font-medium">Email</th>
                <th className="px-4 py-3 font-medium">Role</th>
                <th className="px-4 py-3 font-medium">Status</th>
                <th className="px-4 py-3 font-medium">Date Joined</th>
                <th className="px-4 py-3 font-medium">Actions</th>
              </tr>
            </thead>
            <tbody>
              {loading ? (
                <tr>
                  <td className="px-4 py-8 text-center text-muted-foreground" colSpan={6}>
                    Loading users...
                  </td>
                </tr>
              ) : pageItems.length === 0 ? (
                <tr>
                  <td className="px-4 py-8 text-center text-muted-foreground" colSpan={6}>
                    No users found.
                  </td>
                </tr>
              ) : (
                pageItems.map((item) => (
                  <tr key={item.id} className="border-b border-border">
                    <td className="px-4 py-3 font-medium">{item.name || "—"}</td>
                    <td className="px-4 py-3">{item.email}</td>
                    <td className="px-4 py-3 capitalize">{item.role || "User"}</td>
                    <td className="px-4 py-3">
                      <Badge tone={item.status === "pending" ? "muted" : "neutral"}>
                        {item.status === "pending"
                          ? "Pending Invitation"
                          : item.is_active
                            ? "Active"
                            : "Inactive"}
                      </Badge>
                    </td>
                    <td className="px-4 py-3 text-muted-foreground">
                      {item.date_joined || item.created_at
                        ? new Date(item.date_joined || item.created_at || "").toLocaleDateString()
                        : "—"}
                    </td>
                    <td className="px-4 py-3">
                      <div className="flex flex-wrap gap-2">
                        {item.status !== "pending" && (
                          <Button size="sm" variant="ghost" onClick={() => openEdit(item)}>
                            <PencilLine className="h-4 w-4" />
                            Edit
                          </Button>
                        )}
                        {item.status === "pending" ? (
                          <>
                            <Button
                              size="sm"
                              variant="ghost"
                              onClick={() => void handleResend(item)}
                            >
                              <Send className="h-4 w-4" />
                              Resend Invitation
                            </Button>
                            <Button size="sm" variant="ghost" onClick={() => setDeleteItem(item)}>
                              <Trash2 className="h-4 w-4" />
                              Cancel Invitation
                            </Button>
                          </>
                        ) : (
                          <Button size="sm" variant="ghost" onClick={() => setDeleteItem(item)}>
                            <Trash2 className="h-4 w-4" />
                            Delete
                          </Button>
                        )}
                      </div>
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>

        <div className="flex items-center justify-between border-t border-border px-4 py-3 text-sm">
          <div className="text-muted-foreground">
            Showing {pageItems.length ? (page - 1) * PAGE_SIZE + 1 : 0} -{" "}
            {Math.min(page * PAGE_SIZE, filtered.length)} of {filtered.length}
          </div>
          <div className="flex gap-2">
            <Button
              variant="secondary"
              size="sm"
              disabled={page <= 1}
              onClick={() => setPage((p) => p - 1)}
            >
              Previous
            </Button>
            <Button
              variant="secondary"
              size="sm"
              disabled={page >= totalPages}
              onClick={() => setPage((p) => p + 1)}
            >
              Next
            </Button>
          </div>
        </div>
      </Card>

      <Dialog open={inviteOpen} onOpenChange={setInviteOpen}>
        <DialogContent className="w-full max-w-md">
          <DialogHeader>
            <DialogTitle>Invite User</DialogTitle>
            <DialogDescription>
              Sends an invitation email. The user account is created after they accept.
            </DialogDescription>
          </DialogHeader>
          {!inviteLink ? (
            <>
              <div className="space-y-4">
                <Field label="Email">
                  <Input
                    type="email"
                    value={inviteForm.email}
                    onChange={(e) => setInviteForm((prev) => ({ ...prev, email: e.target.value }))}
                    placeholder="john@company.com"
                  />
                </Field>
                <Field label="Role">
                  <select
                    value={inviteForm.role}
                    onChange={(e) =>
                      setInviteForm((prev) => ({
                        ...prev,
                    role: e.target.value as "member" | "admin",
                      }))
                    }
                    className="h-10 w-full rounded-md border border-border bg-background px-3 text-sm"
                  >
                    <option value="member">Member</option>
                    <option value="admin">Admin</option>
                  </select>
                </Field>
              </div>
              <DialogFooter>
                <Button variant="secondary" onClick={() => setInviteOpen(false)}>
                  Cancel
                </Button>
                <Button onClick={() => void handleInvite()} disabled={saving}>
                  <Mail className="h-4 w-4" />
                  Send Invitation
                </Button>
              </DialogFooter>
            </>
          ) : (
            <>
              <div className="space-y-4">
                <div className="rounded-lg border border-border bg-muted/30 p-4">
                  <div className="text-sm font-medium">Invitation sent</div>
                  <p className="mt-1 text-xs text-muted-foreground">
                    Copy this link if you want to share it directly.
                  </p>
                </div>
                <div className="flex items-center gap-2 rounded-md border border-border bg-muted/50 px-3 py-2">
                  <div className="min-w-0 flex-1 truncate font-mono text-xs">{inviteLink}</div>
                  <Button variant="secondary" size="sm" onClick={() => void copyInviteLink()}>
                    {inviteCopied ? "Copied" : "Copy"}
                  </Button>
                </div>
              </div>
              <DialogFooter>
                <Button variant="secondary" onClick={() => setInviteOpen(false)}>
                  Close
                </Button>
                <Button
                  onClick={() => {
                    setInviteLink(null);
                    setInviteForm({ email: "", role: "member" });
                  }}
                >
                  Invite another
                </Button>
              </DialogFooter>
            </>
          )}
        </DialogContent>
      </Dialog>

      <Dialog open={Boolean(editItem)} onOpenChange={(open) => !open && setEditItem(null)}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Edit User</DialogTitle>
            <DialogDescription>
              Update the user name and role. Email cannot be changed here.
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-4">
            <Field label="Name">
              <Input
                value={editForm.name}
                onChange={(e) => setEditForm((prev) => ({ ...prev, name: e.target.value }))}
              />
            </Field>
            <Field label="Role">
              <select
                value={editForm.role}
                onChange={(e) =>
                  setEditForm((prev) => ({
                    ...prev,
                    role: e.target.value as "member" | "admin",
                  }))
                }
                className="h-10 w-full rounded-md border border-border bg-background px-3 text-sm"
              >
                <option value="member">User</option>
                <option value="admin">Admin</option>
              </select>
            </Field>
          </div>
          <DialogFooter>
            <Button variant="secondary" onClick={() => setEditItem(null)}>
              Cancel
            </Button>
            <Button onClick={() => void handleSaveEdit()} disabled={saving}>
              Save Changes
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <AlertDialog open={Boolean(deleteItem)} onOpenChange={(open) => !open && setDeleteItem(null)}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>
              {deleteItem?.status === "pending" ? "Cancel invitation?" : "Delete user?"}
            </AlertDialogTitle>
            <AlertDialogDescription>
              {deleteItem?.status === "pending"
                ? "This will invalidate the invitation."
                : "This will deactivate the account rather than permanently removing it."}
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel onClick={() => setDeleteItem(null)}>Cancel</AlertDialogCancel>
            <AlertDialogAction onClick={() => void handleDelete()} disabled={saving}>
              Confirm
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </AppShell>
  );
}

