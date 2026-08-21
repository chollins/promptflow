import { } from "react-router-dom";
import { useEffect, useState } from "react";
import { AppShell, PageHeader } from "@/components/app-shell";
import { Card, Button, Input, Field } from "@/components/ui-kit";
import { apiGet, apiPost } from "@/lib/api";
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { toast } from "sonner";

type UserProfile = {
  id: string;
  name: string;
  email: string;
  role: string | null;
  organization_name: string | null;
  is_active: boolean;
};

export default function SettingsPage() {
  const [item, setItem] = useState<UserProfile | null>(null);

  // Password change states
  const [dialogOpen, setDialogOpen] = useState(false);
  const [currentPassword, setCurrentPassword] = useState("");
  const [newPassword, setNewPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [submitting, setSubmitting] = useState(false);

  useEffect(() => {
    apiGet<{ item: UserProfile | null }>("/settings")
      .then((data) => setItem(data.item))
      .catch(() => setItem(null));
  }, []);

  async function handleChangePassword(e: React.FormEvent) {
    e.preventDefault();
    if (newPassword !== confirmPassword) {
      toast.error("New passwords do not match.");
      return;
    }

    setSubmitting(true);
    try {
      await apiPost("/auth/change-password", {
        current_password: currentPassword,
        new_password: newPassword,
        confirm_password: confirmPassword,
      });
      toast.success("Password changed successfully.");
      setDialogOpen(false);
      setCurrentPassword("");
      setNewPassword("");
      setConfirmPassword("");
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Failed to change password.");
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <AppShell>
      <PageHeader
        title="Settings"
        description="Manage your profile and account security."
      />

      <div className="space-y-6">
        {/* Profile */}
        <Card className="p-5">
          <div className="mb-4">
            <h2 className="text-base font-semibold">Profile</h2>
            <p className="text-sm text-muted-foreground">
              Your account information.
            </p>
          </div>

          {item ? (
            <div className="space-y-4">
              <div>
                <p className="text-xs text-muted-foreground">Name</p>
                <p className="font-medium">{item.name}</p>
              </div>

              <div>
                <p className="text-xs text-muted-foreground">Email</p>
                <p className="font-medium">{item.email}</p>
              </div>

              <div>
                <p className="text-xs text-muted-foreground">Organization</p>
                <p className="font-medium">
                  {item.organization_name || "No organization"}
                </p>
              </div>

              <div>
                <p className="text-xs text-muted-foreground">Role</p>
                <p className="font-medium capitalize">
                  {item.role || "No role"}
                </p>
              </div>
            </div>
          ) : (
            <div className="font-medium">No profile data yet</div>
          )}
        </Card>

        {/* Security */}
        <Card className="p-5">
          <div className="mb-4">
            <h2 className="text-base font-semibold">Security</h2>
            <p className="text-sm text-muted-foreground">
              Manage your password and account security.
            </p>
          </div>

          <div className="flex items-center justify-between gap-4">
            <div>
              <p className="font-medium">Password</p>
              <p className="text-sm text-muted-foreground">
                Change your password to keep your account secure.
              </p>
            </div>

            <Button variant="secondary" onClick={() => setDialogOpen(true)}>
              Change Password
            </Button>
          </div>
        </Card>
      </div>

      <Dialog open={dialogOpen} onOpenChange={setDialogOpen}>
        <DialogContent className="sm:max-w-[425px]">
          <DialogHeader>
            <DialogTitle>Change Password</DialogTitle>
            <DialogDescription>
              Enter your current password and your new password.
            </DialogDescription>
          </DialogHeader>

          <form onSubmit={handleChangePassword} className="space-y-4 py-4">
            <Field label="Current Password">
              <Input
                type="password"
                value={currentPassword}
                onChange={(e) => setCurrentPassword(e.target.value)}
                required
              />
            </Field>

            <Field label="New Password">
              <Input
                type="password"
                value={newPassword}
                onChange={(e) => setNewPassword(e.target.value)}
                required
              />
            </Field>

            <Field label="Confirm New Password">
              <Input
                type="password"
                value={confirmPassword}
                onChange={(e) => setConfirmPassword(e.target.value)}
                required
              />
            </Field>

            <div className="flex justify-end gap-2 pt-2">
              <Button
                type="button"
                variant="secondary"
                onClick={() => setDialogOpen(false)}
                disabled={submitting}
              >
                Cancel
              </Button>
              <Button type="submit" disabled={submitting}>
                {submitting ? "Saving..." : "Change Password"}
              </Button>
            </div>
          </form>
        </DialogContent>
      </Dialog>
    </AppShell>
  );
}

