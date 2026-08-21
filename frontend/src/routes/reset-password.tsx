import { useNavigate, useSearchParams } from "react-router-dom";
import { useState } from "react";
import { AuthLayout } from "./login";
import { Button, Input, Field } from "@/components/ui-kit";
import { apiPost } from "@/lib/api";
import { toast } from "sonner";

export default function ResetPasswordPage() {
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const resetToken = searchParams.get("token") || "";
  const [newPassword, setNewPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!resetToken) {
      toast.error("Invalid reset token session. Please try forgot password process again.");
      return;
    }
    if (newPassword !== confirmPassword) {
      toast.error("Passwords do not match.");
      return;
    }

    setLoading(true);
    try {
      await apiPost("/auth/forgot-password/reset", {
        reset_token: resetToken,
        new_password: newPassword,
      });
      toast.success("Password reset successfully. Please sign in with your new password.");
      navigate("/login");
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Failed to reset password.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <AuthLayout title="Create New Password" subtitle="Enter your new password below.">
      <form onSubmit={handleSubmit} className="space-y-5">
        <Field label="New Password">
          <Input
            type="password"
            placeholder="New Password"
            value={newPassword}
            onChange={(e) => setNewPassword(e.target.value)}
            required
          />
        </Field>

        <Field label="Confirm Password">
          <Input
            type="password"
            placeholder="Confirm Password"
            value={confirmPassword}
            onChange={(e) => setConfirmPassword(e.target.value)}
            required
          />
        </Field>

        <Button type="submit" className="w-full" disabled={loading}>
          {loading ? "Resetting Password..." : "Reset Password"}
        </Button>
      </form>
    </AuthLayout>
  );
}

