import { createFileRoute, useNavigate } from "@tanstack/react-router";
import { useState } from "react";
import { AuthLayout } from "./login";
import { Button, Input, Field } from "@/components/ui-kit";
import { apiPost } from "@/lib/api";
import { toast } from "sonner";

type SearchParams = {
  email?: string;
};

export const Route = createFileRoute("/verify-otp")({
  validateSearch: (search: Record<string, unknown>): SearchParams => {
    return {
      email: search.email ? String(search.email) : undefined,
    };
  },
  component: VerifyOtpPage,
});

function VerifyOtpPage() {
  const navigate = useNavigate();
  const { email: initialEmail } = Route.useSearch();
  const [email, setEmail] = useState(initialEmail || "");
  const [otp, setOtp] = useState("");
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!email || !otp) return;

    setLoading(true);
    try {
      const res = await apiPost<{ reset_token: string }>("/auth/forgot-password/verify-otp", {
        email,
        otp,
      });
      toast.success("OTP verified successfully.");
      // Navigate to reset password page with the short-lived reset token
      navigate({ to: "/reset-password", search: { token: res.reset_token } });
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Invalid OTP or email.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <AuthLayout title="Verify OTP" subtitle="Enter the 6-digit verification code sent to your email.">
      <form onSubmit={handleSubmit} className="space-y-5">
        <Field label="Email">
          <Input
            type="email"
            placeholder="you@company.com"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            required
            disabled={!!initialEmail}
          />
        </Field>

        <Field label="Verification Code (OTP)">
          <Input
            type="text"
            placeholder="123456"
            maxLength={6}
            value={otp}
            onChange={(e) => setOtp(e.target.value)}
            required
          />
        </Field>

        <Button type="submit" className="w-full" disabled={loading}>
          {loading ? "Verifying..." : "Verify OTP"}
        </Button>
      </form>
    </AuthLayout>
  );
}
