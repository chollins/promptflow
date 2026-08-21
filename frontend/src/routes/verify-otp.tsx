import { useNavigate, useSearchParams } from "react-router-dom";
import { useState } from "react";
import { AuthLayout } from "./login";
import { Button, Input, Field } from "@/components/ui-kit";
import { apiPost } from "@/lib/api";
import { toast } from "sonner";

export default function VerifyOtpPage() {
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const initialEmail = searchParams.get("email") || "";
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
      navigate(`/reset-password?${new URLSearchParams({ token: res.reset_token }).toString()}`);
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

