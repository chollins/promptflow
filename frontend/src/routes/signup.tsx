import { Link, useNavigate } from "react-router-dom";
import { useEffect, useState } from "react";
import { CheckCircle2 } from "lucide-react";
import { apiGet, apiPost } from "@/lib/api";
import { Button, Field, Input } from "@/components/ui-kit";
import { AuthLayout } from "./login";

type InvitationInfo = {
  invitation_id: string;
  email: string;
  role: string | null;
  organization_id: string;
  organization_name: string | null;
  expires_at: string;
};

export default function SignupPage() {
  const navigate = useNavigate();
  const token =
    typeof window === "undefined"
      ? ""
      : new URLSearchParams(window.location.search).get("token") || "";
  const [loading, setLoading] = useState(true);
  const [submitting, setSubmitting] = useState(false);
  const [accepted, setAccepted] = useState(false);
  const [invite, setInvite] = useState<InvitationInfo | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [form, setForm] = useState({ name: "", password: "", confirmPassword: "" });

  useEffect(() => {
    let active = true;
    if (!token) {
      setLoading(false);
      setError("Missing invitation token.");
      return;
    }

    apiGet<InvitationInfo>(`/invitations/validate?token=${encodeURIComponent(token)}`)
      .then((data) => {
        if (!active) return;
        setInvite(data);
        setForm((prev) => ({ ...prev, name: data.email.split("@")[0] || "" }));
      })
      .catch((err: Error) => active && setError(err.message))
      .finally(() => active && setLoading(false));

    return () => {
      active = false;
    };
  }, [token]);

  async function handleSubmit(e: React.FormEvent<HTMLFormElement>) {
    e.preventDefault();
    setSubmitting(true);
    setError(null);
    try {
      if (!token) throw new Error("Missing invitation token.");
      if (!form.name.trim()) throw new Error("Name is required.");
      if (!form.password) throw new Error("Password is required.");
      if (form.password !== form.confirmPassword) throw new Error("Passwords do not match.");

      await apiPost("/invitations/accept", {
        token,
        name: form.name.trim(),
        password: form.password,
      });
      setAccepted(true);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to accept invitation");
    } finally {
      setSubmitting(false);
    }
  }

  if (accepted) {
    return (
      <AuthLayout title="Registration complete" subtitle="Your account is ready.">
        <div className="flex flex-col items-center text-center py-4">
          <div className="mb-4 flex h-12 w-12 items-center justify-center rounded-full bg-foreground text-background">
            <CheckCircle2 className="h-6 w-6" />
          </div>
          <p className="mb-6 text-sm text-muted-foreground">
            You can now sign in with your new account.
          </p>
          <Button onClick={() => void navigate("/login")}>Go to Login</Button>
        </div>
      </AuthLayout>
    );
  }

  return (
    <AuthLayout
      title="Complete registration"
      subtitle={
        invite
          ? `Join ${invite.organization_name || "your organization"}.`
          : "Validating invitation..."
      }
    >
      {loading ? (
        <div className="text-sm text-muted-foreground">Loading invitation...</div>
      ) : error ? (
        <div className="space-y-4">
          <div className="rounded-lg border border-red-200 bg-red-50 p-4 text-sm text-red-700">
            {error}
          </div>
          <Link to="/login" className="text-sm text-foreground hover:underline">
            Back to login
          </Link>
        </div>
      ) : (
        <form onSubmit={handleSubmit} className="space-y-5">
          <div className="rounded-lg border border-border bg-muted/20 p-4 text-sm">
            <div className="font-medium">{invite?.organization_name}</div>
            <div className="mt-1 text-muted-foreground">{invite?.email}</div>
          </div>

          <Field label="Name">
            <Input
              value={form.name}
              onChange={(e) => setForm((prev) => ({ ...prev, name: e.target.value }))}
              placeholder="Your name"
              required
            />
          </Field>
          <Field label="Password">
            <Input
              type="password"
              value={form.password}
              onChange={(e) => setForm((prev) => ({ ...prev, password: e.target.value }))}
              required
            />
          </Field>
          <Field label="Confirm Password">
            <Input
              type="password"
              value={form.confirmPassword}
              onChange={(e) => setForm((prev) => ({ ...prev, confirmPassword: e.target.value }))}
              required
            />
          </Field>

          {error && <div className="text-sm text-red-600">{error}</div>}

          <Button type="submit" className="w-full" disabled={submitting}>
            {submitting ? "Creating account..." : "Complete Registration"}
          </Button>
        </form>
      )}
    </AuthLayout>
  );
}

