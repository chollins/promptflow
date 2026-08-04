import { createFileRoute, Link, useNavigate } from "@tanstack/react-router";
import { useState } from "react";
import { CheckCircle2 } from "lucide-react";
import { apiPost } from "@/lib/api";
import { Button, Field, Input } from "@/components/ui-kit";
import { AuthLayout } from "./login";

type CreateOrganizationPayload = {
  organization_name: string;
  organization_code: string;
  admin_name: string;
  admin_email: string;
  admin_password: string;
  admin_confirm_password: string;
};

export const Route = createFileRoute("/create-organization")({
  component: CreateOrgPage,
});

function slugLikeCode(value: string) {
  return value
    .trim()
    .toUpperCase()
    .replace(/[^A-Z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "");
}

function CreateOrgPage() {
  const navigate = useNavigate();
  const [submitted, setSubmitted] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [form, setForm] = useState<CreateOrganizationPayload>({
    organization_name: "",
    organization_code: "",
    admin_name: "",
    admin_email: "",
    admin_password: "",
    admin_confirm_password: "",
  });

  async function handleSubmit(e: React.FormEvent<HTMLFormElement>) {
    e.preventDefault();
    setLoading(true);
    setError(null);

    try {
      if (form.admin_password !== form.admin_confirm_password) {
        throw new Error("Passwords do not match.");
      }

      const payload = {
        organization_name: form.organization_name.trim(),
        organization_code: slugLikeCode(form.organization_code || form.organization_name),
        admin: {
          name: form.admin_name.trim(),
          email: form.admin_email.trim(),
          password: form.admin_password,
          role: "admin",
        },
      };

      if (!payload.organization_name) throw new Error("Organization name is required.");
      if (!payload.admin.name) throw new Error("Admin name is required.");
      if (!payload.admin.email) throw new Error("Admin email is required.");
      if (!payload.admin.password) throw new Error("Admin password is required.");

      await apiPost("/organizations", payload);
      setSubmitted(true);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to create organization");
    } finally {
      setLoading(false);
    }
  }

  if (submitted) {
    return (
      <AuthLayout
        title="Organization created"
        subtitle="Your workspace and admin account are ready."
      >
        <div className="flex flex-col items-center text-center py-4">
          <div className="mb-4 flex h-12 w-12 items-center justify-center rounded-full bg-foreground text-background">
            <CheckCircle2 className="h-6 w-6" />
          </div>
          <p className="mb-6 text-sm text-muted-foreground">Redirecting you to your dashboard...</p>
          <Button onClick={() => void navigate({ to: "/dashboard" })}>Go to Dashboard</Button>
        </div>
      </AuthLayout>
    );
  }

  return (
    <AuthLayout
      title="Create your organization"
      subtitle="Set up your workspace and its first admin in one step."
    >
      <form onSubmit={(e) => void handleSubmit(e)} className="space-y-5">
        <div className="grid grid-cols-1 gap-5">
          <Field
            label="Organization Name"
            hint="This will be the workspace name shown to your team."
          >
            <Input
              required
              placeholder="Your organization name"
              value={form.organization_name}
              onChange={(e) =>
                setForm((prev) => ({
                  ...prev,
                  organization_name: e.target.value,
                  organization_code: prev.organization_code
                    ? prev.organization_code
                    : slugLikeCode(e.target.value),
                }))
              }
            />
          </Field>
          <Field label="Organization Code" hint="Optional, used for team registration.">
            <Input
              placeholder="ORG"
              value={form.organization_code}
              onChange={(e) =>
                setForm((prev) => ({ ...prev, organization_code: slugLikeCode(e.target.value) }))
              }
            />
          </Field>
        </div>

        <div className="pt-2 border-t border-border" />

        <div className="grid grid-cols-2 gap-4">
          <Field label="Admin Name">
            <Input
              required
              placeholder="Full name"
              value={form.admin_name}
              onChange={(e) => setForm((prev) => ({ ...prev, admin_name: e.target.value }))}
            />
          </Field>
          <Field label="Admin Email">
            <Input
              type="email"
              required
              placeholder="you@company.com"
              value={form.admin_email}
              onChange={(e) => setForm((prev) => ({ ...prev, admin_email: e.target.value }))}
            />
          </Field>
        </div>
        <div className="grid grid-cols-2 gap-4">
          <Field label="Password">
            <Input
              type="password"
              required
              value={form.admin_password}
              onChange={(e) => setForm((prev) => ({ ...prev, admin_password: e.target.value }))}
            />
          </Field>
          <Field label="Confirm Password">
            <Input
              type="password"
              required
              value={form.admin_confirm_password}
              onChange={(e) =>
                setForm((prev) => ({ ...prev, admin_confirm_password: e.target.value }))
              }
            />
          </Field>
        </div>

        {error && <div className="text-sm text-red-600">{error}</div>}

        <Button type="submit" className="w-full" disabled={loading}>
          {loading ? "Creating workspace..." : "Create Workspace"}
        </Button>

        <div className="text-center text-xs text-muted-foreground">
          Already have an account?{" "}
          <Link to="/login" className="text-foreground font-medium hover:underline">
            Sign in
          </Link>
        </div>
      </form>
    </AuthLayout>
  );
}
