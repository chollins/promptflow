import { createFileRoute, Link, useNavigate } from "@tanstack/react-router";
import { ArrowLeft, Sparkles } from "lucide-react";
import { useState, type ReactNode } from "react";
import { Button, Input, Field } from "@/components/ui-kit";
import { authService } from "@/lib/auth";

export const Route = createFileRoute("/login")({
  component: LoginPage,
});

export function AuthLayout({
  children,
  title,
  subtitle,
}: {
  children: ReactNode;
  title: string;
  subtitle?: string;
}) {
  return (
    <div className="min-h-screen bg-background flex flex-col">
      <header className="h-16 border-b border-border">
        <div className="max-w-6xl mx-auto h-full px-6 flex items-center justify-between">
          <Link to="/" className="flex items-center gap-2 font-semibold tracking-tight">
            <Sparkles className="h-4 w-4" strokeWidth={2.25} />
            PromptFlow
          </Link>
          <Link to="/" className="text-xs text-muted-foreground hover:text-foreground inline-flex items-center gap-1">
            <ArrowLeft className="h-3 w-3" /> Home
          </Link>
        </div>
      </header>
      <main className="flex-1 flex items-center justify-center px-6 py-12">
        <div className="w-full max-w-md">
          <div className="mb-8 text-center">
            <h1 className="text-2xl font-semibold tracking-tight">{title}</h1>
            {subtitle && <p className="text-sm text-muted-foreground mt-2">{subtitle}</p>}
          </div>
          <div className="rounded-xl border border-border bg-card p-8">{children}</div>
        </div>
      </main>
    </div>
  );
}
function getErrorMessage(error: unknown): string {
  if (!(error instanceof Error)) {
    return "Something went wrong. Please try again.";
  }

  try {
    const parsed = JSON.parse(error.message);

    if (typeof parsed.detail === "string") {
      return parsed.detail;
    }

    if (typeof parsed.message === "string") {
      return parsed.message;
    }
  } catch {
    // Error message isn't JSON
  }

  if (error.message.includes("401")) {
    return "Invalid email or password.";
  }

  if (error.message.includes("403")) {
    return "You don't have permission to sign in.";
  }

  if (error.message.includes("Network")) {
    return "Unable to connect to the server. Please try again.";
  }

  return error.message || "Unable to sign in. Please try again.";
} 

function LoginPage() {
  const navigate = useNavigate();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const onSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    const form = new FormData(e.currentTarget as HTMLFormElement);
    const email = String(form.get("email") || "");
    const password = String(form.get("password") || "");
    authService
      .login(email, password)
      .then((user) => {
        const role = user.role;
        if (role === "superadmin") {
          navigate({ to: "/admin/organizations" });
          return;
        }
        navigate({ to: "/dashboard" });
      })
      .catch((err: Error) => setError(getErrorMessage(err)))
      .finally(() => setLoading(false));
  };

  return (
    <AuthLayout title="Welcome back" subtitle="Sign in to your PromptFlow workspace.">
      <form onSubmit={onSubmit} className="space-y-5">
        <Field label="Email">
          <Input name="email" type="email" placeholder="you@company.com" required />
        </Field>
        <Field label="Password">
          <Input name="password" type="password" placeholder="Password" required />
        </Field>
        {error && <div className="text-sm text-red-600">{error}</div>}
        <Button type="submit" className="w-full" disabled={loading}>
          {loading ? "Signing in..." : "Sign In"}
        </Button>
        <div className="text-center">
          <Link to="/forgot-password" className="text-xs text-muted-foreground hover:text-foreground">
            Forgot password?
          </Link>
        </div>
      </form>
      <div className="mt-6 pt-6 border-t border-border text-center text-xs text-muted-foreground">
        Don't have an organization?{" "}
        <Link to="/create-organization" className="text-foreground font-medium hover:underline">
          Create one
        </Link>
      </div>
    </AuthLayout>
  );
}
