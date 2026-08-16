import { createFileRoute, Link } from "@tanstack/react-router";
import { useState } from "react";
import { CheckCircle2 } from "lucide-react";
import { Button, Input, Field } from "@/components/ui-kit";
import { AuthLayout } from "./login";

export const Route = createFileRoute("/create-organization")({
  component: CreateOrgPage,
});

function CreateOrgPage() {
  const [submitted, setSubmitted] = useState(false);

  if (submitted) {
    return (
      <AuthLayout title="Organization created" subtitle="Your workspace is ready.">
        <div className="flex flex-col items-center text-center py-4">
          <div className="h-12 w-12 rounded-full bg-foreground text-background flex items-center justify-center mb-4">
            <CheckCircle2 className="h-6 w-6" />
          </div>
          <p className="text-sm text-muted-foreground mb-6">
            Redirecting you to your dashboard…
          </p>
          <Link to="/dashboard">
            <Button>Go to Dashboard</Button>
          </Link>
        </div>
      </AuthLayout>
    );
  }

  return (
    <AuthLayout
      title="Create your organization"
      subtitle="Set up your workspace and admin account."
    >
      <form
        onSubmit={(e) => {
          e.preventDefault();
          setSubmitted(true);
        }}
        className="space-y-5"
      >
        <div className="grid grid-cols-1 gap-5">
          <Field label="Organization Name">
            <Input required placeholder="Digital Mixology" />
          </Field>
          <Field label="Organization Code" hint="Optional — used for team registration.">
            <Input placeholder="DMX" />
          </Field>
        </div>

        <div className="pt-2 border-t border-border" />

        <div className="grid grid-cols-2 gap-4">
          <Field label="Admin First Name">
            <Input required placeholder="John" />
          </Field>
          <Field label="Admin Last Name">
            <Input required placeholder="Doe" />
          </Field>
        </div>
        <Field label="Email">
          <Input type="email" required placeholder="john@email.com" />
        </Field>
        <div className="grid grid-cols-2 gap-4">
          <Field label="Password">
            <Input type="password" required />
          </Field>
          <Field label="Confirm Password">
            <Input type="password" required />
          </Field>
        </div>

        <Button type="submit" className="w-full">Create Workspace</Button>

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
