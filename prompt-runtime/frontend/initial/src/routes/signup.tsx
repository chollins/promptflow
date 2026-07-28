import { createFileRoute, Link } from "@tanstack/react-router";
import { useState } from "react";
import { AlertCircle, CheckCircle2, Building2 } from "lucide-react";
import { z } from "zod";
import { Button, Input, Field } from "@/components/ui-kit";
import { AuthLayout } from "./login";
import { ORG } from "@/lib/mock-store";

const searchSchema = z.object({
  token: z.string().optional(),
  oc: z.string().optional(),
});

export const Route = createFileRoute("/signup")({
  validateSearch: (search) => searchSchema.parse(search),
  component: SignupPage,
});

function SignupPage() {
  const { token, oc } = Route.useSearch();

  // Expired / invalid token
  if (token === "expired") {
    return (
      <AuthLayout title="Invitation Expired">
        <div className="flex flex-col items-center text-center py-4">
          <div className="h-14 w-14 rounded-full border border-border flex items-center justify-center mb-5">
            <AlertCircle className="h-7 w-7 text-muted-foreground" strokeWidth={1.75} />
          </div>
          <p className="text-sm text-muted-foreground max-w-xs mb-6 leading-relaxed">
            This invitation has expired or has already been used. Please contact your
            organization administrator for a new invitation.
          </p>
          <Link to="/">
            <Button variant="secondary">Return Home</Button>
          </Link>
        </div>
      </AuthLayout>
    );
  }

  // Invalid org code
  if (oc && oc !== ORG.code) {
    return (
      <AuthLayout title="Organization not found">
        <div className="flex flex-col items-center text-center py-4">
          <div className="h-14 w-14 rounded-full border border-border flex items-center justify-center mb-5">
            <AlertCircle className="h-7 w-7 text-muted-foreground" strokeWidth={1.75} />
          </div>
          <p className="text-sm text-muted-foreground max-w-xs mb-6 leading-relaxed">
            Please verify the organization code or contact your administrator.
          </p>
          <Link to="/">
            <Button variant="secondary">Return Home</Button>
          </Link>
        </div>
      </AuthLayout>
    );
  }

  // Token flow (validated)
  if (token) {
    return <TokenSignup />;
  }

  // Org code flow (validated)
  if (oc) {
    return <OrgCodeSignup />;
  }

  // No params
  return (
    <AuthLayout title="Join an organization" subtitle="You need an invitation link or organization code.">
      <div className="text-center py-4 space-y-4">
        <p className="text-sm text-muted-foreground">
          Ask your administrator for an invitation link, or explore the example flows:
        </p>
        <div className="grid gap-2">
          <Link to="/signup" search={{ token: "abc123" }}>
            <Button variant="secondary" className="w-full">Try invitation link</Button>
          </Link>
          <Link to="/signup" search={{ oc: "DMX" }}>
            <Button variant="secondary" className="w-full">Try organization code</Button>
          </Link>
          <Link to="/signup" search={{ token: "expired" }}>
            <Button variant="ghost" className="w-full">Try expired invitation</Button>
          </Link>
        </div>
      </div>
    </AuthLayout>
  );
}

function SuccessScreen() {
  return (
    <AuthLayout title="Account Created" subtitle="Welcome to PromptFlow.">
      <div className="flex flex-col items-center text-center py-4">
        <div className="h-12 w-12 rounded-full bg-foreground text-background flex items-center justify-center mb-4">
          <CheckCircle2 className="h-6 w-6" />
        </div>
        <p className="text-sm text-muted-foreground mb-6">Your account is ready to go.</p>
        <Link to="/login">
          <Button>Go to Login</Button>
        </Link>
      </div>
    </AuthLayout>
  );
}

function TokenSignup() {
  const [done, setDone] = useState(false);
  if (done) return <SuccessScreen />;

  return (
    <AuthLayout title="Complete your registration">
      <div className="mb-6 flex items-start gap-3 p-4 rounded-lg bg-muted">
        <Building2 className="h-4 w-4 mt-0.5 shrink-0" />
        <div className="text-sm">
          <div className="text-muted-foreground text-xs">You're joining</div>
          <div className="font-medium">{ORG.name}</div>
        </div>
      </div>

      <form
        onSubmit={(e) => {
          e.preventDefault();
          setDone(true);
        }}
        className="space-y-5"
      >
        <div className="grid grid-cols-2 gap-4">
          <Field label="Email">
            <Input value="john@email.com" readOnly disabled />
          </Field>
          <Field label="Role">
            <Input value="User" readOnly disabled />
          </Field>
        </div>

        <div className="pt-2 border-t border-border" />

        <div className="grid grid-cols-2 gap-4">
          <Field label="First Name">
            <Input required placeholder="John" />
          </Field>
          <Field label="Last Name">
            <Input required placeholder="Doe" />
          </Field>
        </div>
        <div className="grid grid-cols-2 gap-4">
          <Field label="Password">
            <Input type="password" required />
          </Field>
          <Field label="Confirm Password">
            <Input type="password" required />
          </Field>
        </div>

        <Button type="submit" className="w-full">Create Account</Button>
      </form>
    </AuthLayout>
  );
}

function OrgCodeSignup() {
  const [done, setDone] = useState(false);
  if (done) return <SuccessScreen />;

  return (
    <AuthLayout title="Complete your registration">
      <div className="mb-6 flex items-start gap-3 p-4 rounded-lg bg-muted">
        <Building2 className="h-4 w-4 mt-0.5 shrink-0" />
        <div className="text-sm">
          <div className="text-muted-foreground text-xs">Organization</div>
          <div className="font-medium">{ORG.name}</div>
        </div>
      </div>

      <form
        onSubmit={(e) => {
          e.preventDefault();
          setDone(true);
        }}
        className="space-y-5"
      >
        <div className="grid grid-cols-2 gap-4">
          <Field label="First Name">
            <Input required />
          </Field>
          <Field label="Last Name">
            <Input required />
          </Field>
        </div>
        <Field label="Email">
          <Input type="email" required />
        </Field>
        <div className="grid grid-cols-2 gap-4">
          <Field label="Password">
            <Input type="password" required />
          </Field>
          <Field label="Confirm Password">
            <Input type="password" required />
          </Field>
        </div>
        <Button type="submit" className="w-full">Create Account</Button>
      </form>
    </AuthLayout>
  );
}
