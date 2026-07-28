import { createFileRoute, Link } from "@tanstack/react-router";
import { CheckCircle2 } from "lucide-react";
import { Button } from "@/components/ui-kit";
import { AuthLayout } from "./login";

export const Route = createFileRoute("/signup")({
  component: SignupPage,
});

function SignupPage() {
  return (
    <AuthLayout title="Registration" subtitle="This flow will be backed by the new API later.">
      <div className="space-y-4">
        <p className="text-sm text-muted-foreground">
          Invitation and organization-code signup flows are being rebuilt against the backend.
        </p>
        <Link to="/">
          <Button variant="secondary" className="w-full">Return Home</Button>
        </Link>
      </div>
    </AuthLayout>
  );
}
