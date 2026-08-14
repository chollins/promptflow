import { createFileRoute, redirect, isRedirect } from "@tanstack/react-router";
import { useEffect, useState } from "react";
import { AppShell, PageHeader } from "@/components/app-shell";
import { FormRunner } from "@/components/form-runner";
import { SuperadminFormRunner } from "@/components/SuperadminFormRunner";
import { authService } from "@/lib/auth";
import { apiGet } from "@/lib/api";

export const Route = createFileRoute("/forms_/$formId")({
  beforeLoad: async () => {
    try {
      await apiGet<{ role: string | null }>("/auth/me");
    } catch (e) {
      if (isRedirect(e)) throw e;
      throw redirect({ to: "/login" });
    }
  },
  component: FormDetailPage,
});

function FormDetailPage() {
  const { formId } = Route.useParams();
  const [role, setRole] = useState<string | null>(null);

  useEffect(() => {
    let active = true;
    authService
      .getMe()
      .then((user) => {
        if (active) setRole(user.role);
      })
      .catch(() => {
        if (active) setRole(null);
      });
    return () => {
      active = false;
    };
  }, []);

  const isSuperadmin = role === "superadmin";

  return (
    <AppShell>
      <PageHeader
        title="Form Runner"
        description={
          isSuperadmin
            ? "Execute this reusable form and inspect developer diagnostics."
            : "Execute this reusable form."
        }
      />
      {isSuperadmin ? <SuperadminFormRunner formId={formId} /> : <FormRunner formId={formId} />}
    </AppShell>
  );
}
