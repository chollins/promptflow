import { useParams } from "react-router-dom";
import { useEffect, useState } from "react";
import { AppShell, PageHeader } from "@/components/app-shell";
import { Card } from "@/components/ui-kit";
import { FlowRunner } from "@/components/flow-runner";
import { SuperadminFlowRunner } from "@/components/SuperadminFlowRunner";
import { authService } from "@/lib/auth";

export default function FlowDetailPage() {
  const { flowId } = useParams();
  const [role, setRole] = useState<string | null>(null);
  const [loadingRole, setLoadingRole] = useState(true);

  useEffect(() => {
    let active = true;
    authService
      .getMe()
      .then((user) => {
        if (active) setRole(user.role);
      })
      .catch(() => {
        if (active) setRole(null);
      })
      .finally(() => {
        if (active) setLoadingRole(false);
      });
    return () => {
      active = false;
    };
  }, []);

  const isSuperadmin = role === "superadmin";

  return (
    <AppShell>
      <PageHeader
        title="Flow Runner"
        description={
          isSuperadmin
            ? "Execute this workflow and inspect developer diagnostics."
            : "Execute this workflow."
        }
      />
      {loadingRole ? (
        <Card className="p-5 text-sm text-muted-foreground">Loading runner...</Card>
      ) : isSuperadmin ? (
        <SuperadminFlowRunner flowId={flowId} />
      ) : (
        <FlowRunner flowId={flowId} />
      )}
    </AppShell>
  );
}

