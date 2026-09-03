import { useParams } from "react-router-dom";
import { AppShell, PageHeader } from "@/components/app-shell";
import { SuperadminFlowRunner } from "@/components/SuperadminFlowRunner";
import { useCurrentUserId } from "@/lib/use-current-user-id";

export default function FlowDetailPage() {
  const { flowId } = useParams();
  const userId = useCurrentUserId();

  return (
    <AppShell>
      <PageHeader
        title="Flow Runner"
        description="Execute this workflow."
      />
      {/* key=userId forces a full remount when the authenticated user changes,
          clearing all diagnostic state so data permitted to one user cannot
          remain visible to the next user on the same tab. */}
      <SuperadminFlowRunner key={userId ?? "loading"} flowId={flowId ?? ""} />
    </AppShell>
  );
}

