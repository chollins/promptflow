import { useParams } from "react-router-dom";
import { AppShell, PageHeader } from "@/components/app-shell";
import { SuperadminFlowRunner } from "@/components/SuperadminFlowRunner";

export default function FlowDetailPage() {
  const { flowId } = useParams();

  return (
    <AppShell>
      <PageHeader
        title="Flow Runner"
        description="Execute this workflow."
      />
      <SuperadminFlowRunner flowId={flowId} />
    </AppShell>
  );
}

