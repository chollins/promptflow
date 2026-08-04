import { createFileRoute } from "@tanstack/react-router";
import { AppShell, PageHeader } from "@/components/app-shell";
import { FlowRunner } from "@/components/flow-runner";

export const Route = createFileRoute("/flows_/$flowId")({
  component: FlowDetailPage,
});

function FlowDetailPage() {
  const { flowId } = Route.useParams();

  return (
    <AppShell>
      <PageHeader
        title="Flow runner"
        description="Render and execute the selected PromptFlow one step at a time."
      />
      <FlowRunner flowId={flowId} />
    </AppShell>
  );
}

