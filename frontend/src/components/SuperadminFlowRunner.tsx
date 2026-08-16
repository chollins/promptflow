import { useMemo, useState } from "react";
import { DebugJsonViewer } from "@/components/debug-json-viewer";
import { ExecutionDebugPanel } from "@/components/execution-debug-panel";
import { InputSourcesTable } from "@/components/input-sources-table";
import { FlowRunner, type FlowRunnerSnapshot } from "@/components/flow-runner";
import { PromptTabs } from "@/components/prompt-tabs";

export function SuperadminFlowRunner({ flowId }: { flowId: string }) {
  const [snapshot, setSnapshot] = useState<FlowRunnerSnapshot | null>(null);

  const inputSources = useMemo(() => {
    if (!snapshot) return [];
    return (
      snapshot.debug?.input_sources ??
      (snapshot.currentForm?.fields || []).map((field) => ({
        field_id: field.id,
        label: field.label,
        source_type: "Current Form Input",
        source_name: "Current Form Input",
        path: `values.${field.id}`,
        value: snapshot.values[field.id],
      }))
    );
  }, [snapshot]);

  const promptTemplate = snapshot?.debug?.prompt_template ?? snapshot?.currentForm?.prompt ?? null;
  const resolvedPrompt = snapshot?.debug?.resolved_prompt ?? null;
  const modelConfiguration = snapshot?.debug?.model_configuration ?? snapshot?.currentForm?.model ?? null;
  const outputSchema = snapshot?.debug?.output_schema ?? {
    type: "object",
    properties: Object.fromEntries(
      (snapshot?.currentForm?.fields || []).map((field) => [
        field.id,
        {
          label: field.label,
          type: field.type,
          required: field.required,
          description: field.description,
          default: field.default,
          options: field.options,
        },
      ]),
    ),
  };
  const rawResponse = snapshot?.debug?.raw_response ?? snapshot?.stepResult ?? null;
  const executionDetails = snapshot
    ? {
        ...(snapshot.debug?.execution_details || {}),
        flow_name: snapshot.flow.name,
        current_step_id: snapshot.currentStep.id,
        current_step_name: snapshot.currentStep.name,
        current_step_sequence: snapshot.currentStep.sequence,
        current_step_index: snapshot.currentStepIndex,
      }
    : null;
  const runtimeState = snapshot
    ? {
        ...(snapshot.debug?.runtime_state || {}),
        status: snapshot.stepResult ? "completed" : "running",
        current_step: snapshot.currentStep.id,
        completed_steps: snapshot.stepResults.map((step) => step.id),
        pending_steps: snapshot.flow.steps
          .slice(snapshot.currentStepIndex + 1)
          .map((step) => step.id),
        context: snapshot.context,
      }
    : null;

  return (
    <div className="space-y-6">
      <FlowRunner flowId={flowId} onDebugSnapshot={setSnapshot} />

      <ExecutionDebugPanel
        sections={[
          {
            id: "input-sources",
            title: "Input Sources",
            defaultOpen: true,
            content: <InputSourcesTable sources={inputSources} />,
          },
          {
            id: "prompt",
            title: "Prompt",
            content: (
              <PromptTabs
                template={promptTemplate}
                rendered={resolvedPrompt}
              />
            ),
          },
          {
            id: "model-configuration",
            title: "Model Configuration",
            content: <DebugJsonViewer value={modelConfiguration} />,
          },
          {
            id: "output-schema",
            title: "Output Schema",
            content: <DebugJsonViewer value={outputSchema} />,
          },
          {
            id: "raw-response",
            title: "Raw Response",
            content: <DebugJsonViewer value={rawResponse} />,
          },
          {
            id: "execution-details",
            title: "Execution Details",
            defaultOpen: true,
            content: <DebugJsonViewer value={executionDetails} />,
          },
          {
            id: "runtime-state",
            title: "Runtime State",
            defaultOpen: true,
            content: <DebugJsonViewer value={runtimeState} />,
          },
        ]}
      />
    </div>
  );
}
