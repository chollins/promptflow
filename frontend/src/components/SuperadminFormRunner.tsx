import { useMemo, useState } from "react";
import { DebugJsonViewer } from "@/components/debug-json-viewer";
import { ExecutionDebugPanel } from "@/components/execution-debug-panel";
import { InputSourcesTable } from "@/components/input-sources-table";
import { FormRunner, type FormRunnerSnapshot } from "@/components/form-runner";
import { PromptTabs } from "@/components/prompt-tabs";

export function SuperadminFormRunner({ formId }: { formId: string }) {
  const [snapshot, setSnapshot] = useState<FormRunnerSnapshot | null>(null);

  const inputSources = useMemo(() => {
    if (!snapshot) return [];
    return (
      snapshot.debug?.input_sources ??
      snapshot.form.fields.map((field) => ({
        field_id: field.id,
        label: field.label,
        source_type: "Current Form Input",
        source_name: "Current Form Input",
        path: `values.${field.id}`,
        value: snapshot.values[field.id],
      }))
    );
  }, [snapshot]);

  const promptTemplate = snapshot?.debug?.prompt_template ?? snapshot?.form.prompt ?? null;
  const resolvedPrompt = snapshot?.debug?.resolved_prompt ?? null;
  const modelConfiguration = snapshot?.debug?.model_configuration ?? snapshot?.form.model ?? null;
  const outputSchema = snapshot?.debug?.output_schema ?? {
    type: "object",
    properties: Object.fromEntries(
      (snapshot?.form.fields || []).map((field) => [
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
  const rawResponse = snapshot?.debug?.raw_response ?? snapshot?.result ?? null;
  const executionDetails = snapshot
    ? {
        ...(snapshot.debug?.execution_details || {}),
        form_name: snapshot.form.name,
        form_id: snapshot.form.id,
      }
    : null;
  const runtimeState = snapshot
    ? {
        ...(snapshot.debug?.runtime_state || {}),
        status: snapshot.result ? "completed" : "running",
        current_form_id: snapshot.form.id,
        values: snapshot.values,
      }
    : null;

  return (
    <div className="space-y-6">
      <FormRunner formId={formId} onDebugSnapshot={setSnapshot} />

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
