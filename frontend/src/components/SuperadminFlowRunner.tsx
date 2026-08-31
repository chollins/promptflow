import { useMemo, useState } from "react";
import { DebugJsonViewer } from "@/components/debug-json-viewer";
import { ExecutionDebugPanel } from "@/components/execution-debug-panel";
import { InputSourcesTable } from "@/components/input-sources-table";
import { FlowRunner, type FlowRunnerSnapshot } from "@/components/flow-runner";
import { PromptTabs } from "@/components/prompt-tabs";

export function SuperadminFlowRunner({ flowId }: { flowId: string }) {
  const [snapshot, setSnapshot] = useState<FlowRunnerSnapshot | null>(null);

  const capabilities = snapshot?.diagnostic_capabilities ?? [];
  const debug = snapshot?.debug ?? {};
  const sections: any[] = [];

  if (capabilities.includes("input_sources") && debug.input_sources) {
    sections.push({
      id: "input-sources",
      title: "Input Sources",
      defaultOpen: true,
      content: <InputSourcesTable sources={debug.input_sources} />,
    });
  }

  if (capabilities.includes("prompts") && (debug.prompt_template || debug.resolved_prompt)) {
    sections.push({
      id: "prompt",
      title: "Prompt",
      content: (
        <PromptTabs
          template={debug.prompt_template}
          rendered={debug.resolved_prompt}
        />
      ),
    });
  }

  if (capabilities.includes("model") && debug.model_configuration) {
    sections.push({
      id: "model-configuration",
      title: "Model Configuration",
      content: <DebugJsonViewer value={debug.model_configuration} />,
    });
  }

  if (capabilities.includes("output_schema") && debug.output_schema) {
    sections.push({
      id: "output-schema",
      title: "Output Schema",
      content: <DebugJsonViewer value={debug.output_schema} />,
    });
  }

  if (capabilities.includes("raw_response") && debug.raw_response) {
    sections.push({
      id: "raw-response",
      title: "Raw Response",
      content: <DebugJsonViewer value={debug.raw_response} />,
    });
  }

  if (capabilities.includes("structured_output") && snapshot?.stepResult) {
    // Only show structured_output if result is valid JSON
    let parsed: any = null;
    try {
      parsed = JSON.parse(snapshot.stepResult);
    } catch {
      // Not JSON
    }
    if (parsed) {
      sections.push({
        id: "structured-output",
        title: "JSON / Structured Output",
        content: <DebugJsonViewer value={parsed} />,
      });
    }
  }

  if (capabilities.includes("execution") && (debug.execution_details || debug.runtime_state)) {
    if (debug.execution_details) {
      sections.push({
        id: "execution-details",
        title: "Execution Details",
        defaultOpen: true,
        content: <DebugJsonViewer value={debug.execution_details} />,
      });
    }
    if (debug.runtime_state) {
      sections.push({
        id: "runtime-state",
        title: "Runtime State",
        defaultOpen: true,
        content: <DebugJsonViewer value={debug.runtime_state} />,
      });
    }
  }

  return (
    <div className="space-y-6">
      <FlowRunner flowId={flowId} onDebugSnapshot={setSnapshot} />

      {sections.length > 0 && (
        <ExecutionDebugPanel sections={sections} />
      )}
    </div>
  );
}
