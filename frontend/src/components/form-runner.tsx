import { useEffect, useState } from "react";
import DynamicForm, { type PromptForm } from "@/components/DynamicForm";
import { Button, Card } from "@/components/ui-kit";
import { apiGet, apiPost } from "@/lib/api";
import { logActivity } from "@/lib/activity";
import { RotateCcw, Zap } from "lucide-react";

type FormExecuteResult = {
  form_id: string;
  prompt: string;
  result: string;
  values: Record<string, string>;
  debug?: FormExecutionDebug | null;
};

export type FormExecutionDebug = {
  input_sources?: Array<{
    field_id: string;
    label: string;
    source_type: string;
    source_name: string;
    path: string;
    value: unknown;
  }>;
  prompt_template?: { system: string; user: string } | null;
  resolved_prompt?: { system: string; user: string } | null;
  model_configuration?: Record<string, unknown> | null;
  output_schema?: unknown;
  raw_response?: unknown;
  execution_details?: Record<string, unknown> | null;
  runtime_state?: Record<string, unknown> | null;
};

export type FormRunnerSnapshot = {
  form: PromptForm;
  values: Record<string, string>;
  prompt: string;
  result: string;
  debug: FormExecutionDebug | null;
};

type FormRunnerProps = {
  formId: string;
  onDebugSnapshot?: (snapshot: FormRunnerSnapshot | null) => void;
};

function formatLlmResult(text: string) {
  return text
    .replace(/\r\n/g, "\n")
    .replace(/^\s*#{1,6}\s*/gm, "")
    .replace(/^\s*[-*+]\s+/gm, "")
    .replace(/^\s*\d+\.\s+/gm, "")
    .replace(/\*\*(.*?)\*\*/g, "$1")
    .replace(/\*(.*?)\*/g, "$1")
    .replace(/`([^`]+)`/g, "$1")
    .replace(/^\s*>\s?/gm, "")
    .trim();
}

export function FormRunner({ formId, onDebugSnapshot }: FormRunnerProps) {
  const [form, setForm] = useState<PromptForm | null>(null);
  const [values, setValues] = useState<Record<string, string>>({});
  const [prompt, setPrompt] = useState("");
  const [result, setResult] = useState("");
  const [debug, setDebug] = useState<FormExecutionDebug | null>(null);
  const [loading, setLoading] = useState(true);
  const [running, setRunning] = useState(false);
  const [isExecuted, setIsExecuted] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let active = true;
    setLoading(true);
    setError(null);
    apiGet<PromptForm>(`/forms/${formId}`)
      .then((data) => {
        if (!active) return;
        setForm(data);
        logActivity("viewed form", data.name);
        setIsExecuted(false);
        setDebug(null);
        const next: Record<string, string> = {};
        for (const field of data.fields) {
          next[field.id] = field.type === "checkbox" ? "false" : (field.default ?? "");
        }
        setValues(next);
      })
      .catch((err: Error) => active && setError(err.message))
      .finally(() => active && setLoading(false));

    return () => {
      active = false;
    };
  }, [formId]);

  async function runForm(nextValues: Record<string, string>) {
    if (!form) return;
    setRunning(true);
    setError(null);
    setValues(nextValues);
    try {
      const response = await apiPost<FormExecuteResult>(
        `/forms/${form.id}/execute`,
        {
          values: nextValues,
        },
      );

      setPrompt(response.prompt);
      setResult(response.result);
      setDebug(response.debug ?? null);
      setIsExecuted(true);
      logActivity("executed form", form.name);

    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to execute form");
    } finally {
      setRunning(false);
    }
  }

  useEffect(() => {
    if (!onDebugSnapshot) return;
    if (!form) {
      onDebugSnapshot(null);
      return;
    }
    onDebugSnapshot({
      form,
      values,
      prompt,
      result,
      debug,
    });
  }, [onDebugSnapshot, form, values, prompt, result, debug]);

  if (loading) {
    return <Card className="p-5">Loading form...</Card>;
  }

  if (error) {
    return <Card className="p-5 text-sm text-red-600">{error}</Card>;
  }

  if (!form) {
    return <Card className="p-5">Form not found.</Card>;
  }

  const hasRequiredValues = form.fields
  .filter((field) => field.required)
  .every((field) => {
    const value = values[field.id];
    return value !== undefined && value.trim() !== "";
  });

  return (
    <div className="space-y-6">
      <Card className="p-5">
      <DynamicForm
        form={form}
        values={values}
        onValuesChange={setValues}
      />
  
      <div className="flex items-center gap-2 pt-2">
        <Button
          variant="default"
          onClick={() => void runForm(values)}
          disabled={running || !hasRequiredValues}
          className="gap-2 bg-black text-white hover:bg-black/90"
        >
          {running ? (
            <>
              <span className="h-4 w-4 animate-spin rounded-full border-2 border-current border-t-transparent" />
              Running...
            </>
          ) : isExecuted ? (
            <>
              <RotateCcw className="h-4 w-4" />
              Execute again
            </>
          ) : (
            <>
              <Zap className="h-4 w-4" />
              Execute
            </>
          )}
        </Button>
      </div>
      </Card>

      {prompt && (
        <Card className="p-5">
          <div className="text-sm font-medium">Rendered prompt</div>
          <pre className="mt-3 max-h-80 overflow-auto whitespace-pre-wrap rounded-lg bg-muted p-4 text-sm">
            {prompt}
          </pre>
        </Card>
      )}

      {result && (
        <Card className="p-5">
          <div className="text-sm font-medium">LLM result</div>
          <div className="mt-3 rounded-lg border border-border bg-background p-4 text-sm leading-7 whitespace-pre-wrap">
            {formatLlmResult(result)}
          </div>
        </Card>
      )}
    </div>
  );
}
