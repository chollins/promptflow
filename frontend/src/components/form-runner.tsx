import { useEffect, useMemo, useState } from "react";
import DynamicForm, { type PromptForm } from "@/components/DynamicForm";
import { Button, Card } from "@/components/ui-kit";
import { apiGet, apiPost } from "@/lib/api";
import { logActivity } from "@/lib/activity";
import { RotateCcw, Zap, BookmarkCheck } from "lucide-react";
import { toast } from "sonner";
import { ExportDropdown } from "@/components/export-dropdown";
import { FormattedResultView } from "@/components/formatted-result-view";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { useFavoriteModels, sortModelsWithFavorites, getDefaultModelId, getFavoriteModelIds, INITIAL_DEFAULT_MODELS } from "@/lib/favorite-models";

type FormExecuteResult = {
  form_id: string;
  prompt: string;
  result: string;
  values: Record<string, string>;
  debug?: FormExecutionDebug | null;
  diagnostic_capabilities?: string[];
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
  diagnostic_capabilities: string[];
};

type FormRunnerProps = {
  formId: string;
  onDebugSnapshot?: (snapshot: FormRunnerSnapshot | null) => void;
};


export function FormRunner({ formId, onDebugSnapshot }: FormRunnerProps) {
  const [form, setForm] = useState<PromptForm | null>(null);
  const [values, setValues] = useState<Record<string, string>>({});
  const [prompt, setPrompt] = useState("");
  const [result, setResult] = useState("");
  const [debug, setDebug] = useState<FormExecutionDebug | null>(null);
  const [diagnosticCapabilities, setDiagnosticCapabilities] = useState<string[]>([]);
  const [loading, setLoading] = useState(true);
  const [running, setRunning] = useState(false);
  const [isExecuted, setIsExecuted] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const { favoriteIds } = useFavoriteModels();
  const [selectedModel, setSelectedModel] = useState<string>(() => getDefaultModelId());
  const [hasUserSelectedModel, setHasUserSelectedModel] = useState(false);
  const [modelsList, setModelsList] = useState<Array<{ id: string; name: string; provider: string; category: string }>>(INITIAL_DEFAULT_MODELS);

  const prioritizedModels = useMemo(() => {
    return sortModelsWithFavorites(modelsList, favoriteIds);
  }, [modelsList, favoriteIds]);

  // Default to favorite model if there is one; if no favorite, default to gpt-4o
  useEffect(() => {
    if (!hasUserSelectedModel) {
      setSelectedModel(getDefaultModelId(favoriteIds, modelsList));
    }
  }, [favoriteIds, modelsList, hasUserSelectedModel]);

  useEffect(() => {
    apiGet<{ models: Array<{ id: string; name: string; provider: string; category: string }> }>("/models")
      .then((res) => {
        if (res && res.models && res.models.length > 0) {
          setModelsList(res.models);
        }
      })
      .catch(() => {});
  }, []);

  useEffect(() => {
    let active = true;
    setLoading(true);
    setError(null);
    setHasUserSelectedModel(false);
    apiGet<PromptForm>(`/forms/${formId}`)
      .then((data) => {
        if (!active) return;
        setForm(data);
        const defaultMod = getDefaultModelId(getFavoriteModelIds(), modelsList);
        setSelectedModel(defaultMod);
        logActivity("viewed form", data.name);
        setIsExecuted(false);
        setDebug(null);
        setDiagnosticCapabilities([]);
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
          model_override: selectedModel,
        },
      );

      setPrompt(response.prompt);
      setResult(response.result);
      setDebug(response.debug ?? null);
      setDiagnosticCapabilities(response.diagnostic_capabilities ?? []);
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
      diagnostic_capabilities: diagnosticCapabilities,
    });
  }, [onDebugSnapshot, form, values, prompt, result, debug, diagnosticCapabilities]);

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
          <Select
            value={selectedModel}
            onValueChange={(val) => {
              setSelectedModel(val);
              setHasUserSelectedModel(true);
            }}
          >
            <SelectTrigger className="w-[170px] h-10 text-xs">
              <SelectValue placeholder="Model" />
            </SelectTrigger>
            <SelectContent className="max-h-60 overflow-y-auto">
              {prioritizedModels.map((m) => {
                const isFav = favoriteIds.includes(m.id);
                return (
                  <SelectItem key={m.id} value={m.id}>
                    <span className="flex items-center gap-1.5">
                      {isFav && <span className="text-amber-500 font-bold">★</span>}
                      <span>{m.name}</span>
                      <span className="text-muted-foreground text-[10px]">({m.provider})</span>
                    </span>
                  </SelectItem>
                );
              })}
            </SelectContent>
          </Select>

          <Button
            variant="primary"
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
        <Card className="p-5 space-y-3">
          <div className="flex items-center justify-between">
            <div className="text-sm font-medium">LLM result</div>
            <div className="flex items-center gap-2">
              <ExportDropdown title={form?.name || "Form Output"} text={result} />
              <Button
                size="sm"
                variant="secondary"
                className="gap-1.5 text-xs"
                onClick={async () => {
                  try {
                    await apiPost("/saved-results", {
                      source_type: "form",
                      source_id: form?.id || formId,
                      source_name: form?.name || "Form Output",
                      input_summary: values,
                      output_text: result,
                    });
                    toast.success("Output saved to Saved Results!");
                  } catch (e: any) {
                    toast.error(e.message || "Failed to save output");
                  }
                }}
              >
                <BookmarkCheck className="h-3.5 w-3.5" />
                Save Result
              </Button>
            </div>
          </div>
          <div className="rounded-lg border border-border bg-background p-4">
            <FormattedResultView content={result} />
          </div>
        </Card>
      )}
    </div>
  );
}
