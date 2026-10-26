import React, { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { ChevronLeft, ChevronRight, RotateCcw, Zap, BookmarkCheck, Pencil, ShieldAlert } from "lucide-react";
import { FormattedResultView } from "@/components/formatted-result-view";
import { toast } from "sonner";
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "@/components/ui/accordion";
import { Badge, Button, Card, Field, Input } from "@/components/ui-kit";
import { Progress } from "@/components/ui/progress";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Textarea } from "@/components/ui/textarea";
import { apiGet, apiPost } from "@/lib/api";
import { logActivity } from "@/lib/activity";
import { ExportDropdown } from "@/components/export-dropdown";
import { useFavoriteModels, sortModelsWithFavorites, getDefaultModelId, getFavoriteModelIds, INITIAL_DEFAULT_MODELS } from "@/lib/favorite-models";

export type RuntimeField = {
  id: string;
  label: string;
  description?: string | null;
  type: "text" | "textarea" | "date" | "checkbox" | "radio" | "dropdown" | "multiselect" | "hidden";
  required?: boolean;
  default?: string | null;
  options?: Array<string | { label?: string; value?: string; theme?: string; description?: string }>;
  data_source?: { type: "step_output"; step_id: string; path: string } | null;
};

export type RuntimeForm = {
  id: string;
  name: string;
  description?: string | null;
  version: string;
  fields: RuntimeField[];
  prompt: { system: string; user: string };
  model?: { provider: string; name: string; temperature: number };
  output?: { type: string; schema?: any } | null;
  execution?: { mode: "interactive" | "automatic" } | null;
};

export type RuntimeStep = {
  id: string;
  sequence: number;
  name: string;
  prompt_form_id: string;
  input_bindings: Record<string, string>;
  dynamic_fields: string[];
  review: { required: boolean; editable: boolean };
  output?: { save_as: string; formats: string[] } | null;
  next: string | null;
};

export type RuntimeFlow = {
  id: string;
  version: string;
  name: string;
  description: string;
  runtime: { mode: "guided" | "automatic"; default_review_required: boolean };
  steps: RuntimeStep[];
};

type FlowExecuteStep = {
  id: string;
  name: string;
  sequence: number;
  prompt: string;
  result: string;
  completed: boolean;
  next: string | null;
  output?: { save_as: string; formats: string[] } | null;
};

type FlowExecuteResponse = {
  context: Record<string, unknown>;
  steps: FlowExecuteStep[];
  debug?: FlowExecutionDebug | null;
  diagnostic_capabilities?: string[];
};

export type FlowExecutionDebug = {
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

export type FlowRunnerSnapshot = {
  flow: RuntimeFlow;
  currentStep: RuntimeStep;
  currentForm: RuntimeForm | null;
  currentStepIndex: number;
  values: FieldState;
  context: Record<string, unknown>;
  stepPrompt: string;
  stepResult: string;
  stepResults: FlowExecuteStep[];
  debug: FlowExecutionDebug | null;
  diagnostic_capabilities: string[];
};

type FlowRunnerProps = {
  flowId: string;
  onDebugSnapshot?: (snapshot: FlowRunnerSnapshot | null) => void;
};

type FieldState = Record<string, string>;

function initialValue(field: RuntimeField): string {
  return field.default ?? "";
}

function fieldLabel(field: RuntimeField) {
  return field.description ? `${field.label} (${field.description})` : field.label;
}

function normalizeOption(option: NonNullable<RuntimeField["options"]>[number]) {
  if (typeof option === "string") {
    return { label: option, value: option, description: undefined as string | undefined };
  }
  const label = option.label ?? option.theme ?? option.value ?? "";
  const value = option.value ?? option.theme ?? label;
  return { label, value, description: option.description };
}

function resolvePath(obj: any, path: string): any {
  if (!obj || !path) return undefined;
  const parts = path.split(".");
  let current = obj;
  for (const part of parts) {
    if (current === null || current === undefined) return undefined;
    current = current[part];
  }
  return current;
}

function FieldEditor({
  field,
  value,
  onChange,
}: {
  field: RuntimeField;
  value: string;
  onChange: (value: string) => void;
}) {
  if (field.type === "hidden") return <input type="hidden" value={value} readOnly />;
  if (field.type === "textarea") {
    return (
      <Field label={fieldLabel(field)} hint={field.description || undefined}>
        <Textarea value={value} onChange={(e) => onChange(e.target.value)} rows={5} />
      </Field>
    );
  }
  if (field.type === "date") {
    return (
      <Field label={fieldLabel(field)} hint={field.description || undefined}>
        <Input type="date" value={value} onChange={(e) => onChange(e.target.value)} />
      </Field>
    );
  }
  if (field.type === "dropdown") {
    return (
      <Field label={fieldLabel(field)} hint={field.description || undefined}>
        <Select value={value} onValueChange={onChange}>
          <SelectTrigger>
            <SelectValue placeholder={`Select ${field.label.toLowerCase()}`} />
          </SelectTrigger>
          <SelectContent>
            {(field.options || []).map((option) => {
              const normalized = normalizeOption(option);
              return (
                <SelectItem key={normalized.value} value={normalized.value}>
                  {normalized.label}
                </SelectItem>
              );
            })}
          </SelectContent>
        </Select>
      </Field>
    );
  }
  if (field.type === "radio") {
    return (
      <Field label={fieldLabel(field)} hint={field.description || undefined}>
        <div className="space-y-2">
          {(field.options || []).map((option) => {
            const normalized = normalizeOption(option);
            return (
              <label key={normalized.value} className="flex items-start gap-2 text-sm text-foreground">
                <input
                  className="mt-0.5 h-4 w-4 rounded-full border-border"
                  type="radio"
                  name={field.id}
                  value={normalized.value}
                  checked={value === normalized.value}
                  required={field.required}
                  onChange={(e) => onChange(e.target.value)}
                />
                <span>
                  <span className="block">{normalized.label}</span>
                  {normalized.description ? (
                    <span className="block text-xs text-muted-foreground">{normalized.description}</span>
                  ) : null}
                </span>
              </label>
            );
          })}
        </div>
      </Field>
    );
  }
  if (field.type === "checkbox" || field.type === "multiselect") {
    if (field.options && field.options.length > 0) {
      const selected = value ? value.split(",").filter(Boolean) : [];
      return (
        <Field label={fieldLabel(field)} hint={field.description || undefined}>
          <div className="space-y-2 mt-2">
            {field.options.map((option) => {
              const normalized = normalizeOption(option);
              const isChecked = selected.includes(normalized.value);
              return (
                <label key={normalized.value} className="flex items-start gap-2 text-sm text-foreground">
                  <input
                    className="mt-0.5 h-4 w-4 rounded border-border"
                    type="checkbox"
                    checked={isChecked}
                    onChange={(e) => {
                      const newSelected = e.target.checked
                        ? [...selected, normalized.value]
                        : selected.filter((v) => v !== normalized.value);
                      onChange(newSelected.join(","));
                    }}
                  />
                  <span>
                    <span className="block">{normalized.label}</span>
                    {normalized.description ? (
                      <span className="block text-xs text-muted-foreground">{normalized.description}</span>
                    ) : null}
                  </span>
                </label>
              );
            })}
          </div>
        </Field>
      );
    }

    return (
      <label className="flex items-start gap-2 text-sm text-foreground">
        <input
          className="mt-0.5 h-4 w-4 rounded border-border"
          type="checkbox"
          checked={value === "true"}
          onChange={(e) => onChange(e.target.checked ? "true" : "false")}
        />
        {fieldLabel(field)}
      </label>
    );
  }
  return (
    <Field label={fieldLabel(field)} hint={field.description || undefined}>
      <Input value={value} onChange={(e) => onChange(e.target.value)} />
    </Field>
  );
}

interface ModelRegistryItem {
  id: string;
  name: string;
  provider: string;
  category: string;
  recommended?: boolean;
  custom_label?: string | null;
}

export function FlowRunner({ flowId, onDebugSnapshot }: FlowRunnerProps) {
  const [flow, setFlow] = useState<RuntimeFlow | null>(null);
  const [forms, setForms] = useState<Record<string, RuntimeForm>>({});
  const [currentStepIndex, setCurrentStepIndex] = useState(0);
  const [values, setValues] = useState<FieldState>({});
  const [context, setContext] = useState<Record<string, unknown>>({});
  const [stepResults, setStepResults] = useState<FlowExecuteStep[]>([]);
  const [stepPrompt, setStepPrompt] = useState<string>("");
  const [stepResult, setStepResult] = useState<string>("");
  const [stepDebug, setStepDebug] = useState<FlowExecutionDebug | null>(null);
  const [diagnosticCapabilities, setDiagnosticCapabilities] = useState<string[]>([]);
  const { favoriteIds } = useFavoriteModels();
  const [selectedModel, setSelectedModel] = useState<string>(() => getDefaultModelId());
  const [hasUserSelectedModel, setHasUserSelectedModel] = useState(false);
  const [modelsList, setModelsList] = useState<ModelRegistryItem[]>(INITIAL_DEFAULT_MODELS);
  const [loading, setLoading] = useState(true);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [activeAccordion, setActiveAccordion] = useState<string>("step-0");
  const [isEditingStepResult, setIsEditingStepResult] = useState(false);

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
    Promise.all([
      apiGet<{ models: ModelRegistryItem[]; labels?: Record<string, string> }>(
        `/models?flow_id=${encodeURIComponent(flowId)}`
      ).catch(() => null),
      apiGet<{ labels?: Record<string, string> }>("/organization/model-labels").catch(() => null),
    ]).then(([modelsRes, labelsRes]) => {
      const models = modelsRes?.models || [];
      if (models.length > 0) {
        const labelsMap: Record<string, string> = {
          ...(modelsRes?.labels || {}),
          ...(labelsRes?.labels || {}),
        };
        const enriched = models.map((m) => ({
          ...m,
          custom_label: m.custom_label || labelsMap[m.id] || null,
        }));
        setModelsList(enriched);
      }
    });
  }, [flowId]);

  useEffect(() => {
    setLoading(true);
    setHasUserSelectedModel(false);
    apiGet<RuntimeFlow>(`/flows/${flowId}`)
      .then((data) => {
        setFlow(data);
        const defaultMod = getDefaultModelId(getFavoriteModelIds(), modelsList);
        setSelectedModel(defaultMod);
        logActivity("viewed flow", data.name);
        setCurrentStepIndex(0);
        setValues({});
        setContext({});
        setStepResults([]);
        setStepPrompt("");
        setStepResult("");
        setStepDebug(null);
        setDiagnosticCapabilities([]);
        setActiveAccordion("step-0");
      })
      .catch((err: Error) => setError(err.message))
      .finally(() => setLoading(false));
  }, [flowId]);

  const currentStep = useMemo(
    () => flow?.steps.slice().sort((a, b) => a.sequence - b.sequence)[currentStepIndex] ?? null,
    [flow, currentStepIndex],
  );

  const currentForm = currentStep ? forms[currentStep.prompt_form_id] : undefined;

  useEffect(() => {
    if (!currentStep || forms[currentStep.prompt_form_id]) return;
    const formKey = currentStep.prompt_form_id;
    apiGet<RuntimeForm>(`/forms/${formKey}`)
      .then((form) => {
        setForms((prev) => ({ ...prev, [form.id]: form, [formKey]: form }));
        setValues((prev) => {
          const next = { ...prev };
          for (const field of form.fields) {
            if (next[field.id] === undefined) next[field.id] = initialValue(field);
          }
          return next;
        });
      })
      .catch((err: Error) => setError(err.message));
  }, [currentStep, forms]);

  useEffect(() => {
    if (!onDebugSnapshot) return;
    if (!flow || !currentStep) {
      onDebugSnapshot(null);
      return;
    }
    onDebugSnapshot({
      flow,
      currentStep,
      currentForm: currentForm ?? null,
      currentStepIndex,
      values,
      context,
      stepPrompt,
      stepResult,
      stepResults,
      debug: stepDebug,
      diagnostic_capabilities: diagnosticCapabilities,
    });
  }, [
    onDebugSnapshot,
    flow,
    currentStep,
    currentForm,
    currentStepIndex,
    values,
    context,
    stepPrompt,
    stepResult,
    stepResults,
    stepDebug,
    diagnosticCapabilities,
  ]);

  const renderedFields = useMemo(() => {
    if (!currentStep || !currentForm) return [];
    let fields = currentForm.fields;
    if (currentStep.dynamic_fields && currentStep.dynamic_fields.length > 0) {
      fields = currentStep.dynamic_fields
        .map((fieldId) => currentForm.fields.find((field) => field.id === fieldId))
        .filter(Boolean) as RuntimeField[];
    }

    return fields.map((field) => {
      if (field.data_source?.type === "step_output") {
        const { step_id, path } = field.data_source;

        // DEBUG: log full context to trace the bug
        // console.group(`[FlowRunner] Dynamic field "${field.id}" — step_id="${step_id}" path="${path}"`);
        // console.log("context.steps:", (context.steps as any));
        const contextSteps = context.steps as Record<string, any> | undefined;
        const stepState = contextSteps?.[step_id];
        // console.log(`context.steps["${step_id}"]`, stepState);
        // console.log("stepState.output:", stepState?.output);
        console.groupEnd();

        if (!stepState) {
          return { ...field, options: ["Loading options..."] };
        }

        const resolved = resolvePath(stepState.output, path);
        // console.log(`[FlowRunner] resolved path "${path}":`, resolved);
        if (Array.isArray(resolved)) {
          if (resolved.length === 0) {
            return { ...field, options: ["No options available."] };
          }
          return { ...field, options: resolved.map(String) };
        }

        return { ...field, options: ["Unable to load options."] };
      }
      return field;
    });
  }, [currentStep, currentForm, context]);

  // Dependent Field Invalidation
  useEffect(() => {
    setValues((prev) => {
      let changed = false;
      const next = { ...prev };

      for (const field of renderedFields) {
        if (field.data_source?.type === "step_output") {
          const selected = next[field.id];
          if (selected && field.options) {
            if (field.type === "checkbox" || field.type === "multiselect") {
              // For multiselect, filter out any selected items that are no longer in options
              const selectedItems = selected.split(",").filter(Boolean);
              const validItems = selectedItems.filter((item) => field.options!.includes(item));
              if (validItems.length !== selectedItems.length) {
                next[field.id] = validItems.join(",");
                changed = true;
              }
            } else {
              // For single select, clear if not exactly matching an option
              if (!field.options.includes(selected)) {
                next[field.id] = "";
                changed = true;
              }
            }
          }
        }
      }

      return changed ? next : prev;
    });
  }, [renderedFields]);

  async function runCurrentStep() {
    if (!flow || !currentStep) return;
    setRunning(true);
    setError(null);
    try {
      const currentModelObj = modelsList.find((m) => m.id === selectedModel);
      const response = await apiPost<FlowExecuteResponse>(`/flows/${flow.id}/execute`, {
        step_id: currentStep.id,
        values,
        context,
        model_override: {
          provider: currentModelObj?.provider || "openai",
          name: selectedModel,
          temperature: 0.7,
        },
      });
      const executed = response.steps[0];

      // DEBUG: log the raw response context
      // console.log("[FlowRunner] runCurrentStep — response.context:", JSON.stringify(response.context));
      // console.log("[FlowRunner] runCurrentStep — response.context.steps:", response.context.steps);

      setContext(response.context);
      setStepPrompt(executed.prompt);
      setStepResult(executed.result);
      setStepDebug(response.debug ?? null);
      setDiagnosticCapabilities(response.diagnostic_capabilities ?? []);
      setStepResults((prev) => [...prev.slice(0, currentStepIndex), executed]);
      setActiveAccordion(`step-${currentStepIndex}`);
      setIsEditingStepResult(false);
      logActivity("executed flow step", `${flow.name} - ${currentStep.name}`);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to execute flow step");
    } finally {
      setRunning(false);
    }
  }

  function continueToNext() {
    if (!currentStep) return;

    // DEBUG: log context before transition
    // console.log("[FlowRunner] continueToNext — context BEFORE:", JSON.stringify(context));
    // console.log("[FlowRunner] continueToNext — context.steps BEFORE:", context.steps);

    setContext((latestContext) => {
      // console.log("[FlowRunner] continueToNext — latestContext inside updater:", JSON.stringify(latestContext));
      // console.log("[FlowRunner] continueToNext — latestContext.steps:", latestContext.steps);
      if (currentStep.output?.save_as) {
        return { ...latestContext, [currentStep.output.save_as]: stepResult };
      }
      return latestContext;
    });
    const nextIndex = currentStepIndex + 1;
    if (flow && nextIndex < flow.steps.length) {
      setCurrentStepIndex(nextIndex);
      setStepPrompt("");
      setStepResult("");
      setActiveAccordion(`step-${nextIndex}`);
    }
  }

  function goToPrevious() {
    if (currentStepIndex <= 0) return;
    const prevIndex = currentStepIndex - 1;
    setCurrentStepIndex(prevIndex);
    // Restore the prompt/result for that step if we have it
    const prevResult = stepResults[prevIndex];
    setStepPrompt(prevResult?.prompt ?? "");
    setStepResult(prevResult?.result ?? "");
    setStepDebug(null);
    setActiveAccordion(`step-${prevIndex}`);
    setError(null);
  }

  if (loading) return <Card className="p-5">Loading flow...</Card>;
  if (error && !flow) {
    const isForbidden =
      error.includes("403") ||
      error.toLowerCase().includes("forbidden") ||
      error.toLowerCase().includes("access denied");
    return (
      <Card className="p-8 text-center max-w-lg mx-auto my-12 border border-border shadow-xs">
        <div className="mx-auto flex h-12 w-12 items-center justify-center rounded-full bg-destructive/10 text-destructive mb-4">
          <ShieldAlert className="h-6 w-6" />
        </div>
        <h3 className="text-base font-semibold text-foreground">
          {isForbidden ? "Flow Access Restricted" : "Unable to Load Flow"}
        </h3>
        <p className="mt-2 text-xs text-muted-foreground leading-relaxed">
          {isForbidden
            ? "You do not currently have access to run this flow. Access to this workflow must be assigned to your organization by an administrator."
            : error}
        </p>
        <div className="mt-6 flex items-center justify-center gap-3">
          <Link to="/flows">
            <Button size="sm" variant="secondary">
              Back to Flows
            </Button>
          </Link>
          <Link to="/dashboard">
            <Button size="sm" variant="ghost">
              Go to Dashboard
            </Button>
          </Link>
        </div>
      </Card>
    );
  }
  if (!flow || !currentStep) return <Card className="p-5">Flow not found.</Card>;

  const sortedSteps = flow.steps.slice().sort((a, b) => a.sequence - b.sequence);
  const isLastStep = currentStepIndex === sortedSteps.length - 1;
  const isFirstStep = currentStepIndex === 0;
  const isExecuted = !!stepResult;
  const progress = flow.steps.length === 0 ? 0 : ((currentStepIndex + 1) / flow.steps.length) * 100;

  // Check if all required rendered fields have a value
  const hasRequiredValues = renderedFields
    .filter((f) => f.required && f.type !== "hidden")
    .every((f) => {
      const val = values[f.id];
      return val !== undefined && val !== "" && val !== "Loading options...";
    });


  return (
    <div className="space-y-5">
      {/* Header progress card */}
      <Card className="p-5">
        <div className="flex items-start justify-between gap-4">
          <div>
            <div className="text-lg font-semibold">{flow.name}</div>
            {flow.description && <p className="mt-1 text-sm text-muted-foreground">{flow.description}</p>}
          </div>
          <Badge tone="neutral">
            {currentStepIndex + 1} / {flow.steps.length}
          </Badge>
        </div>
        <div className="mt-4 space-y-1.5">
          <Progress value={progress} />
          <div className="flex justify-between text-xs text-muted-foreground">
            <span>{currentStep.name}</span>
            {/* <span>{Math.round(progress)}% complete</span> */}
          </div>
        </div>
        {/* Step pills */}
        <div className="mt-4 flex gap-2 flex-wrap">
          {sortedSteps.map((step, index) => (
            <div
              key={step.id}
              className={`flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-medium border transition-colors ${index === currentStepIndex
                ? "bg-foreground text-background border-foreground"
                : index < currentStepIndex
                  ? "bg-muted text-muted-foreground border-border"
                  : "text-muted-foreground border-border"
                }`}
            >
              <span
                className={`h-1.5 w-1.5 rounded-full ${index < currentStepIndex
                  ? "bg-green-500"
                  : index === currentStepIndex
                    ? "bg-background"
                    : "bg-muted-foreground/30"
                  }`}
              />
              {step.name}
            </div>
          ))}
        </div>
      </Card>

      {/* Main step card */}
      <Card className="p-6 space-y-6">
        <div className="flex items-center justify-between">
          <div>
            <div className="text-xs uppercase tracking-wider text-muted-foreground mb-1">Step {currentStepIndex + 1}</div>
            <div className="text-xl font-semibold">{currentStep.name}</div>
          </div>
          {isExecuted && (
            <Badge tone="neutral">
              {isLastStep ? "Completed" : "Executed"}
            </Badge>
          )}
        </div>

        {/* Fields */}
        <div className="grid grid-cols-1 gap-5">
          {renderedFields.map((field) => (
            <FieldEditor
              key={field.id}
              field={field}
              value={values[field.id] ?? initialValue(field)}
              onChange={(value) => setValues((prev) => ({ ...prev, [field.id]: value }))}
            />
          ))}
        </div>

        {/* Error */}
        {error && (
          <div className="rounded-lg border border-red-200 bg-red-50 p-3 text-sm text-red-700">
            {error}
          </div>
        )}

        {stepPrompt && (
          <details className="group">
            <summary className="cursor-pointer text-xs uppercase tracking-wide text-muted-foreground hover:text-foreground select-none">
              Rendered prompt
            </summary>
            <div className="mt-2 rounded-lg border border-border bg-muted/30 p-4">
              <pre className="whitespace-pre-wrap text-sm">{stepPrompt}</pre>
            </div>
          </details>
        )}

        {/* Result section */}
        {isExecuted && (
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <div className="text-xs uppercase tracking-wide text-muted-foreground">
                  {isLastStep ? "Final Output" : "Result"}
                </div>
                {isLastStep && <span className="text-xs text-green-600 font-medium">✓ Flow complete</span>}
              </div>
              {!isLastStep && (
                <button
                  type="button"
                  onClick={() => setIsEditingStepResult((prev) => !prev)}
                  className="text-xs text-muted-foreground hover:text-foreground flex items-center gap-1.5 transition-colors"
                >
                  <Pencil className="h-3 w-3" />
                  <span>{isEditingStepResult ? "View Formatted" : "Edit Raw Text"}</span>
                </button>
              )}
            </div>

            {isEditingStepResult ? (
              <Textarea
                value={stepResult}
                onChange={(e) => setStepResult(e.target.value)}
                rows={8}
                className="font-mono text-sm"
              />
            ) : (
              <div className="rounded-xl border border-border bg-muted/20 p-5 overflow-hidden">
                <FormattedResultView content={stepResult} />
              </div>
            )}
          </div>
        )}

        {/* Action buttons */}
        {/* Action buttons */}
        <div className="flex items-center justify-between gap-3 border-t border-border pt-4">
          {/* Previous */}
          <Button
            variant="ghost"
            onClick={goToPrevious}
            disabled={isFirstStep || running}
            className="gap-2"
          >
            <ChevronLeft className="h-4 w-4" />
            Previous
          </Button>

          {/* Execute + Next */}
          <div className="flex items-center gap-2">
            <Select
              value={selectedModel}
              onValueChange={(val) => {
                setSelectedModel(val);
                setHasUserSelectedModel(true);
              }}
            >
              <SelectTrigger className="min-w-[190px] max-w-[320px] h-10 text-xs [&>span]:flex [&>span]:items-center [&>span]:gap-1.5 [&>span]:truncate">
                <SelectValue placeholder="Model">
                  {(() => {
                    const selectedObj = modelsList.find((m) => m.id === selectedModel);
                    if (!selectedObj) return undefined;
                    const isFav = favoriteIds.includes(selectedModel);
                    return (
                      <span className="flex items-center gap-1.5 truncate">
                        {isFav && <span className="text-amber-500 font-bold shrink-0">★</span>}
                        {selectedObj.custom_label ? (
                          <span className="flex items-center gap-1.5 truncate">
                            <span className="font-semibold text-foreground truncate">
                              {selectedObj.custom_label}
                            </span>
                            <span className="text-muted-foreground text-[10px] shrink-0 font-normal">
                              ({selectedObj.name})
                            </span>
                          </span>
                        ) : (
                          <span className="font-medium text-foreground truncate">{selectedObj.name}</span>
                        )}
                      </span>
                    );
                  })()}
                </SelectValue>
              </SelectTrigger>
              <SelectContent className="max-h-60 overflow-y-auto">
                {prioritizedModels.map((m) => {
                  const isFav = favoriteIds.includes(m.id);
                  const displayTitle = m.custom_label ? `${m.custom_label} (${m.name})` : m.name;
                  return (
                    <SelectItem
                      key={m.id}
                      value={m.id}
                      textValue={displayTitle}
                    >
                      <div className="flex items-center justify-between gap-2 py-0.5 w-full">
                        <div className="flex items-center gap-1.5 min-w-0">
                          {isFav && <span className="text-amber-500 font-bold shrink-0">★</span>}
                          {m.custom_label ? (
                            <span className="flex items-center gap-1.5 min-w-0">
                              <span className="font-semibold text-foreground truncate">{m.custom_label}</span>
                              <span className="text-muted-foreground text-[10px] shrink-0">({m.name})</span>
                              <span className="px-1.5 py-0.2 rounded text-[9px] bg-emerald-500/10 text-emerald-700 dark:text-emerald-400 border border-emerald-500/20 font-medium shrink-0">
                                Org Label
                              </span>
                            </span>
                          ) : (
                            <span className="truncate">{m.name}</span>
                          )}
                        </div>
                        <span className="text-muted-foreground text-[10px] font-mono uppercase shrink-0 pl-1">
                          [{m.provider}]
                        </span>
                      </div>
                    </SelectItem>
                  );
                })}
              </SelectContent>
            </Select>

            <Button
              variant="primary"
              onClick={() => void runCurrentStep()}
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

            {!isLastStep && (
              <Button
                variant="outline"
                onClick={continueToNext}
                disabled={!isExecuted || running}
                className="gap-2"
              >
                Next
                <ChevronRight className="h-4 w-4" />
              </Button>
            )}
          </div>
        </div>
      </Card>

      {/* Previous step outputs accordion */}
      {stepResults.length > 0 && (
        <Card className="p-5">
          <div className="text-sm font-medium mb-3">Step history</div>
          <Accordion type="single" collapsible value={activeAccordion} onValueChange={setActiveAccordion}>
            {stepResults.map((step, index) => (
              <AccordionItem key={step.id} value={`step-${index}`}>
                <AccordionTrigger>
                  <div className="flex items-center gap-2">
                    <span className="h-2 w-2 rounded-full bg-green-500" />
                    <span>{step.name}</span>
                  </div>
                </AccordionTrigger>
                <AccordionContent>
                  <div className="space-y-3">
                    {/* <details className="group">
                      <summary className="cursor-pointer text-xs uppercase tracking-wide text-muted-foreground hover:text-foreground select-none">
                        Prompt
                      </summary>
                      <pre className="mt-2 whitespace-pre-wrap text-sm bg-muted/30 rounded p-3">{step.prompt}</pre>
                    </details> */}
                    <div>
                      <div className="flex items-center justify-between mb-2">
                        <div className="text-xs uppercase tracking-wide text-muted-foreground">Result</div>
                        <div className="flex items-center gap-2">
                          <ExportDropdown title={`${flow?.name || "Flow"} - ${step.name}`} text={step.result} />
                          <Button
                            size="sm"
                            variant="secondary"
                            className="gap-1.5 text-xs"
                            onClick={async () => {
                              try {
                                await apiPost("/saved-results", {
                                  source_type: "flow",
                                  source_id: flow?.id || flowId,
                                  source_name: `${flow?.name || "Flow"} - ${step.name}`,
                                  input_summary: { step_id: step.id, values },
                                  output_text: step.result,
                                });
                                toast.success("Flow step result saved to Saved Results!");
                              } catch (e: any) {
                                toast.error(e.message || "Failed to save result");
                              }
                            }}
                          >
                            <BookmarkCheck className="h-3.5 w-3.5" />
                            Save Result
                          </Button>
                        </div>
                      </div>
                      <div className="rounded-xl border border-border bg-muted/20 p-4 overflow-hidden">
                        <FormattedResultView content={step.result} />
                      </div>
                    </div>
                    {step.output && (
                      <div className="text-xs text-muted-foreground">
                        Saved as <code className="font-mono">{step.output.save_as}</code> in {step.output.formats.join(", ")}
                      </div>
                    )}
                  </div>
                </AccordionContent>
              </AccordionItem>
            ))}
          </Accordion>
        </Card>
      )}
    </div>
  );
}
