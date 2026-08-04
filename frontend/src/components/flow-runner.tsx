import { useEffect, useMemo, useState } from "react";
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "@/components/ui/accordion";
import { Badge, Button, Card, Field, Input } from "@/components/ui-kit";
import { Progress } from "@/components/ui/progress";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Textarea } from "@/components/ui/textarea";
import { apiGet, apiPost } from "@/lib/api";

export type RuntimeField = {
  id: string;
  label: string;
  description?: string | null;
  type: "text" | "textarea" | "checkbox" | "radio" | "dropdown" | "hidden";
  required?: boolean;
  default?: string | null;
  options?: string[];
};

export type RuntimeForm = {
  id: string;
  name: string;
  description?: string | null;
  version: string;
  fields: RuntimeField[];
  prompt: { system: string; user: string };
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
};

type FlowRunnerProps = {
  flowId: string;
};

type FieldState = Record<string, string>;

function initialValue(field: RuntimeField): string {
  return field.default ?? "";
}

function fieldLabel(field: RuntimeField) {
  return field.description ? `${field.label} (${field.description})` : field.label;
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
  if (field.type === "dropdown" || field.type === "radio") {
    return (
      <Field label={fieldLabel(field)} hint={field.description || undefined}>
        <Select value={value} onValueChange={onChange}>
          <SelectTrigger>
            <SelectValue placeholder={`Select ${field.label.toLowerCase()}`} />
          </SelectTrigger>
          <SelectContent>
            {(field.options || []).map((option) => (
              <SelectItem key={option} value={option}>
                {option}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </Field>
    );
  }
  if (field.type === "checkbox") {
    return (
      <label className="flex items-center gap-2 text-sm text-foreground">
        <input
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

export function FlowRunner({ flowId }: FlowRunnerProps) {
  const [flow, setFlow] = useState<RuntimeFlow | null>(null);
  const [forms, setForms] = useState<Record<string, RuntimeForm>>({});
  const [currentStepIndex, setCurrentStepIndex] = useState(0);
  const [values, setValues] = useState<FieldState>({});
  const [context, setContext] = useState<Record<string, unknown>>({});
  const [stepResults, setStepResults] = useState<FlowExecuteStep[]>([]);
  const [stepPrompt, setStepPrompt] = useState<string>("");
  const [stepResult, setStepResult] = useState<string>("");
  const [loading, setLoading] = useState(true);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [activeAccordion, setActiveAccordion] = useState<string>("step-0");

  useEffect(() => {
    setLoading(true);
    apiGet<RuntimeFlow>(`/flows/${flowId}`)
      .then((data) => {
        setFlow(data);
        setCurrentStepIndex(0);
        setValues({});
        setContext({});
        setStepResults([]);
        setStepPrompt("");
        setStepResult("");
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
    apiGet<RuntimeForm>(`/forms/${currentStep.prompt_form_id}`)
      .then((form) => {
        setForms((prev) => ({ ...prev, [form.id]: form }));
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

  const renderedFields = useMemo(() => {
    if (!currentStep || !currentForm) return [];
    if (currentStep.dynamic_fields.length > 0) {
      return currentStep.dynamic_fields
        .map((fieldId) => currentForm.fields.find((field) => field.id === fieldId))
        .filter(Boolean) as RuntimeField[];
    }
    return currentForm.fields;
  }, [currentStep, currentForm]);

  async function runCurrentStep() {
    if (!flow || !currentStep) return;
    setRunning(true);
    setError(null);
    try {
      const response = await apiPost<FlowExecuteResponse>(`/flows/${flow.id}/execute`, {
        step_id: currentStep.id,
        values,
        context,
      });
      const executed = response.steps[0];
      setContext(response.context);
      setStepPrompt(executed.prompt);
      setStepResult(executed.result);
      setStepResults((prev) => [...prev.slice(0, currentStepIndex), executed]);
      setActiveAccordion(`step-${currentStepIndex}`);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to execute flow step");
    } finally {
      setRunning(false);
    }
  }

  function continueToNext() {
    if (!currentStep) return;
    const nextContext = { ...context };
    if (currentStep.output?.save_as) {
      nextContext[currentStep.output.save_as] = stepResult;
    }
    setContext(nextContext);
    const nextIndex = currentStepIndex + 1;
    if (flow && nextIndex < flow.steps.length) {
      setCurrentStepIndex(nextIndex);
      setStepPrompt("");
      setStepResult("");
      setActiveAccordion(`step-${nextIndex}`);
    }
  }

  if (loading) return <Card className="p-5">Loading flow...</Card>;
  if (error) return <Card className="p-5 text-sm text-red-600">{error}</Card>;
  if (!flow || !currentStep) return <Card className="p-5">Flow not found.</Card>;

  const progress = flow.steps.length === 0 ? 0 : ((currentStepIndex + 1) / flow.steps.length) * 100;

  return (
    <div className="space-y-6">
      <Card className="p-5">
        <div className="flex items-start justify-between gap-4">
          <div>
            <div className="text-lg font-medium">{flow.name}</div>
            <p className="mt-1 text-sm text-muted-foreground">{flow.description}</p>
          </div>
          <Badge tone="neutral">
            Step {currentStepIndex + 1} of {flow.steps.length}
          </Badge>
        </div>
        <div className="mt-4 space-y-2">
          <Progress value={progress} />
          <div className="text-xs text-muted-foreground">{Math.round(progress)}% complete</div>
        </div>
      </Card>

      <Card className="p-5 space-y-5">
        <div>
          <div className="text-sm font-medium">Current step</div>
          <div className="text-base">{currentStep.name}</div>
        </div>

        <div className="grid grid-cols-1 gap-4">
          {renderedFields.map((field) => (
            <FieldEditor
              key={field.id}
              field={field}
              value={values[field.id] ?? initialValue(field)}
              onChange={(value) => setValues((prev) => ({ ...prev, [field.id]: value }))}
            />
          ))}
        </div>

        <div className="flex gap-2">
          <Button onClick={runCurrentStep} disabled={running}>
            {running ? "Running..." : "Execute step"}
          </Button>
          <Button
            variant="secondary"
            onClick={continueToNext}
            disabled={!stepResult || currentStepIndex + 1 >= flow.steps.length}
          >
            Continue
          </Button>
        </div>

        {stepPrompt && (
          <div className="rounded-lg border border-border bg-muted/30 p-4 space-y-2">
            <div className="text-xs uppercase tracking-wide text-muted-foreground">Rendered prompt</div>
            <pre className="whitespace-pre-wrap text-sm">{stepPrompt}</pre>
          </div>
        )}

        {stepResult && (
          <div className="space-y-2">
            <div className="text-xs uppercase tracking-wide text-muted-foreground">LLM result</div>
            <Textarea value={stepResult} onChange={(e) => setStepResult(e.target.value)} rows={10} />
          </div>
        )}
      </Card>

      <Card className="p-5">
        <div className="text-sm font-medium mb-3">Previous outputs</div>
        <Accordion type="single" collapsible value={activeAccordion} onValueChange={setActiveAccordion}>
          {stepResults.map((step, index) => (
            <AccordionItem key={step.id} value={`step-${index}`}>
              <AccordionTrigger>
                <span>{step.name}</span>
              </AccordionTrigger>
              <AccordionContent>
                <div className="space-y-3">
                  <div>
                    <div className="text-xs uppercase tracking-wide text-muted-foreground">Prompt</div>
                    <pre className="whitespace-pre-wrap text-sm">{step.prompt}</pre>
                  </div>
                  <div>
                    <div className="text-xs uppercase tracking-wide text-muted-foreground">Result</div>
                    <Textarea value={step.result} onChange={(e) => {
                      setStepResults((prev) =>
                        prev.map((item, itemIndex) =>
                          itemIndex === index ? { ...item, result: e.target.value } : item,
                        ),
                      );
                    }} rows={8} />
                  </div>
                  {step.output && (
                    <div className="text-xs text-muted-foreground">
                      Saved as {step.output.save_as} in {step.output.formats.join(", ")}
                    </div>
                  )}
                </div>
              </AccordionContent>
            </AccordionItem>
          ))}
        </Accordion>
      </Card>
    </div>
  );
}
