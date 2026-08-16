import { useEffect, useMemo, useState } from "react";

import { getErrorMessage, getForm } from "../api/forms";
import { executeFlow } from "../api/flows";
import type { FlowExecuteResponse, FlowStep, PromptFlow, PromptForm } from "../types/promptForm";
import DynamicForm from "./DynamicForm";
import LoadingSpinner from "./LoadingSpinner";
import ProgressBar from "./ProgressBar";
import StepReview from "./StepReview";

type Phase = "form" | "review" | "complete";

interface CompletedStep {
  step: FlowStep;
  response: FlowExecuteResponse;
}

interface Props {
  flow: PromptFlow;
}

function sortedSteps(flow: PromptFlow): FlowStep[] {
  return [...flow.steps].sort((a, b) => a.sequence - b.sequence);
}

export default function FlowRunner({ flow }: Props) {
  const steps = useMemo(() => sortedSteps(flow), [flow]);
  const [phase, setPhase] = useState<Phase>("form");
  const [currentStepIndex, setCurrentStepIndex] = useState(0);
  const [currentForm, setCurrentForm] = useState<PromptForm | null>(null);
  const [values, setValues] = useState<Record<string, string>>({});
  const [context, setContext] = useState<Record<string, unknown>>({});
  const [completedSteps, setCompletedSteps] = useState<CompletedStep[]>([]);
  const [nextStepId, setNextStepId] = useState<string | null>(null);
  const [editedResult, setEditedResult] = useState("");
  const [loading, setLoading] = useState(true);
  const [executing, setExecuting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const currentStep = steps[currentStepIndex] ?? null;
  const activeCompletedStep = completedSteps[completedSteps.length - 1] ?? null;
  const totalSteps = steps.length;
  const progressStep =
    phase === "review"
      ? Math.max(1, completedSteps.length)
      : Math.min(totalSteps, completedSteps.length + 1);

  function getStepResult(stepIndex: number) {
    const stepState = completedSteps[stepIndex];
    if (!stepState) {
      return null;
    }

    const output = stepState.step.output?.save_as ?? null;
    const isActiveReview = phase === "review" && stepIndex === completedSteps.length - 1;
    const result =
      isActiveReview ? editedResult : stepState.response.steps[0]?.result ?? "";

    return {
      name: stepState.step.name,
      sequence: stepState.step.sequence,
      output,
      prompt: stepState.response.steps[0]?.prompt ?? "",
      result,
      next: stepState.response.steps[0]?.next ?? null,
    };
  }

  useEffect(() => {
    if (!currentStep) {
      setCurrentForm(null);
      setLoading(false);
      return;
    }

    async function loadForm() {
      try {
        setLoading(true);
        setError(null);
        const data = await getForm(currentStep.prompt_form_id);
        setCurrentForm(data);
      } catch (err) {
        setError(getErrorMessage(err));
      } finally {
        setLoading(false);
      }
    }

    loadForm();
  }, [currentStep]);

  useEffect(() => {
    if (phase === "review" && activeCompletedStep) {
      setEditedResult(activeCompletedStep.response.steps[0]?.result ?? "");
    }
  }, [phase, activeCompletedStep]);

  async function executeStep(step: FlowStep, stepValues: Record<string, string>, stepContext: Record<string, unknown>) {
    setExecuting(true);
    setError(null);

    try {
      const response = await executeFlow(flow.id, {
        step_id: step.id,
        values: stepValues,
        context: stepContext,
      });

      const executed = response.steps[0];
      setContext({ ...response.context });
      setNextStepId(executed?.next ?? null);
      setCompletedSteps((prev) => [...prev, { step, response }]);
      setEditedResult(executed?.result ?? "");
      setPhase("review");
    } catch (err) {
      setError(getErrorMessage(err));
    } finally {
      setExecuting(false);
    }
  }

  async function handleSubmit(stepValues: Record<string, string>) {
    if (!currentStep) {
      return;
    }

    setValues(stepValues);
    await executeStep(currentStep, stepValues, context);
  }

  async function handleContinue() {
    if (!activeCompletedStep) {
      return;
    }

    const saveAs = activeCompletedStep.step.output?.save_as;
    const updatedContext = {
      ...context,
      ...(saveAs ? { [saveAs]: editedResult } : {}),
    };

    setContext(updatedContext);
    setCompletedSteps((prev) => {
      const updated = [...prev];
      const last = updated[updated.length - 1];
      if (!last) {
        return prev;
      }

      updated[updated.length - 1] = {
        ...last,
        response: {
          ...last.response,
          context: updatedContext,
          steps: last.response.steps.map((step) => ({
            ...step,
            result: editedResult,
          })),
        },
      };
      return updated;
    });

    if (!nextStepId) {
      setPhase("complete");
      return;
    }

    const nextIndex = steps.findIndex((step) => step.id === nextStepId);
    if (nextIndex < 0) {
      setPhase("complete");
      return;
    }

    const nextStep = steps[nextIndex];
    setCurrentStepIndex(nextIndex);
    setPhase("form");

    try {
      setLoading(true);
      setError(null);
      const nextForm = await getForm(nextStep.prompt_form_id);
      setCurrentForm(nextForm);
      await executeStep(nextStep, values, updatedContext);
    } catch (err) {
      setError(getErrorMessage(err));
    } finally {
      setLoading(false);
    }
  }

  function handleCancel() {
    setPhase("form");
    setCompletedSteps([]);
    setContext({});
    setNextStepId(null);
    setEditedResult("");
    setCurrentStepIndex(0);
  }

  if (loading && !currentForm) {
    return <LoadingSpinner />;
  }

  if (!currentStep || !currentForm) {
    return (
      <div className="rounded-xl border border-gray-200 bg-white p-6 text-gray-600">
        No steps found.
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="rounded-xl border border-gray-200 bg-white p-6 shadow-sm space-y-4">
        <div>
          <h1 className="text-3xl font-semibold text-gray-900">{flow.name}</h1>
          <p className="mt-2 text-gray-600">{flow.description}</p>
        </div>
        <ProgressBar current={progressStep} total={totalSteps} />
        <div className="text-sm text-gray-600">
          Current step: <span className="font-medium text-gray-900">{currentStep.name}</span>
        </div>
      </div>

      {error && (
        <div className="rounded-xl border border-red-200 bg-red-50 p-4 text-red-700">
          {error}
        </div>
      )}

      {phase === "form" && (
        <div className="rounded-xl border border-gray-200 bg-white p-6 shadow-sm">
          <DynamicForm
            form={currentForm}
            loading={executing}
            onSubmit={handleSubmit}
            values={values}
            onValuesChange={setValues}
            submitLabel={completedSteps.length === 0 ? "Start" : "Run Step"}
          />
        </div>
      )}

      {phase === "review" && activeCompletedStep && (
        <StepReview
          prompt={activeCompletedStep.response.steps[0]?.prompt ?? ""}
          result={activeCompletedStep.response.steps[0]?.result ?? ""}
          editedResult={editedResult}
          onEditedResultChange={setEditedResult}
          onContinue={handleContinue}
          onCancel={handleCancel}
          saving={executing}
          saveAs={activeCompletedStep.step.output?.save_as}
        />
      )}

      {phase === "complete" && (
        <div className="rounded-xl border border-gray-200 bg-white p-6 shadow-sm">
          <h2 className="text-xl font-semibold text-gray-900">Flow complete</h2>
          <p className="mt-2 text-gray-600">All steps have been executed.</p>
        </div>
      )}

      {completedSteps.length > 0 && (
        <div className="space-y-4">
          <h2 className="text-lg font-semibold text-gray-900">Generated Outputs</h2>
          <div className="space-y-3">
            {completedSteps.map((_, index) => {
              const item = getStepResult(index);
              if (!item) {
                return null;
              }

              return (
                <details
                  key={`${item.name}-${item.sequence}`}
                  className="group rounded-xl border border-gray-200 bg-white shadow-sm"
                >
                  <summary className="flex cursor-pointer list-none items-center justify-between gap-3 p-4">
                    <div className="min-w-0">
                      <div className="flex flex-wrap items-center gap-2">
                        <h3 className="truncate text-sm font-semibold text-gray-900">
                          Step {item.sequence}: {item.name}
                        </h3>
                        {item.output && (
                          <span className="rounded-full bg-gray-100 px-2.5 py-1 text-xs font-medium text-gray-600">
                            {item.output}
                          </span>
                        )}
                      </div>
                      <p className="mt-1 text-xs text-gray-500">
                        {item.next ? `Next: ${item.next}` : "Final step"}
                      </p>
                    </div>
                    <span className="rounded-full bg-gray-100 px-3 py-1 text-xs font-medium text-gray-600 group-open:bg-gray-900 group-open:text-white">
                      View
                    </span>
                  </summary>

                  <div className="border-t border-gray-200 p-4 space-y-3">
                    <div>
                      <h4 className="text-xs font-semibold uppercase tracking-wide text-gray-500">
                        Prompt
                      </h4>
                      <pre className="mt-2 whitespace-pre-wrap rounded-lg bg-gray-50 p-3 text-sm text-gray-700">
                        {item.prompt}
                      </pre>
                    </div>

                    <div>
                      <h4 className="text-xs font-semibold uppercase tracking-wide text-gray-500">
                        Result
                      </h4>
                      <div className="mt-2 whitespace-pre-wrap rounded-lg bg-gray-50 p-3 text-sm text-gray-700">
                        {item.result}
                      </div>
                    </div>
                  </div>
                </details>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}
