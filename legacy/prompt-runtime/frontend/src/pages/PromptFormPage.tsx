import { useEffect, useState } from "react";
import { useParams } from "react-router-dom";

import { executeForm, getErrorMessage, getForm } from "../api/forms";
import DynamicForm from "../components/DynamicForm";
import LoadingSpinner from "../components/LoadingSpinner";
import ResultViewer from "../components/ResultViewer";
import type { ExecuteResponse, PromptForm } from "../types/promptForm";

export default function PromptFormPage() {
  const { id } = useParams<{ id: string }>();
  const [form, setForm] = useState<PromptForm | null>(null);
  const [result, setResult] = useState<ExecuteResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [executing, setExecuting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const formId = id ?? null;

  useEffect(() => {
    const currentFormId = formId;
    if (!currentFormId) {
      setError("Form not found.");
      setLoading(false);
      return;
    }

    async function loadForm() {
      try {
        setLoading(true);
        setError(null);
        const data = await getForm(currentFormId!);
        setForm(data);
      } catch (err) {
        const message = getErrorMessage(err);
        setError(
          message.includes("not found") ? "Form not found." : message,
        );
      } finally {
        setLoading(false);
      }
    }

    loadForm();
  }, [formId]);

  async function handleSubmit(values: Record<string, string>) {
    if (!formId) {
      return;
    }

    try {
      setExecuting(true);
      setError(null);
      const response = await executeForm(formId, values);
      setResult(response);
    } catch (err) {
      setError(getErrorMessage(err));
    } finally {
      setExecuting(false);
    }
  }

  if (loading) {
    return <LoadingSpinner />;
  }

  if (error && !form) {
    return (
      <div className="bg-red-50 border border-red-200 text-red-700 rounded-lg p-4">
        {error}
      </div>
    );
  }

  if (!form) {
    return null;
  }

  return (
    <div className="space-y-6">
      <div className="bg-white border rounded-lg shadow-sm p-6">
        <DynamicForm
          form={form}
          loading={executing}
          onSubmit={handleSubmit}
        />
      </div>

      {error && (
        <div className="bg-red-50 border border-red-200 text-red-700 rounded-lg p-4">
          {error}
        </div>
      )}

      <ResultViewer result={result} />
    </div>
  );
}
