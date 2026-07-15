import { useState, type FormEvent } from "react";

import type { Field, PromptForm } from "../types/promptForm";
import FieldRenderer from "./FieldRenderer";

interface Props {
  form: PromptForm;
  loading?: boolean;
  onSubmit: (values: Record<string, string>) => void;
}

function buildInitialValues(fields: Field[]): Record<string, string> {
  return fields.reduce<Record<string, string>>((acc, field) => {
    acc[field.id] = field.default ?? "";
    return acc;
  }, {});
}

export default function DynamicForm({ form, loading = false, onSubmit }: Props) {
  const [values, setValues] = useState<Record<string, string>>(() =>
    buildInitialValues(form.fields),
  );

  function updateField(id: string, value: string) {
    setValues((prev) => ({ ...prev, [id]: value }));
  }

  function handleSubmit(e: FormEvent) {
    e.preventDefault();
    onSubmit(values);
  }

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      <div>
        <h1 className="text-2xl font-semibold text-gray-900">{form.name}</h1>
        {form.description && (
          <p className="mt-1 text-gray-600">{form.description}</p>
        )}
      </div>

      {form.fields.map((field) => (
        <div key={field.id} className="space-y-1">
          {field.type !== "checkbox" && field.type !== "hidden" && (
            <label
              htmlFor={field.id}
              className="block text-sm font-medium text-gray-700"
            >
              {field.label}
              {field.required && <span className="text-red-500"> *</span>}
            </label>
          )}
          {field.description && field.type !== "checkbox" && (
            <p className="text-sm text-gray-500">{field.description}</p>
          )}
          <FieldRenderer
            field={field}
            value={values[field.id] ?? ""}
            onChange={(value) => updateField(field.id, value)}
          />
        </div>
      ))}

      <button
        type="submit"
        disabled={loading}
        className="bg-blue-600 hover:bg-blue-700 disabled:bg-blue-400 text-white px-4 py-2 rounded"
      >
        {loading ? "Generating..." : "Generate"}
      </button>
    </form>
  );
}
