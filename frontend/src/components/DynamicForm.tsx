import { useEffect, useState } from "react";
import FieldRenderer, { type Field } from "@/components/FieldRenderer";

export type PromptForm = {
  id: string;
  name: string;
  description?: string | null;
  version: string;
  fields: Field[];
  prompt?: { system: string; user: string };
  model?: { provider?: string; name: string; temperature?: number };
};

interface Props {
  form: PromptForm;
  values?: Record<string, string>;
  onValuesChange?: (values: Record<string, string>) => void;
}

function buildInitialValues(fields: Field[]): Record<string, string> {
  return fields.reduce<Record<string, string>>((acc, field) => {
    acc[field.id] = field.type === "checkbox" ? "false" : field.default ?? "";
    return acc;
  }, {});
}

export default function DynamicForm({
  form,
  values: controlledValues,
  onValuesChange,
}: Props) {
  const [internalValues, setInternalValues] = useState<Record<string, string>>(() =>
    buildInitialValues(form.fields),
  );

  useEffect(() => {
    if (controlledValues) {
      setInternalValues((prev) => ({ ...buildInitialValues(form.fields), ...prev, ...controlledValues }));
    }
  }, [controlledValues, form.fields]);

  const values = controlledValues ?? internalValues;

  function updateField(id: string, value: string) {
    const next = { ...values, [id]: value };
    if (controlledValues) {
      onValuesChange?.(next);
      return;
    }
    setInternalValues(next);
  }



  return (
    <div className="space-y-4">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight text-foreground">{form.name}</h1>
        {form.description && <p className="mt-1 text-sm text-muted-foreground">{form.description}</p>}
      </div>

      {form.fields.map((field) => (
        <div key={field.id} className="space-y-1.5">
          {field.type !== "checkbox" && field.type !== "hidden" && (
            <label htmlFor={field.id} className="block text-sm font-medium text-foreground/80">
              {field.label}
              {field.required && <span className="text-red-500"> *</span>}
            </label>
          )}
          {field.description && field.type !== "checkbox" && (
            <p className="text-xs text-muted-foreground">{field.description}</p>
          )}
          <FieldRenderer
            field={field}
            value={values[field.id] ?? ""}
            onChange={(value) => updateField(field.id, value)}
          />
        </div>
      ))}


    </div>
  );
}
