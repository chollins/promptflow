import React, { useEffect } from "react";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Textarea } from "@/components/ui/textarea";
import { Input } from "@/components/ui-kit";

export type Field = {
  id: string;
  label: string;
  description?: string | null;
  placeholder?: string | null;
  type: "text" | "textarea" | "date" | "checkbox" | "radio" | "dropdown" | "multiselect" | "hidden";
  required?: boolean;
  default?: string | null;
  options?: Array<string | { label?: string; value?: string; theme?: string; description?: string }>;
  data_source?: { type: string; step_id?: string; path?: string } | null;
};

interface Props {
  field: Field;
  value: string;
  onChange: (value: string) => void;
  radioGroupPrefix?: string;
  disabled?: boolean;
}

const inputClass =
  "border border-border rounded-md px-3 py-2 w-full text-foreground bg-background";

export function normalizeOption(option: NonNullable<Field["options"]>[number]) {
  if (typeof option === "string") {
    return { label: option, value: option, description: undefined as string | undefined };
  }
  const label = option.label ?? option.theme ?? option.value ?? "";
  const value = option.value ?? option.theme ?? label;
  return { label, value, description: option.description };
}

export default function FieldRenderer({
  field,
  value,
  onChange,
  radioGroupPrefix,
  disabled = false,
}: Props) {
  // Sync default value if value is currently empty or undefined
  useEffect(() => {
    if ((value === undefined || value === "") && field.default) {
      onChange(field.default);
    }
  }, [field.default, value, onChange]);

  const placeholderText = field.placeholder || field.description || field.label;

  switch (field.type) {
    case "textarea":
      return (
        <Textarea
          id={field.id}
          className={inputClass}
          value={value ?? ""}
          required={field.required}
          placeholder={placeholderText}
          rows={4}
          disabled={disabled}
          onChange={(e) => onChange(e.target.value)}
        />
      );

    case "date":
      return (
        <Input
          id={field.id}
          type="date"
          className={inputClass}
          value={value ?? ""}
          required={field.required}
          disabled={disabled}
          onChange={(e) => onChange(e.target.value)}
        />
      );

    case "dropdown":
      return (
        <Select
          value={value || undefined}
          onValueChange={onChange}
          disabled={disabled}
        >
          <SelectTrigger id={field.id} className={inputClass}>
            <SelectValue placeholder={field.placeholder || field.description || `Select ${field.label}`} />
          </SelectTrigger>
          <SelectContent>
            {field.options?.map((option) => {
              const normalized = normalizeOption(option);
              return (
                <SelectItem key={normalized.value} value={normalized.value}>
                  {normalized.label}
                </SelectItem>
              );
            })}
          </SelectContent>
        </Select>
      );

    case "multiselect":
    case "checkbox":
      if (field.options && field.options.length > 0) {
        const selected = value
          ? value.split(",").map((v) => v.trim()).filter(Boolean)
          : [];
        return (
          <div className="space-y-2">
            {field.options.map((option) => {
              const normalized = normalizeOption(option);
              const isChecked = selected.includes(normalized.value);
              const inputId = `${field.id}-${normalized.value}`;
              return (
                <label
                  key={normalized.value}
                  className="flex items-start gap-2 text-sm text-foreground cursor-pointer select-none"
                >
                  <input
                    id={inputId}
                    type="checkbox"
                    className="mt-1 h-4 w-4 rounded border-border cursor-pointer"
                    checked={isChecked}
                    disabled={disabled}
                    required={field.required && selected.length === 0}
                    onChange={(e) => {
                      const allOptions = (field.options || []).map((o) => normalizeOption(o).value);
                      const updated = e.target.checked
                        ? [...selected, normalized.value]
                        : selected.filter((v) => v !== normalized.value);
                      // Preserve canonical option order and join with comma + space
                      const ordered = allOptions.filter((optVal) => updated.includes(optVal));
                      onChange(ordered.join(", "));
                    }}
                  />
                  <span>
                    <span className="block font-medium">{normalized.label}</span>
                    {normalized.description ? (
                      <span className="block text-xs text-muted-foreground">
                        {normalized.description}
                      </span>
                    ) : null}
                  </span>
                </label>
              );
            })}
          </div>
        );
      }

      return (
        <label className="flex items-start gap-2 text-sm text-foreground cursor-pointer select-none">
          <input
            id={field.id}
            type="checkbox"
            className="mt-0.5 h-4 w-4 rounded border-border cursor-pointer"
            checked={value === "true"}
            disabled={disabled}
            required={field.required}
            onChange={(e) => onChange(e.target.checked ? "true" : "false")}
          />
          <span>{field.label}</span>
        </label>
      );

    case "radio": {
      const radioName = radioGroupPrefix ? `${radioGroupPrefix}-${field.id}` : field.id;
      return (
        <fieldset className="space-y-2">
          <legend className="sr-only">{field.label}</legend>
          {field.options?.map((option) => {
            const normalized = normalizeOption(option);
            const isChecked =
              value === normalized.value ||
              (value === "" && field.default === normalized.value);
            const inputId = `${radioName}-${normalized.value}`;
            return (
              <label
                key={normalized.value}
                className="flex items-start gap-2 text-sm text-foreground cursor-pointer select-none"
                onClick={() => {
                  if (!disabled && value !== normalized.value) {
                    onChange(normalized.value);
                  }
                }}
              >
                <input
                  id={inputId}
                  type="radio"
                  name={radioName}
                  value={normalized.value}
                  className="mt-0.5 h-4 w-4 rounded-full border-border cursor-pointer"
                  checked={isChecked}
                  disabled={disabled}
                  required={field.required && !value && !field.default}
                  onChange={(e) => onChange(e.target.value)}
                />
                <div>
                  <span className="block font-medium">{normalized.label}</span>
                  {normalized.description ? (
                    <span className="block text-xs text-muted-foreground">
                      {normalized.description}
                    </span>
                  ) : null}
                </div>
              </label>
            );
          })}
        </fieldset>
      );
    }

    case "hidden":
      return <input id={field.id} type="hidden" value={value ?? ""} readOnly />;

    case "text":
    default:
      return (
        <Input
          id={field.id}
          className={inputClass}
          value={value ?? ""}
          required={field.required}
          placeholder={placeholderText}
          disabled={disabled}
          onChange={(e) => onChange(e.target.value)}
        />
      );
  }
}
