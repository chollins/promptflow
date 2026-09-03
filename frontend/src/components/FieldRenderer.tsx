import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Textarea } from "@/components/ui/textarea";
import { Input } from "@/components/ui-kit";

export type Field = {
  id: string;
  label: string;
  description?: string | null;
  type: "text" | "textarea" | "date" | "checkbox" | "radio" | "dropdown" | "hidden";
  required?: boolean;
  default?: string | null;
  options?: Array<string | { label?: string; value?: string; theme?: string; description?: string }>;
};

interface Props {
  field: Field;
  value: string;
  onChange: (value: string) => void;
}

const inputClass =
  "border border-border rounded-md px-3 py-2 w-full text-foreground bg-background";

function normalizeOption(option: NonNullable<Field["options"]>[number]) {
  if (typeof option === "string") {
    return { label: option, value: option, description: undefined as string | undefined };
  }
  const label = option.label ?? option.theme ?? option.value ?? "";
  const value = option.value ?? option.theme ?? label;
  return { label, value, description: option.description };
}

export default function FieldRenderer({ field, value, onChange }: Props) {
  switch (field.type) {
    case "textarea":
      return (
        <Textarea
          id={field.id}
          className={inputClass}
          value={value}
          required={field.required}
          placeholder={field.label}
          rows={4}
          onChange={(e) => onChange(e.target.value)}
        />
      );

    case "date":
      return (
        <Input
          id={field.id}
          type="date"
          className={inputClass}
          value={value}
          required={field.required}
          onChange={(e) => onChange(e.target.value)}
        />
      );

    case "dropdown":
      return (
        <Select value={value} onValueChange={onChange}>
          <SelectTrigger id={field.id} className={inputClass}>
            <SelectValue placeholder={`Select ${field.label}`} />
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

    case "checkbox":
      if (field.options && field.options.length > 0) {
        const selected = value ? value.split(",").filter(Boolean) : [];
        return (
          <div className="space-y-2">
            {field.options.map((option) => {
              const normalized = normalizeOption(option);
              const isChecked = selected.includes(normalized.value);
              return (
                <label key={normalized.value} className="flex items-start gap-2 text-sm text-foreground">
                  <input
                    id={field.id}
                    type="checkbox"
                    className="mt-1 h-4 w-4 rounded border-border"
                    checked={isChecked}
                    required={field.required && selected.length === 0}
                    onChange={(e) => {
                      const next = e.target.checked
                        ? [...selected, normalized.value]
                        : selected.filter((v) => v !== normalized.value);
                      onChange(next.join(","));
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
        );
      }

      return (
        <label className="flex items-start gap-2 text-sm text-foreground">
          <input
            id={field.id}
            type="checkbox"
            className="mt-0.5 h-4 w-4 rounded border-border"
            checked={value === "true"}
            required={field.required}
            onChange={(e) => onChange(e.target.checked ? "true" : "")}
          />
          <span>{field.label}</span>
        </label>
      );

    case "radio":
      return (
        <fieldset className="space-y-2">
          <legend className="sr-only">{field.label}</legend>
          {field.options?.map((option) => {
            const normalized = normalizeOption(option);
            return (
            <label key={normalized.value} className="flex items-start gap-2 text-sm text-foreground">
              <input
                type="radio"
                name={field.id}
                value={normalized.value}
                className="mt-0.5 h-4 w-4 rounded-full border-border"
                checked={value === normalized.value}
                required={field.required}
                onChange={(e) => onChange(e.target.value)}
              />
              <span>{normalized.label}</span>
            </label>
            );
          })}
        </fieldset>
      );

    case "hidden":
      return <input id={field.id} type="hidden" value={value} onChange={(e) => onChange(e.target.value)} />;

    case "text":
    default:
      return (
        <Input
          id={field.id}
          className={inputClass}
          value={value}
          required={field.required}
          placeholder={field.label}
          onChange={(e) => onChange(e.target.value)}
        />
      );
  }
}
