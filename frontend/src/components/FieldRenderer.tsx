import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Textarea } from "@/components/ui/textarea";
import { Input } from "@/components/ui-kit";

export type Field = {
  id: string;
  label: string;
  description?: string | null;
  type: "text" | "textarea" | "checkbox" | "radio" | "dropdown" | "hidden";
  required?: boolean;
  default?: string | null;
  options?: string[];
};

interface Props {
  field: Field;
  value: string;
  onChange: (value: string) => void;
}

const inputClass =
  "border border-border rounded-md px-3 py-2 w-full text-foreground bg-background";

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

    case "dropdown":
      return (
        <Select value={value} onValueChange={onChange}>
          <SelectTrigger id={field.id} className={inputClass}>
            <SelectValue placeholder={`Select ${field.label}`} />
          </SelectTrigger>
          <SelectContent>
            {field.options?.map((option) => (
              <SelectItem key={option} value={option}>
                {option}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      );

    case "checkbox":
      return (
        <label className="flex items-center gap-2 text-sm text-foreground">
          <input
            id={field.id}
            type="checkbox"
            className="h-4 w-4 rounded border-border"
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
          {field.options?.map((option) => (
            <label key={option} className="flex items-center gap-2 text-sm text-foreground">
              <input
                type="radio"
                name={field.id}
                value={option}
                className="h-4 w-4 border-border"
                checked={value === option}
                required={field.required}
                onChange={(e) => onChange(e.target.value)}
              />
              <span>{option}</span>
            </label>
          ))}
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
