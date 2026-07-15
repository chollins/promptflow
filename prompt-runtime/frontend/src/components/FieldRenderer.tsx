import type { Field } from "../types/promptForm";

interface Props {
  field: Field;
  value: string;
  onChange: (value: string) => void;
}

const inputClass =
  "border border-gray-300 rounded px-3 py-2 w-full text-gray-900";

export default function FieldRenderer({ field, value, onChange }: Props) {
  switch (field.type) {
    case "textarea":
      return (
        <textarea
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
        <select
          id={field.id}
          className={inputClass}
          value={value}
          required={field.required}
          onChange={(e) => onChange(e.target.value)}
        >
          <option value="">Select {field.label}</option>
          {field.options?.map((option) => (
            <option key={option} value={option}>
              {option}
            </option>
          ))}
        </select>
      );

    case "checkbox":
      return (
        <label className="flex items-center gap-2 text-gray-700">
          <input
            id={field.id}
            type="checkbox"
            className="h-4 w-4 rounded border-gray-300"
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
            <label
              key={option}
              className="flex items-center gap-2 text-gray-700"
            >
              <input
                type="radio"
                name={field.id}
                value={option}
                className="h-4 w-4 border-gray-300"
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
      return (
        <input
          id={field.id}
          type="hidden"
          value={value}
          onChange={(e) => onChange(e.target.value)}
        />
      );

    case "text":
    default:
      return (
        <input
          id={field.id}
          type="text"
          className={inputClass}
          value={value}
          required={field.required}
          placeholder={field.label}
          onChange={(e) => onChange(e.target.value)}
        />
      );
  }
}
