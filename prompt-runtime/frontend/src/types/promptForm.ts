export type FieldType =
  | "text"
  | "textarea"
  | "checkbox"
  | "radio"
  | "dropdown"
  | "hidden";

export interface ModelSettings {
  provider: string;
  name: string;
  temperature: number;
}

export interface Prompt {
  system: string;
  user: string;
}

export interface Field {
  id: string;
  label: string;
  description?: string;
  type: FieldType;
  required?: boolean;
  default?: string;
  options?: string[];
}

export interface PromptForm {
  id: string;
  name: string;
  description?: string;
  version: string;
  fields: Field[];
  prompt: Prompt;
  model: ModelSettings;
}

export interface FormSummary {
  id: string;
  name: string;
  description?: string;
  version: string;
}

export interface ExecuteResponse {
  prompt: string;
  result: string;
}
