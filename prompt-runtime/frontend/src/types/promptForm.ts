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

export interface FlowRuntimeSettings {
  mode: "guided" | "automatic";
  default_review_required: boolean;
}

export interface FlowReviewSettings {
  required: boolean;
  editable: boolean;
}

export interface FlowOutputSettings {
  save_as: string;
  formats: string[];
}

export interface FlowStep {
  id: string;
  sequence: number;
  name: string;
  prompt_form_id: string;
  input_bindings: Record<string, string>;
  dynamic_fields: string[];
  review: FlowReviewSettings;
  output?: FlowOutputSettings | null;
  next?: string | null;
}

export interface PromptFlow {
  id: string;
  version: string;
  name: string;
  description: string;
  runtime: FlowRuntimeSettings;
  steps: FlowStep[];
}

export interface FlowSummary {
  id: string;
  name: string;
  description: string;
  version: string;
}

export interface FlowStepResult {
  id: string;
  sequence: number;
  name: string;
  prompt: string;
  result: string;
  completed: boolean;
  next: string | null;
  output?: FlowOutputSettings | null;
}

export interface FlowExecuteResponse {
  context: Record<string, unknown>;
  steps: FlowStepResult[];
}
