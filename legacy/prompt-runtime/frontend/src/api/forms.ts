import axios from "axios";

import type {
  ExecuteResponse,
  FormSummary,
  PromptForm,
} from "../types/promptForm";

export const BASE_URL = "http://localhost:8000";

const api = axios.create({
  baseURL: BASE_URL,
  headers: { "Content-Type": "application/json" },
});

export async function getForms(): Promise<FormSummary[]> {
  const { data } = await api.get<FormSummary[]>("/forms");
  return data;
}

export async function getForm(id: string): Promise<PromptForm> {
  const { data } = await api.get<PromptForm>(`/forms/${id}`);
  return data;
}

export async function executeForm(
  id: string,
  values: Record<string, string>,
): Promise<ExecuteResponse> {
  const { data } = await api.post<ExecuteResponse>(`/execute/${id}`, {
    values,
  });
  return data;
}

export function getErrorMessage(error: unknown): string {
  if (axios.isAxiosError(error)) {
    const detail = error.response?.data?.detail;
    if (typeof detail === "string") {
      return detail;
    }
    return error.message;
  }
  if (error instanceof Error) {
    return error.message;
  }
  return "An unexpected error occurred";
}
