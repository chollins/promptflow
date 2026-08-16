import axios from "axios";

import type {
  FlowExecuteResponse,
  FlowSummary,
  PromptFlow,
} from "../types/promptForm";
import { BASE_URL } from "./forms";

const api = axios.create({
  baseURL: BASE_URL,
  headers: { "Content-Type": "application/json" },
});

export async function getFlows(): Promise<FlowSummary[]> {
  const { data } = await api.get<FlowSummary[]>("/flows");
  return data;
}

export async function getFlow(id: string): Promise<PromptFlow> {
  const { data } = await api.get<PromptFlow>(`/flows/${id}`);
  return data;
}

export async function executeFlow(
  id: string,
  body: {
    step_id?: string | null;
    values: Record<string, string>;
    context: Record<string, unknown>;
  },
): Promise<FlowExecuteResponse> {
  const { data } = await api.post<FlowExecuteResponse>(`/flows/${id}/execute`, body);
  return data;
}
