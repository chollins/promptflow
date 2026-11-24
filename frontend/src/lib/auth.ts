import { apiGet, apiPost } from "./api";

export interface AuthUser {
  id: string;
  name: string;
  email: string;
  role: string | null;
  organization_id: string;
  diagnostic_capabilities?: string[];
  can_create_flows?: boolean;
  can_share_flows?: boolean;
  organization_settings?: {
    auto_next_step?: boolean;
    allow_user_create_flows?: boolean;
    allow_user_share_flows?: boolean;
    [key: string]: any;
  };
}

export const authService = {
  login(email: string, password: string, rememberMe: boolean) {
    return apiPost<AuthUser>("/auth/login", { email, password, remember_me: rememberMe });
  },

  logout() {
    return apiPost<{ ok: boolean }>("/auth/logout", {});
  },

  getMe() {
    return apiGet<AuthUser>("/auth/me");
  },

  clearSession() {
    // Cookie is HTTP-only; the next authenticated request determines session state.
  },
};
