import { apiGet, apiPost } from "./api";
import { setSessionToken } from "./api";

export interface AuthUser {
  id: string;
  name: string;
  email: string;
  role: string | null;
  organization_id: string;
  session_token: string | null;
}

export const authService = {
  login(email: string, password: string) {
    return apiPost<AuthUser>("/auth/login", { email, password }).then((user) => {
      setSessionToken(user.session_token);
      return user;
    });
  },

  logout() {
    return apiPost<{ ok: boolean }>("/auth/logout", {}).then((result) => {
      setSessionToken(null);
      return result;
    });
  },

  getMe() {
    return apiGet<AuthUser>("/auth/me").then((user) => {
      if (user.session_token) setSessionToken(user.session_token);
      return user;
    });
  },

  clearSession() {
    setSessionToken(null);
  },
};
