import { createFileRoute, Outlet, redirect, isRedirect } from "@tanstack/react-router";
import { apiGet } from "@/lib/api";

export const Route = createFileRoute("/admin")({
  beforeLoad: async () => {
    try {
      const user = await apiGet<{ role: string | null }>("/auth/me");
      if (user.role !== "superadmin") {
        throw redirect({ to: "/dashboard" });
      }
    } catch (e) {
      if (isRedirect(e)) {
         throw e;
      }
      throw redirect({ to: "/login" });
    }
  },
  component: () => <Outlet />,
});

