import { Link, useRouterState } from "@tanstack/react-router";
import { LayoutDashboard, Workflow, Users, Building2, User, LogOut, Sparkles, Shield, ListTree, ShieldCheck } from "lucide-react";
import type { ReactNode } from "react";
import { useMockUser } from "@/hooks/use-mock-user";
import { setMockRole, type Role } from "@/lib/mock-store";

const ALL_ITEMS: {
  title: string;
  url: string;
  icon: typeof LayoutDashboard;
  roles: Role[];
}[] = [
  { title: "Dashboard", url: "/dashboard", icon: LayoutDashboard, roles: ["admin", "user"] },
  { title: "Flows", url: "/flows", icon: Workflow, roles: ["admin", "user"] },
  { title: "Users", url: "/users", icon: Users, roles: ["admin"] },
  { title: "Organization", url: "/organization", icon: Building2, roles: ["admin"] },
  { title: "Profile", url: "/profile", icon: User, roles: ["admin", "user", "superadmin"] },
  { title: "Organizations", url: "/admin/organizations", icon: Building2, roles: ["superadmin"] },
  { title: "Flows Catalog", url: "/admin/flows", icon: ListTree, roles: ["superadmin"] },
  { title: "Manage Flow Access", url: "/admin/manage-flows", icon: ShieldCheck, roles: ["superadmin"] },
];

const ROLES: Role[] = ["admin", "user", "superadmin"];
const ROLE_LABEL: Record<Role, string> = {
  admin: "Admin",
  user: "User",
  superadmin: "Super",
};

export function AppShell({ children }: { children: ReactNode }) {
  const user = useMockUser();
  const pathname = useRouterState({ select: (s) => s.location.pathname });
  const items = ALL_ITEMS.filter((i) => i.roles.includes(user.role));

  return (
    <div className="min-h-screen flex w-full bg-background text-foreground">
      <aside className="hidden md:flex w-64 shrink-0 flex-col border-r border-border bg-background">
        <div className="h-16 flex items-center px-6 border-b border-border">
          <Link to="/" className="flex items-center gap-2 font-semibold tracking-tight">
            <Sparkles className="h-4 w-4" strokeWidth={2.25} />
            <span>PromptFlow</span>
          </Link>
        </div>
        <nav className="flex-1 px-3 py-6 space-y-0.5">
          {user.role === "superadmin" && (
            <div className="px-3 pb-2 flex items-center gap-1.5 text-[10px] uppercase tracking-wider text-muted-foreground">
              <Shield className="h-3 w-3" />
              Superadmin
            </div>
          )}
          {items.map((item) => {
            const active = pathname === item.url || pathname.startsWith(item.url + "/");
            return (
              <Link
                key={item.url}
                to={item.url}
                className={
                  "flex items-center gap-2.5 rounded-md px-3 py-2 text-sm transition-colors " +
                  (active
                    ? "bg-muted text-foreground font-medium"
                    : "text-muted-foreground hover:bg-muted/60 hover:text-foreground")
                }
              >
                <item.icon className="h-4 w-4" />
                {item.title}
              </Link>
            );
          })}
        </nav>
        <div className="border-t border-border p-4 space-y-3">
          <div className="text-xs text-muted-foreground">Preview role</div>
          <div className="flex rounded-md border border-border p-0.5">
            {ROLES.map((r) => (
              <button
                key={r}
                onClick={() => setMockRole(r)}
                className={
                  "flex-1 text-xs py-1.5 rounded-sm transition-colors " +
                  (user.role === r ? "bg-foreground text-background" : "text-muted-foreground hover:text-foreground")
                }
              >
                {ROLE_LABEL[r]}
              </button>
            ))}
          </div>
          <Link
            to="/"
            className="flex items-center gap-2 text-xs text-muted-foreground hover:text-foreground"
          >
            <LogOut className="h-3.5 w-3.5" />
            Sign out
          </Link>
        </div>
      </aside>

      {/* Mobile top nav */}
      <div className="md:hidden fixed top-0 inset-x-0 h-14 border-b border-border bg-background z-30 flex items-center px-4 gap-3 overflow-x-auto">
        <Link to="/" className="flex items-center gap-2 font-semibold shrink-0">
          <Sparkles className="h-4 w-4" />
          PromptFlow
        </Link>
        <div className="flex gap-1 ml-2">
          {items.map((item) => {
            const active = pathname === item.url;
            return (
              <Link
                key={item.url}
                to={item.url}
                className={
                  "text-xs px-2.5 py-1.5 rounded-md shrink-0 " +
                  (active ? "bg-muted text-foreground font-medium" : "text-muted-foreground")
                }
              >
                {item.title}
              </Link>
            );
          })}
        </div>
      </div>

      <main className="flex-1 min-w-0 pt-14 md:pt-0">
        <div className="max-w-5xl mx-auto px-6 md:px-10 py-10 md:py-14">{children}</div>
      </main>
    </div>
  );
}

export function PageHeader({
  title,
  description,
  actions,
}: {
  title: string;
  description?: string;
  actions?: ReactNode;
}) {
  return (
    <div className="flex items-start justify-between gap-4 mb-10">
      <div>
        <h1 className="text-2xl md:text-3xl font-semibold tracking-tight">{title}</h1>
        {description && <p className="text-muted-foreground mt-1.5 text-sm">{description}</p>}
      </div>
      {actions && <div className="flex gap-2 shrink-0">{actions}</div>}
    </div>
  );
}
