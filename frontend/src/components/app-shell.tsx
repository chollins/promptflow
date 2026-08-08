import { Link, useNavigate, useRouterState } from "@tanstack/react-router";
import {
  LayoutDashboard,
  Workflow,
  Users,
  Building2,
  User,
  ListTree,
  ShieldCheck,
  Form,
  Sparkles,
  LogOut,
} from "lucide-react";
import { useEffect, useMemo, useState, type ReactNode } from "react";
import { authService } from "@/lib/auth";
import { Button } from "@/components/ui-kit";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";

const ALL_ITEMS: {
  title: string;
  url: string;
  icon: typeof LayoutDashboard;
  roles: Array<"admin" | "user" | "member" | "superadmin">;
}[] = [
  {
    title: "Dashboard",
    url: "/dashboard",
    icon: LayoutDashboard,
    roles: ["admin", "user", "superadmin"],
  },
  { title: "Flows", url: "/flows", icon: Workflow, roles: ["admin", "member", "superadmin"] },
  { title: "Forms", url: "/forms", icon: Form, roles: ["admin", "member", "superadmin"] },
  { title: "Users", url: "/users", icon: Users, roles: ["admin", "superadmin"] },
  { title: "Organization", url: "/organization", icon: Building2, roles: ["admin"] },
  { title: "Organizations", url: "/admin/organizations", icon: Building2, roles: ["superadmin"] },
  { title: "Forms Catalog", url: "/admin/forms", icon: Form, roles: ["superadmin"] },
  { title: "Flow Composer", url: "/admin/flows", icon: ListTree, roles: ["superadmin"] },

  { 
    title: "Flow Access",
    url: "/admin/manage-flows",
    icon: ShieldCheck,
    roles: ["superadmin"],
  },
  { title: "Profile", url: "/profile", icon: User, roles: ["admin", "user", "member", "superadmin"] },
 
];

export function AppShell({ children }: { children: ReactNode }) {
  const pathname = useRouterState({ select: (s) => s.location.pathname });
  const navigate = useNavigate();
  const [role, setRole] = useState<"admin" | "user" | "superadmin" | null>(null);
  const [name, setName] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [logoutOpen, setLogoutOpen] = useState(false);

  const items = useMemo(
    () => ALL_ITEMS.filter((item) => !role || item.roles.includes(role)),
    [role],
  );

  useEffect(() => {
    let active = true;
    authService
      .getMe()
      .then((user) => {
        if (!active) return;
        setRole((user.role as "admin" | "user" | "superadmin" | null) ?? null);
        setName(user.name || null);
      })
      .catch(() => {
        if (!active) return;
        setRole(null);
        setName(null);
        navigate({ to: "/login" });
      })
      .finally(() => {
        if (active) setLoading(false);
      });

    return () => {
      active = false;
    };
  }, [navigate]);

  useEffect(() => {
    if (!role) return;
    const matched = ALL_ITEMS.find(
      (item) => pathname === item.url || pathname.startsWith(item.url + "/"),
    );
    if (matched && !matched.roles.includes(role)) {
      navigate({ to: "/dashboard" });
    }
  }, [navigate, pathname, role]);

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-background text-foreground">
        <div className="text-sm text-muted-foreground">Loading workspace...</div>
      </div>
    );
  }

  return (
    <div className="min-h-screen flex w-full bg-background text-foreground">
      <aside className="hidden md:flex w-64 shrink-0 flex-col border-r border-border bg-background">
        <div className="h-16 flex items-center justify-between px-6 border-b border-border">
          <Link to="/" className="flex items-center gap-2 font-semibold tracking-tight">
            <Sparkles className="h-4 w-4" strokeWidth={2.25} />
            <span>PromptFlow</span>
          </Link>
          <div className="flex items-center gap-2">
            {/* {role && (
              <span className="rounded-full border border-border bg-muted px-2.5 py-1 text-[11px] font-medium text-foreground">
                {name ? `${name} · ` : ""}
                {role}
              </span>
            )} */}
            <button
              onClick={() => setLogoutOpen(true)}
              className="text-muted-foreground hover:text-foreground"
              aria-label="Logout"
            >
              <LogOut className="h-4 w-4" />
            </button>
          </div>
        </div>
        <nav className="flex-1 px-3 py-6 space-y-0.5">
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
      </aside>

      {/* Mobile top nav */}
      <div className="md:hidden fixed top-0 inset-x-0 h-14 border-b border-border bg-background z-30 flex items-center px-4 gap-3 overflow-x-auto">
        <Link to="/" className="flex items-center gap-2 font-semibold shrink-0">
          <Sparkles className="h-4 w-4" />
          PromptFlowqqq
        </Link>
        {role && (
          <span className="rounded-full border border-border bg-muted px-2 py-1 text-[10px] font-medium text-foreground">
            {name ? `${name} · ` : ""}
            {role}
          </span>
        )}
        <button
          onClick={() => setLogoutOpen(true)}
          className="ml-auto text-muted-foreground hover:text-foreground"
          aria-label="Logout"
        >
          <LogOut className="h-4 w-4" />
        </button>
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

      <Dialog open={logoutOpen} onOpenChange={setLogoutOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Log out?</DialogTitle>
            <DialogDescription>
              You’ll need to sign in again to keep using PromptFlow.
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button variant="secondary" onClick={() => setLogoutOpen(false)}>
              Cancel
            </Button>
            <Button
              onClick={() =>
                authService.logout().then(() => {
                  setLogoutOpen(false);
                  navigate({ to: "/login" });
                })
              }
            >
              Log out
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
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
