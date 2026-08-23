import { Link, useLocation, useNavigate } from "react-router-dom";
import {
  LayoutDashboard,
  Workflow,
  Users,
  Building2,
  Settings,
  ListTree,
  ShieldCheck,
  Form,
  Sparkles,
} from "lucide-react";
import { useEffect, useMemo, useState, type ReactNode } from "react";
import { authService } from "@/lib/auth";
import { hasSessionToken } from "@/lib/api";
import { clearRecentActivity } from "@/lib/activity";
import { Button } from "@/components/ui-kit";
import {
  Sidebar,
  SidebarContent,
  SidebarFooter,
  SidebarGroup,
  SidebarGroupLabel,
  SidebarHeader,
  SidebarInset,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
  SidebarProvider,
  SidebarTrigger,
  SidebarUserFooter,
} from "@/components/ui/sidebar";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Toaster } from "sonner";

const ALL_ITEMS: {
  title: string;
  url: string;
  icon: typeof LayoutDashboard;
  roles: Array<"admin" | "user" | "member" | "superadmin">;
}[] = [
  { title: "Dashboard", url: "/dashboard", icon: LayoutDashboard, roles: ["admin", "user", "superadmin", "member"] },
  { title: "Flows", url: "/flows", icon: Workflow, roles: ["admin", "member", "superadmin"] },
  { title: "Forms", url: "/forms", icon: Form, roles: ["admin", "member", "superadmin"] },
  { title: "Users", url: "/users", icon: Users, roles: ["admin", "superadmin"] },
  { title: "Organization", url: "/organization", icon: Building2, roles: ["admin"] },
  { title: "Organizations", url: "/admin/organizations", icon: Building2, roles: ["superadmin"] },
  { title: "Forms Catalog", url: "/admin/forms", icon: Form, roles: ["superadmin"] },
  { title: "Flow Composer", url: "/admin/flows", icon: ListTree, roles: ["superadmin"] },
  { title: "Flow Access", url: "/admin/manage-flows", icon: ShieldCheck, roles: ["superadmin"] },
  { title: "Settings", url: "/settings", icon: Settings, roles: ["admin", "user", "member", "superadmin"] },
];

export function AppShell({ children }: { children: ReactNode }) {
  const pathname = useLocation().pathname;
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
        if (hasSessionToken()) return;
        setRole(null);
        setName(null);
        navigate("/login");
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
      navigate("/dashboard");
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
    <SidebarProvider defaultOpen>
      <Toaster position="top-center" richColors />
      <div className="flex min-h-svh w-full bg-background text-foreground">
        <Sidebar collapsible="icon">
          <SidebarHeader className="border-b border-sidebar-border px-4 py-4">
            <Link to="/" className="flex items-center gap-2 font-semibold tracking-tight">
              <Sparkles className="h-4 w-4" strokeWidth={2.25} />
              <span>PromptFlow</span>
            </Link>
          </SidebarHeader>

          <SidebarContent>
            <SidebarGroup>
              <SidebarGroupLabel>Workspace</SidebarGroupLabel>
              <SidebarMenu>
                {items.map((item) => {
                  const active = pathname === item.url || pathname.startsWith(item.url + "/");
                  return (
                    <SidebarMenuItem key={item.url}>
                      <SidebarMenuButton asChild isActive={active}>
                        <Link to={item.url}>
                          <item.icon className="h-4 w-4" />
                          <span>{item.title}</span>
                        </Link>
                      </SidebarMenuButton>
                    </SidebarMenuItem>
                  );
                })}
              </SidebarMenu>
            </SidebarGroup>
          </SidebarContent>

          <SidebarFooter className="mt-auto p-2">
            <SidebarUserFooter
              name={name || "User"}
              role={role}
              onClick={() => setLogoutOpen(true)}
            />
          </SidebarFooter>
        </Sidebar>

        <SidebarInset>
          <div className="sticky top-0 z-20 flex h-14 items-center gap-3 border-b border-border bg-background/95 px-4 backdrop-blur md:hidden">
            <SidebarTrigger />
            <Link to="/" className="flex items-center gap-2 font-semibold shrink-0">
              <Sparkles className="h-4 w-4" />
              PromptFlow
            </Link>
          </div>
          <main className=" px-6 py-8 md:px-10 md:py-10">
            {children}
          </main>
        </SidebarInset>
      </div>

      <Dialog open={logoutOpen} onOpenChange={setLogoutOpen}>
        <DialogContent className="max-w-sm">
          <DialogHeader>
            <DialogTitle>Log out?</DialogTitle>
            <DialogDescription>You'll need to sign in again to keep using PromptFlow.</DialogDescription>
          </DialogHeader>
          <DialogFooter className="mt-4">
            <Button variant="secondary" onClick={() => setLogoutOpen(false)}>
              Cancel
            </Button>
            <Button
              onClick={() =>
                void authService.logout().then(() => {
                  clearRecentActivity();
                  setLogoutOpen(false);
                  navigate("/login");
                })
              }
            >
              Log out
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </SidebarProvider>
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
    <div className="mb-6 flex items-start justify-between gap-4">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight md:text-3xl">{title}</h1>
        {description && <p className="mt-1.5 text-sm text-muted-foreground">{description}</p>}
      </div>
      {actions && <div className="flex shrink-0 gap-2">{actions}</div>}
    </div>
  );
}
