import { Link, useRouterState } from "@tanstack/react-router";
import {
  LayoutDashboard,
  Workflow,
  Users,
  Building2,
  User,
  ListTree,
  ShieldCheck,
  Sparkles,
} from "lucide-react";
import type { ReactNode } from "react";

const ALL_ITEMS: {
  title: string;
  url: string;
  icon: typeof LayoutDashboard;
}[] = [
  { title: "Dashboard", url: "/dashboard", icon: LayoutDashboard },
  { title: "Flows", url: "/flows", icon: Workflow },
  { title: "Users", url: "/users", icon: Users },
  { title: "Organization", url: "/organization", icon: Building2 },
  { title: "Profile", url: "/profile", icon: User },
  { title: "Organizations", url: "/admin/organizations", icon: Building2 },
  { title: "Flows Catalog", url: "/admin/flows", icon: ListTree },
  { title: "Manage Flow Access", url: "/admin/manage-flows", icon: ShieldCheck },
];

export function AppShell({ children }: { children: ReactNode }) {
  const pathname = useRouterState({ select: (s) => s.location.pathname });
  const items = ALL_ITEMS;

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
