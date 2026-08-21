import { Link } from "react-router-dom";
import { ArrowRight, Plus, Users, Activity, FileText, Database } from "lucide-react";
import { useEffect, useState } from "react";
import { AppShell, PageHeader } from "@/components/app-shell";
import { Button, Card } from "@/components/ui-kit";
import { apiGet } from "@/lib/api";
import { getRecentActivity, ActivityLog } from "@/lib/activity";

type DashboardStats = {
  flows: number;
  forms: number;
  users: number;
  admins: number;
  profileName: string | null;
  role: string | null;
};

export default function Dashboard() {
  const [stats, setStats] = useState<DashboardStats>({
    flows: 0,
    forms: 0,
    users: 0,
    admins: 0,
    profileName: null,
    role: null,
  });
  const [loading, setLoading] = useState(true);
  const [activities, setActivities] = useState<ActivityLog[]>([]);

  useEffect(() => {
    let active = true;
    const refreshActivities = () => setActivities(getRecentActivity());
    refreshActivities();
    window.addEventListener("storage", refreshActivities);
    window.addEventListener("promptflow-activity-updated", refreshActivities as EventListener);
    
    Promise.all([
      apiGet<{ count: number }>("/flows").catch(() => ({ count: 0 })),
      apiGet<{ count: number }>("/forms").catch(() => ({ count: 0 })),
      apiGet<{ count: number; items?: { role?: string }[] }>("/users").catch(() => ({ count: 0, items: [] })),
      apiGet<{ name: string; role: string | null }>("/auth/me").catch(() => null),
    ])
      .then(([flows, forms, users, profile]) => {
        if (!active) return;
        
        let adminCount = 0;
        if (users.items) {
           adminCount = users.items.filter((u: any) => u.role === "admin" || u.role === "superadmin").length;
        }

        setStats({
          flows: flows?.count || 0,
          forms: forms?.count || 0,
          users: users?.count || 0,
          admins: adminCount,
          profileName: profile?.name ?? null,
          role: profile?.role ?? null,
        });
      })
      .catch(() => {
        // Ignored
      })
      .finally(() => {
        if (active) setLoading(false);
      });

    return () => {
      active = false;
      window.removeEventListener("storage", refreshActivities);
      window.removeEventListener("promptflow-activity-updated", refreshActivities as EventListener);
    };
  }, []);

  const isSuperadmin = stats.role === "superadmin";
  const isAdmin = stats.role === "admin";
  const isMember = stats.role === "member" || stats.role === "user";

  const formatTimeAgo = (isoString: string) => {
    const rtf = new Intl.RelativeTimeFormat('en', { numeric: 'auto' });
    const diff = new Date().getTime() - new Date(isoString).getTime();
    const minutes = Math.floor(diff / 60000);
    if (minutes < 1) return "Just now";
    if (minutes < 60) return rtf.format(-minutes, 'minute');
    const hours = Math.floor(minutes / 60);
    if (hours < 24) return rtf.format(-hours, 'hour');
    return rtf.format(-Math.floor(hours / 24), 'day');
  };

  return (
    <AppShell>
      <div className="w-full space-y-6">
        <PageHeader
          title="Dashboard"
          description={`Welcome back, ${stats.profileName || 'User'}.`}
        />

        {/* Summary Cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          <Card className="p-5 flex flex-col justify-between">
            <div>
              <div className="text-xs text-muted-foreground mb-2 flex items-center gap-2">
                <Activity className="h-4 w-4" /> Flows
              </div>
              <div className="text-2xl font-bold">{stats.flows}</div>
            </div>
            <div className="mt-4">
              <Link to="/flows" className="text-sm text-blue-600 hover:underline flex items-center gap-1">
                View flows <ArrowRight className="h-3 w-3" />
              </Link>
            </div>
          </Card>

          <Card className="p-5 flex flex-col justify-between">
            <div>
              <div className="text-xs text-muted-foreground mb-2 flex items-center gap-2">
                <FileText className="h-4 w-4" /> Forms
              </div>
              <div className="text-2xl font-bold">{stats.forms}</div>
            </div>
            <div className="mt-4">
              <Link to="/forms" className="text-sm text-blue-600 hover:underline flex items-center gap-1">
                View forms <ArrowRight className="h-3 w-3" />
              </Link>
            </div>
          </Card>

          {(isSuperadmin || isAdmin) && (
            <Card className="p-5 flex flex-col justify-between">
              <div>
                <div className="text-xs text-muted-foreground mb-2 flex items-center gap-2">
                  <Users className="h-4 w-4" /> Users
                </div>
                <div className="text-2xl font-bold">{stats.users}</div>
              </div>
              <div className="mt-4">
                <Link to="/users" className="text-sm text-blue-600 hover:underline flex items-center gap-1">
                  Manage users <ArrowRight className="h-3 w-3" />
                </Link>
              </div>
            </Card>
          )}

          {isSuperadmin && (
            <Card className="p-5 flex flex-col justify-between">
              <div>
                <div className="text-xs text-muted-foreground mb-2 flex items-center gap-2">
                  <Database className="h-4 w-4" /> Admins
                </div>
                <div className="text-2xl font-bold">{stats.admins}</div>
              </div>
              <div className="mt-4">
                <Link to="/users" className="text-sm text-blue-600 hover:underline flex items-center gap-1">
                  Manage admins <ArrowRight className="h-3 w-3" />
                </Link>
              </div>
            </Card>
          )}
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* System Status & Quick Actions */}
          <Card className="p-0 overflow-hidden flex flex-col">
            <div className="p-5 border-b flex items-center justify-between bg-muted/20">
              <h3 className="font-medium">System Status</h3>
              <div className="text-xs flex items-center gap-2 text-green-600 bg-green-50 px-2 py-1 rounded-full border border-green-100">
                <span className="h-2 w-2 rounded-full bg-green-500 animate-pulse"></span> All Operational
              </div>
            </div>
            <div className="p-5 flex-1">
              <ul className="space-y-4">
                <li className="flex justify-between text-sm">
                  <span className="text-muted-foreground">API Server</span>
                  <span className="font-medium text-green-600">Operational</span>
                </li>
                <li className="flex justify-between text-sm">
                  <span className="text-muted-foreground">Database</span>
                  <span className="font-medium text-green-600">Operational</span>
                </li>
                <li className="flex justify-between text-sm">
                  <span className="text-muted-foreground">Execution Engine</span>
                  <span className="font-medium text-green-600">Operational</span>
                </li>
              </ul>
              
              {(isSuperadmin || isAdmin) && (
                <div className="mt-8 pt-6 border-t">
                  <h4 className="text-sm font-medium mb-3">Quick Actions</h4>
                  <div className="flex flex-wrap gap-2">
                    <Link to="/users">
                      <Button size="sm" variant="outline" className="gap-2">
                        <Plus className="h-4 w-4" /> Invite User
                      </Button>
                    </Link>
                    {isSuperadmin && (
                      <>
                        <Link to="/admin/flows">
                          <Button size="sm" variant="outline" className="gap-2">
                            <Plus className="h-4 w-4" /> Create Flow
                          </Button>
                        </Link>
                        <Link to="/admin/forms">
                          <Button size="sm" variant="outline" className="gap-2">
                            <Plus className="h-4 w-4" /> Create Form
                          </Button>
                        </Link>
                      </>
                    )}
                  </div>
                </div>
              )}
            </div>
          </Card>

          {/* Secondary Panel: User Management (Admin) or Recent Activity (Member/Superadmin) */}
          {isAdmin && !isSuperadmin ? (
            <Card className="p-0 overflow-hidden flex flex-col">
              <div className="p-5 border-b bg-muted/20">
                <h3 className="font-medium">User Management</h3>
              </div>
              <div className="p-5 flex-1 flex flex-col justify-center items-center text-center">
                <Users className="h-12 w-12 text-muted-foreground/30 mb-4" />
                <h4 className="font-medium text-lg mb-1">{stats.users} Active Members</h4>
                <p className="text-sm text-muted-foreground mb-6">Manage organization members and pending invitations.</p>
                <Link to="/users">
                  <Button>Manage Users <ArrowRight className="h-4 w-4 ml-2" /></Button>
                </Link>
              </div>
            </Card>
          ) : (
            <Card className="p-0 overflow-hidden flex flex-col">
              <div className="p-5 border-b bg-muted/20 flex justify-between items-center">
                <h3 className="font-medium">Recent Activity</h3>
                <span className="text-xs text-muted-foreground">Local session</span>
              </div>
              <div className="p-5 flex-1">
                {activities.length === 0 ? (
                  <div className="h-full flex flex-col items-center justify-center text-muted-foreground text-sm min-h-[150px]">
                    No recent activity recorded yet.
                  </div>
                ) : (
                  <ul className="space-y-4">
                    {activities.map(activity => (
                      <li key={activity.id} className="flex justify-between items-start text-sm">
                        <div>
                          <span className="font-medium">{stats.profileName || "You"}</span> {activity.action} <span className="font-medium">"{activity.target}"</span>
                        </div>
                        <span className="text-muted-foreground whitespace-nowrap ml-4">
                          {formatTimeAgo(activity.timestamp)}
                        </span>
                      </li>
                    ))}
                  </ul>
                )}
              </div>
            </Card>
          )}
        </div>
      </div>
    </AppShell>
  );
}

