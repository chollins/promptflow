import { createFileRoute } from "@tanstack/react-router";
import { useMemo, useState } from "react";
import { Search, Check, Minus, Building2, Workflow, ShieldCheck } from "lucide-react";
import { AppShell, PageHeader } from "@/components/app-shell";
import { Button, Card, Input, Badge } from "@/components/ui-kit";
import { ORGANIZATIONS, FLOWS, type Organization } from "@/lib/mock-store";

export const Route = createFileRoute("/admin/manage-flows")({
  head: () => ({
    meta: [
      { title: "Manage Flow Access — PromptFlow" },
      { name: "description", content: "Assign or remove flow access across organizations." },
      { property: "og:title", content: "Manage Flow Access — PromptFlow" },
      { property: "og:description", content: "Assign or remove flow access across organizations." },
    ],
  }),
  component: ManageFlowsPage,
});

function ManageFlowsPage() {
  const [orgs, setOrgs] = useState<Organization[]>(ORGANIZATIONS);
  const [query, setQuery] = useState("");
  const [planFilter, setPlanFilter] = useState<"All" | Organization["plan"]>("All");

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    return orgs.filter((o) => {
      const matchesQ = !q || o.name.toLowerCase().includes(q) || o.code.toLowerCase().includes(q);
      const matchesP = planFilter === "All" || o.plan === planFilter;
      return matchesQ && matchesP;
    });
  }, [orgs, query, planFilter]);

  const toggle = (orgId: string, flowId: string) => {
    setOrgs((prev) =>
      prev.map((o) =>
        o.id !== orgId
          ? o
          : {
              ...o,
              flowIds: o.flowIds.includes(flowId)
                ? o.flowIds.filter((f) => f !== flowId)
                : [...o.flowIds, flowId],
            },
      ),
    );
  };

  const setAll = (orgId: string, assign: boolean) => {
    setOrgs((prev) =>
      prev.map((o) =>
        o.id !== orgId ? o : { ...o, flowIds: assign ? FLOWS.map((f) => f.id) : [] },
      ),
    );
  };

  const totalAssignments = orgs.reduce((sum, o) => sum + o.flowIds.length, 0);

  return (
    <AppShell>
      <PageHeader
        title="Manage Flow Access"
        description="Assign or remove flows for each organization on the platform."
      />

      <div className="grid grid-cols-1 md:grid-cols-3 gap-3 mb-6">
        <StatCard icon={Building2} label="Organizations" value={orgs.length} />
        <StatCard icon={Workflow} label="Available Flows" value={FLOWS.length} />
        <StatCard icon={ShieldCheck} label="Active Assignments" value={totalAssignments} />
      </div>

      <div className="flex flex-col sm:flex-row gap-3 mb-4">
        <div className="relative flex-1">
          <Search className="h-4 w-4 absolute left-3 top-1/2 -translate-y-1/2 text-muted-foreground" />
          <Input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search organizations…"
            className="pl-9"
          />
        </div>
        <div className="flex rounded-md border border-border p-0.5">
          {(["All", "Free", "Team", "Enterprise"] as const).map((p) => (
            <button
              key={p}
              onClick={() => setPlanFilter(p)}
              className={
                "text-sm px-3 py-1.5 rounded-sm transition-colors " +
                (planFilter === p
                  ? "bg-foreground text-background"
                  : "text-muted-foreground hover:text-foreground")
              }
            >
              {p}
            </button>
          ))}
        </div>
      </div>

      <div className="rounded-xl border border-border overflow-hidden bg-card">
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-border bg-muted/50">
                <th className="text-left font-medium text-xs uppercase tracking-wider text-muted-foreground px-6 py-3 sticky left-0 bg-muted/50">
                  Organization
                </th>
                {FLOWS.map((f) => (
                  <th
                    key={f.id}
                    className="text-left font-medium text-xs uppercase tracking-wider text-muted-foreground px-4 py-3 whitespace-nowrap"
                  >
                    {f.name}
                  </th>
                ))}
                <th className="px-4 py-3 text-right font-medium text-xs uppercase tracking-wider text-muted-foreground">
                  Bulk
                </th>
              </tr>
            </thead>
            <tbody>
              {filtered.map((o, i) => {
                const all = o.flowIds.length === FLOWS.length;
                const none = o.flowIds.length === 0;
                return (
                  <tr key={o.id} className={i > 0 ? "border-t border-border" : ""}>
                    <td className="px-6 py-4 sticky left-0 bg-card">
                      <div className="flex items-center gap-3">
                        <div className="h-8 w-8 rounded-md border border-border flex items-center justify-center">
                          <Building2 className="h-4 w-4" />
                        </div>
                        <div>
                          <div className="font-medium leading-tight">{o.name}</div>
                          <div className="text-xs text-muted-foreground font-mono">
                            {o.code} · {o.plan}
                          </div>
                        </div>
                      </div>
                    </td>
                    {FLOWS.map((f) => {
                      const assigned = o.flowIds.includes(f.id);
                      return (
                        <td key={f.id} className="px-4 py-4">
                          <button
                            onClick={() => toggle(o.id, f.id)}
                            aria-pressed={assigned}
                            title={assigned ? "Remove access" : "Assign access"}
                            className={
                              "h-7 w-7 rounded-md border flex items-center justify-center transition-colors " +
                              (assigned
                                ? "bg-foreground border-foreground text-background hover:bg-foreground/80"
                                : "border-border text-muted-foreground hover:text-foreground hover:border-foreground/40")
                            }
                          >
                            {assigned ? <Check className="h-4 w-4" /> : <Minus className="h-3 w-3" />}
                          </button>
                        </td>
                      );
                    })}
                    <td className="px-4 py-4">
                      <div className="flex items-center gap-2 justify-end">
                        <Badge tone="muted">
                          {o.flowIds.length}/{FLOWS.length}
                        </Badge>
                        <Button
                          variant="secondary"
                          onClick={() => setAll(o.id, true)}
                          disabled={all}
                          className="h-7 px-2 text-xs"
                        >
                          All
                        </Button>
                        <Button
                          variant="secondary"
                          onClick={() => setAll(o.id, false)}
                          disabled={none}
                          className="h-7 px-2 text-xs"
                        >
                          None
                        </Button>
                      </div>
                    </td>
                  </tr>
                );
              })}
              {filtered.length === 0 && (
                <tr>
                  <td
                    colSpan={FLOWS.length + 2}
                    className="px-6 py-12 text-center text-sm text-muted-foreground"
                  >
                    No organizations match your search.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>

      <div className="mt-6">
        <Card className="p-5">
          <div className="text-sm font-medium mb-3">Flow legend</div>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            {FLOWS.map((f) => (
              <div key={f.id} className="flex items-start gap-3">
                <div className="h-8 w-8 rounded-md border border-border flex items-center justify-center shrink-0">
                  <Workflow className="h-4 w-4" />
                </div>
                <div className="min-w-0">
                  <div className="text-sm font-medium flex items-center gap-2">
                    {f.name}
                    {!f.enabled && <Badge tone="muted">Disabled globally</Badge>}
                  </div>
                  <div className="text-xs text-muted-foreground">{f.description}</div>
                </div>
              </div>
            ))}
          </div>
        </Card>
      </div>
    </AppShell>
  );
}

function StatCard({
  icon: Icon,
  label,
  value,
}: {
  icon: typeof Building2;
  label: string;
  value: number;
}) {
  return (
    <Card className="p-4 flex items-center gap-3">
      <div className="h-10 w-10 rounded-md border border-border flex items-center justify-center">
        <Icon className="h-4 w-4" />
      </div>
      <div>
        <div className="text-xs text-muted-foreground">{label}</div>
        <div className="text-xl font-medium">{value}</div>
      </div>
    </Card>
  );
}
