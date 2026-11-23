import { useEffect, useMemo, useState } from "react";
import {
  Building2,
  Copy,
  Check,
  Cpu,
  Workflow,
} from "lucide-react";
import { AppShell, PageHeader } from "@/components/app-shell";
import { Badge, Button, Card } from "@/components/ui-kit";
import { apiGet, apiPut } from "@/lib/api";
import { authService } from "@/lib/auth";
import { toast } from "sonner";

type Organization = {
  id: string;
  name: string;
  slug: string;
  code: string;
  is_active: boolean;
  auto_next_step?: boolean;
  allow_user_create_flows?: boolean;
  allow_user_share_flows?: boolean;
  created_at?: string | null;
  updated_at?: string | null;
  users?: Array<{
    id: string;
    name: string;
    email: string;
    role: string | null;
    is_active: boolean;
  }>;
};

type AuthUser = {
  id: string;
  name: string;
  email: string;
  role: string | null;
  organization_id: string;
};

export default function OrganizationPage() {
  const [user, setUser] = useState<AuthUser | null>(null);
  const [organization, setOrganization] = useState<Organization | null>(null);
  const [loading, setLoading] = useState(true);
  const [showLink, setShowLink] = useState(false);
  const [copied, setCopied] = useState(false);

  // Workflow & Member Permissions Settings
  const [autoNextStep, setAutoNextStep] = useState(false);
  const [allowCreateFlows, setAllowCreateFlows] = useState(false);
  const [allowShareFlows, setAllowShareFlows] = useState(false);
  const [savingSettings, setSavingSettings] = useState(false);

  const registrationLink = useMemo(() => {
    if (!organization) return "";
    if (typeof window === "undefined") return "";
    return `${window.location.origin}/signup?oc=${organization.code}`;
  }, [organization]);

  useEffect(() => {
    let active = true;

    async function load() {
      setLoading(true);
      try {
        const me = await authService.getMe();
        if (!active) return;
        setUser(me);

        if (!me.organization_id) {
          setOrganization(null);
          return;
        }

        const org = await apiGet<Organization>(`/organizations/${me.organization_id}`);
        if (!active) return;
        setOrganization(org);
        setAutoNextStep(Boolean(org.auto_next_step));
        setAllowCreateFlows(Boolean(org.allow_user_create_flows));
        setAllowShareFlows(Boolean(org.allow_user_share_flows));
      } catch {
        if (active) {
          setUser(null);
          setOrganization(null);
        }
      } finally {
        if (active) setLoading(false);
      }
    }

    void load();

    return () => {
      active = false;
    };
  }, []);

  async function handleSaveSettings() {
    if (!organization) return;
    setSavingSettings(true);
    try {
      const updated = await apiPut<Organization>(`/organizations/${organization.id}`, {
        auto_next_step: autoNextStep,
        allow_user_create_flows: allowCreateFlows,
        allow_user_share_flows: allowShareFlows,
      });
      setOrganization(updated);
      toast.success("Organization settings saved successfully");
    } catch (err: any) {
      toast.error(err.message || "Failed to save settings");
    } finally {
      setSavingSettings(false);
    }
  }

  function copyRegistrationLink() {
    if (!registrationLink) return;
    navigator.clipboard?.writeText(registrationLink);
    setCopied(true);
    window.setTimeout(() => setCopied(false), 1500);
  }

  return (
    <AppShell>
      <PageHeader
        title="Organization"
        description="Manage your organization profile, membership details, and invite registration links."
      />

      <div className="space-y-6">
        {/* Organization details card */}
        <Card className="p-5">
          <div className="flex items-start justify-between gap-4">
            <div>
              <div className="flex items-center gap-2 text-sm font-medium">
                <Building2 className="h-4 w-4" />
                Organization details
              </div>
              <p className="mt-1 text-sm text-muted-foreground">
                {loading
                  ? "Loading your organization..."
                  : organization
                    ? "Loaded from your account membership."
                    : "No organization assigned to this account."}
              </p>
            </div>
            {organization && (
              <Badge tone={organization.is_active ? "neutral" : "muted"}>
                {organization.is_active ? "Active" : "Inactive"}
              </Badge>
            )}
          </div>

          {organization && (
            <div className="mt-6 grid gap-4 md:grid-cols-2">
              <div>
                <div className="text-xs text-muted-foreground mb-1.5">Organization Name</div>
                <div className="text-base font-medium">{organization.name}</div>
              </div>
              <div>
                <div className="text-xs text-muted-foreground mb-1.5">Organization Code</div>
                <div className="text-base font-mono font-medium">{organization.code}</div>
              </div>
              <div>
                <div className="text-xs text-muted-foreground mb-1.5">Slug</div>
                <div className="text-sm font-mono">{organization.slug}</div>
              </div>
              <div>
                <div className="text-xs text-muted-foreground mb-1.5">Signed-in user</div>
                <div className="text-sm">
                  {user?.name || "Unknown"} {user?.role ? `(${user.role})` : ""}
                </div>
              </div>
            </div>
          )}
        </Card>

        {/* Workflow & User Permissions Settings Card */}
        {organization && (
          <Card className="p-5 space-y-5">
            <div className="flex items-center justify-between border-b border-border pb-3">
              <div>
                <div className="flex items-center gap-2 text-sm font-semibold text-foreground">
                  <Workflow className="h-4 w-4 text-primary" />
                  Workflow &amp; Member Permissions
                </div>
                <p className="text-xs text-muted-foreground mt-0.5">
                  Configure execution defaults and member capabilities across your organization.
                </p>
              </div>
              <Button
                size="sm"
                onClick={handleSaveSettings}
                disabled={savingSettings}
                className="text-xs"
              >
                {savingSettings ? "Saving..." : "Save Settings"}
              </Button>
            </div>

            {/* Auto Next Step setting */}
            <div className="space-y-2">
              <label className="flex items-start gap-3 cursor-pointer select-none">
                <input
                  type="checkbox"
                  checked={autoNextStep}
                  onChange={(e) => setAutoNextStep(e.target.checked)}
                  className="mt-0.5 h-4 w-4 rounded border-border text-primary focus:ring-primary cursor-pointer"
                />
                <div className="space-y-0.5">
                  <span className="text-sm font-medium text-foreground">Auto Next Step</span>
                  <p className="text-xs text-muted-foreground">
                    Determines whether multi-step flows automatically advance to the next step after successful execution.
                  </p>
                </div>
              </label>
            </div>

            {/* Allow user to: section */}
            <div className="border-t border-border pt-4 space-y-3">
              <div className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                Allow user to:
              </div>
              <div className="grid gap-3 sm:grid-cols-2">
                <label className="flex items-center gap-2.5 p-3 rounded-lg border border-border bg-muted/10 hover:bg-muted/20 transition-colors cursor-pointer select-none">
                  <input
                    type="checkbox"
                    checked={allowCreateFlows}
                    onChange={(e) => setAllowCreateFlows(e.target.checked)}
                    className="h-4 w-4 rounded border-border text-primary focus:ring-primary cursor-pointer"
                  />
                  <div>
                    <div className="text-xs font-medium text-foreground">Create flows</div>
                    <div className="text-[11px] text-muted-foreground">Allow regular members to create and edit flows</div>
                  </div>
                </label>

                <label className="flex items-center gap-2.5 p-3 rounded-lg border border-border bg-muted/10 hover:bg-muted/20 transition-colors cursor-pointer select-none">
                  <input
                    type="checkbox"
                    checked={allowShareFlows}
                    onChange={(e) => setAllowShareFlows(e.target.checked)}
                    className="h-4 w-4 rounded border-border text-primary focus:ring-primary cursor-pointer"
                  />
                  <div>
                    <div className="text-xs font-medium text-foreground">Share flows</div>
                    <div className="text-[11px] text-muted-foreground">Allow regular members to share flows and execution results</div>
                  </div>
                </label>
              </div>
            </div>
          </Card>
        )}

        {/* AI Models quick link banner */}
        <div className="flex flex-col sm:flex-row sm:items-center justify-between p-5 rounded-xl border border-border bg-muted/20 gap-4">
          <div className="flex items-center gap-3">
            <div className="h-10 w-10 rounded-lg bg-primary/10 border border-primary/20 text-primary flex items-center justify-center shrink-0">
              <Cpu className="h-5 w-5" />
            </div>
            <div>
              <div className="text-sm font-semibold text-foreground">AI Model Management</div>
              <p className="text-xs text-muted-foreground">
                Browse available models, customize organization nicknames, and set favorites in the Models directory.
              </p>
            </div>
          </div>
          <Button variant="outline" size="sm" asChild className="shrink-0">
            <a href="/models" className="text-xs">
              Manage Models &rarr;
            </a>
          </Button>
        </div>
      </div>

      {showLink && organization && (
        <LinkModal
          link={registrationLink}
          organizationName={organization.name}
          copied={copied}
          onCopy={copyRegistrationLink}
          onClose={() => setShowLink(false)}
        />
      )}
    </AppShell>
  );
}

function LinkModal({
  link,
  organizationName,
  copied,
  onCopy,
  onClose,
}: {
  link: string;
  organizationName: string;
  copied: boolean;
  onCopy: () => void;
  onClose: () => void;
}) {
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-foreground/20 backdrop-blur-sm p-4">
      <div className="w-full max-w-md rounded-xl border border-border bg-background shadow-xl">
        <div className="flex items-center justify-between border-b border-border px-6 py-4">
          <div className="font-medium">Registration link</div>
          <button onClick={onClose} className="text-muted-foreground hover:text-foreground">
            <Check className="h-4 w-4 opacity-0" />
          </button>
        </div>
        <div className="space-y-4 p-6">
          <div className="text-xs text-muted-foreground">
            Anyone with this link can register into{" "}
            <b className="text-foreground">{organizationName}</b>.
          </div>
          <div className="flex items-center gap-2 rounded-md border border-border bg-muted/50 px-3 py-2">
            <div className="min-w-0 flex-1 truncate font-mono text-xs">{link}</div>
            <button
              onClick={onCopy}
              className="shrink-0 text-muted-foreground hover:text-foreground"
            >
              {copied ? <Check className="h-3.5 w-3.5" /> : <Copy className="h-3.5 w-3.5" />}
            </button>
          </div>
          <div className="flex justify-end">
            <Button onClick={onClose}>Done</Button>
          </div>
        </div>
      </div>
    </div>
  );
}
