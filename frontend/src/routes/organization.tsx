import { createFileRoute } from "@tanstack/react-router";
import { useEffect, useMemo, useState } from "react";
import { Building2, Copy, Check, Link as LinkIcon } from "lucide-react";
import { AppShell, PageHeader } from "@/components/app-shell";
import { Badge, Button, Card } from "@/components/ui-kit";
import { apiGet } from "@/lib/api";
import { authService } from "@/lib/auth";

type Organization = {
  id: string;
  name: string;
  slug: string;
  code: string;
  is_active: boolean;
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

export const Route = createFileRoute("/organization")({
  component: OrganizationPage,
});

function OrganizationPage() {
  const [user, setUser] = useState<AuthUser | null>(null);
  const [organization, setOrganization] = useState<Organization | null>(null);
  const [loading, setLoading] = useState(true);
  const [showLink, setShowLink] = useState(false);
  const [copied, setCopied] = useState(false);

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

  function copyRegistrationLink() {
    if (!registrationLink) return;
    navigator.clipboard?.writeText(registrationLink);
    setCopied(true);
    window.setTimeout(() => setCopied(false), 1500);
  }

  return (
    <AppShell>
      <PageHeader title="Organization" description="Your organization scope is read-only here." />

      <div className="space-y-4">
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

        {organization && (
          <Card className="p-5">
            <div className="flex items-start justify-between gap-4">
              <div>
                <div className="flex items-center gap-2 text-sm font-medium">
                  <LinkIcon className="h-4 w-4" />
                  Registration link
                </div>
                <div className="text-xs text-muted-foreground mt-1 max-w-md">
                  Share this link with team members to let them join this organization.
                </div>
              </div>
              <Button variant="secondary" onClick={() => setShowLink(true)}>
                <Copy className="h-4 w-4" />
                Copy Registration Link
              </Button>
            </div>
          </Card>
        )}

        {organization && (
          <Card className="p-5">
            <div className="font-medium">Members</div>
            <div className="mt-4 space-y-3">
              {(organization.users || []).map((member) => (
                <div
                  key={member.id}
                  className="flex items-center justify-between rounded-md border border-border p-3"
                >
                  <div>
                    <div className="text-sm font-medium">{member.name}</div>
                    <div className="text-xs text-muted-foreground">{member.email}</div>
                  </div>
                  <div className="text-xs text-muted-foreground">{member.role || "Member"}</div>
                </div>
              ))}
              {organization.users && organization.users.length === 0 && (
                <p className="text-sm text-muted-foreground">No members assigned.</p>
              )}
            </div>
          </Card>
        )}
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
