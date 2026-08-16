import { createFileRoute } from "@tanstack/react-router";
import { useState } from "react";
import { Copy, Check, X } from "lucide-react";
import { AppShell, PageHeader } from "@/components/app-shell";
import { Button, Card } from "@/components/ui-kit";
import { ORG } from "@/lib/mock-store";

export const Route = createFileRoute("/organization")({
  component: OrgPage,
});

function OrgPage() {
  const [showLink, setShowLink] = useState(false);
  const link = `https://promptflow.ai/signup?oc=${ORG.code}`;

  return (
    <AppShell>
      <PageHeader title="Organization" description="Workspace details and registration." />

      <Card className="mb-4">
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-6">
          <div>
            <div className="text-xs text-muted-foreground mb-1.5">Organization Name</div>
            <div className="text-base font-medium">{ORG.name}</div>
          </div>
          <div>
            <div className="text-xs text-muted-foreground mb-1.5">Organization Code</div>
            <div className="text-base font-mono font-medium">{ORG.code}</div>
          </div>
        </div>
      </Card>

      <Card>
        <div className="flex items-start justify-between gap-4">
          <div>
            <div className="font-medium">Registration link</div>
            <div className="text-xs text-muted-foreground mt-1 max-w-md">
              Share this link with team members to let them join using your organization code.
            </div>
          </div>
          <Button variant="secondary" onClick={() => setShowLink(true)}>
            <Copy className="h-4 w-4" />
            Copy Registration Link
          </Button>
        </div>
      </Card>

      {showLink && <LinkModal link={link} onClose={() => setShowLink(false)} />}
    </AppShell>
  );
}

function LinkModal({ link, onClose }: { link: string; onClose: () => void }) {
  const [copied, setCopied] = useState(false);
  const copy = () => {
    navigator.clipboard?.writeText(link);
    setCopied(true);
    setTimeout(() => setCopied(false), 1500);
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-foreground/20 backdrop-blur-sm p-4">
      <div className="w-full max-w-md rounded-xl border border-border bg-background shadow-xl">
        <div className="flex items-center justify-between px-6 py-4 border-b border-border">
          <div className="font-medium">Registration link</div>
          <button onClick={onClose} className="text-muted-foreground hover:text-foreground">
            <X className="h-4 w-4" />
          </button>
        </div>
        <div className="p-6 space-y-4">
          <div className="text-xs text-muted-foreground">
            Anyone with this link can register into <b className="text-foreground">{ORG.name}</b>.
          </div>
          <div className="flex items-center gap-2 rounded-md border border-border bg-muted/50 px-3 py-2">
            <div className="text-xs font-mono truncate flex-1">{link}</div>
            <button onClick={copy} className="text-muted-foreground hover:text-foreground shrink-0">
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
