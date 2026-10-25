import { useState } from "react";
import { useParams } from "react-router-dom";
import { Share2, Copy, Check, ShieldCheck } from "lucide-react";
import { toast } from "sonner";
import { AppShell, PageHeader } from "@/components/app-shell";
import { Button, Input } from "@/components/ui-kit";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
} from "@/components/ui/dialog";
import { SuperadminFlowRunner } from "@/components/SuperadminFlowRunner";
import { useCurrentUserId } from "@/lib/use-current-user-id";

export default function FlowDetailPage() {
  const { flowId } = useParams();
  const userId = useCurrentUserId();
  const [shareOpen, setShareOpen] = useState(false);
  const [copied, setCopied] = useState(false);

  const flowUrl = typeof window !== "undefined" && flowId ? `${window.location.origin}/flows/${flowId}` : "";

  const handleCopy = () => {
    if (!flowUrl) return;
    navigator.clipboard.writeText(flowUrl);
    setCopied(true);
    toast.success("Flow link copied to clipboard!");
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <AppShell>
      <PageHeader
        title="Flow Runner"
        description="Execute this workflow."
        actions={
          <Button
            size="sm"
            variant="secondary"
            className="text-xs gap-1.5"
            onClick={() => {
              setShareOpen(true);
              setCopied(false);
            }}
          >
            <Share2 className="h-3.5 w-3.5" />
            Share Flow
          </Button>
        }
      />
      {/* key=userId forces a full remount when the authenticated user changes,
          clearing all diagnostic state so data permitted to one user cannot
          remain visible to the next user on the same tab. */}
      <SuperadminFlowRunner key={userId ?? "loading"} flowId={flowId ?? ""} />

      {/* Share Flow Dialog */}
      <Dialog open={shareOpen} onOpenChange={setShareOpen}>
        <DialogContent className="max-w-md">
          <DialogHeader>
            <div className="flex items-center gap-2">
              <div className="flex h-8 w-8 items-center justify-center rounded-full bg-primary/10 text-primary">
                <Share2 className="h-4 w-4" />
              </div>
              <div>
                <DialogTitle className="text-base font-semibold">Share Flow</DialogTitle>
                <DialogDescription className="text-xs">
                  Copy direct access link to this flow runner
                </DialogDescription>
              </div>
            </div>
          </DialogHeader>

          <div className="space-y-4 py-2">
            <div className="space-y-2">
              <label className="text-xs font-medium text-foreground">Direct Flow URL</label>
              <div className="flex items-center gap-2">
                <Input
                  readOnly
                  value={flowUrl}
                  className="text-xs font-mono select-all bg-muted/40"
                />
                <Button
                  size="sm"
                  variant="primary"
                  className="shrink-0 text-xs"
                  onClick={handleCopy}
                >
                  {copied ? <Check className="h-3.5 w-3.5 mr-1" /> : <Copy className="h-3.5 w-3.5 mr-1" />}
                  {copied ? "Copied" : "Copy"}
                </Button>
              </div>
            </div>

            <div className="rounded-lg border border-border bg-muted/30 p-3 text-xs text-muted-foreground space-y-1">
              <div className="font-semibold text-foreground flex items-center gap-1.5">
                <ShieldCheck className="h-3.5 w-3.5 text-primary" />
                Authentication & Access Required
              </div>
              <p className="leading-relaxed">
                Users opening this link must log in. Only users whose organization has access to this flow can run it.
              </p>
            </div>
          </div>

          <div className="flex justify-end pt-2 border-t border-border">
            <Button size="sm" variant="secondary" onClick={() => setShareOpen(false)}>
              Close
            </Button>
          </div>
        </DialogContent>
      </Dialog>
    </AppShell>
  );
}
