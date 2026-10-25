import { useEffect, useState } from "react";
import { useParams, Link } from "react-router-dom";
import ReactMarkdown from "react-markdown";
import {
  Sparkles,
  Copy,
  Check,
  FileText,
  Code,
  FormInput,
  Calendar,
  Building2,
  User as UserIcon,
  Share2,
  AlertCircle,
  ArrowLeft,
} from "lucide-react";
import { toast, Toaster } from "sonner";
import { Card, Badge, Button } from "@/components/ui-kit";
import { apiGet } from "@/lib/api";

interface PublicSharedResult {
  id: string;
  share_token: string;
  source_type: "form" | "flow";
  source_id: string;
  source_name: string;
  input_summary: Record<string, any> | null;
  output_text: string;
  output_json: Record<string, any> | any[] | null;
  shared_at: string | null;
  created_at: string | null;
  user_name: string;
  organization_name: string | null;
}

export default function SharedResultPage() {
  const { shareToken } = useParams<{ shareToken: string }>();
  const [result, setResult] = useState<PublicSharedResult | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<"markdown" | "json" | "inputs">("markdown");
  const [copied, setCopied] = useState(false);
  const [copiedLink, setCopiedLink] = useState(false);

  useEffect(() => {
    if (!shareToken) {
      setError("Invalid share link.");
      setLoading(false);
      return;
    }

    setLoading(true);
    apiGet<PublicSharedResult>(`/shared/results/${shareToken}`)
      .then((data) => {
        setResult(data);
        setError(null);
      })
      .catch((err: any) => {
        setError(err.message || "This shared result could not be found or has been unshared.");
      })
      .finally(() => {
        setLoading(false);
      });
  }, [shareToken]);

  const handleCopyOutput = () => {
    if (!result?.output_text) return;
    navigator.clipboard.writeText(result.output_text);
    setCopied(true);
    toast.success("Result output copied to clipboard.");
    setTimeout(() => setCopied(false), 2000);
  };

  const handleCopyLink = () => {
    navigator.clipboard.writeText(window.location.href);
    setCopiedLink(true);
    toast.success("Share link copied to clipboard.");
    setTimeout(() => setCopiedLink(false), 2000);
  };

  return (
    <div className="min-h-screen bg-background flex flex-col text-foreground">
      <Toaster position="bottom-right" />
      {/* Header */}
      <header className="h-16 border-b border-border bg-card/60 backdrop-blur sticky top-0 z-10">
        <div className="max-w-5xl mx-auto h-full px-6 flex items-center justify-between">
          <Link to="/" className="flex items-center gap-2 font-semibold tracking-tight hover:opacity-90 transition-opacity">
            <Sparkles className="h-4 w-4" strokeWidth={2.25} />
            <span>PromptFlow</span>
            <Badge tone="muted" className="ml-2 text-[10px] uppercase font-mono tracking-wider">
              Shared View
            </Badge>
          </Link>
          <div className="flex items-center gap-2">
            <Link to="/login">
              <Button size="sm" variant="ghost" className="text-xs">
                Log In
              </Button>
            </Link>
            <Link to="/">
              <Button size="sm" variant="secondary" className="text-xs">
                Open App
              </Button>
            </Link>
          </div>
        </div>
      </header>

      {/* Main Container */}
      <main className="flex-1 max-w-5xl w-full mx-auto px-6 py-8">
        {loading ? (
          <div className="flex flex-col items-center justify-center py-24 text-muted-foreground">
            <div className="h-6 w-6 animate-spin rounded-full border-2 border-foreground border-t-transparent" />
            <p className="mt-4 text-sm font-medium">Loading shared result...</p>
          </div>
        ) : error || !result ? (
          <div className="max-w-md mx-auto py-16 text-center">
            <div className="mx-auto flex h-12 w-12 items-center justify-center rounded-full bg-destructive/10 text-destructive mb-4">
              <AlertCircle className="h-6 w-6" />
            </div>
            <h2 className="text-lg font-semibold tracking-tight">Shared Result Unavailable</h2>
            <p className="mt-2 text-sm text-muted-foreground leading-relaxed">
              {error || "This shared result does not exist or sharing has been disabled by the author."}
            </p>
            <div className="mt-6 flex items-center justify-center gap-3">
              <Link to="/">
                <Button size="sm" variant="secondary">
                  <ArrowLeft className="h-3.5 w-3.5 mr-1" /> Return Home
                </Button>
              </Link>
            </div>
          </div>
        ) : (
          <div className="space-y-6">
            {/* Header Card */}
            <Card className="p-6 border border-border shadow-xs">
              <div className="flex flex-col gap-4 sm:flex-row sm:items-start sm:justify-between">
                <div>
                  <div className="flex items-center gap-2 mb-2">
                    <Badge tone={result.source_type === "form" ? "neutral" : "muted"}>
                      <span className="capitalize">{result.source_type}</span>
                    </Badge>
                    <Badge tone="neutral" className="gap-1 text-[11px]">
                      <Share2 className="h-3 w-3 text-emerald-500" />
                      Shared Output
                    </Badge>
                  </div>
                  <h1 className="text-xl font-bold tracking-tight text-foreground">{result.source_name}</h1>
                  <div className="mt-2 flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-muted-foreground">
                    <span className="flex items-center gap-1">
                      <UserIcon className="h-3.5 w-3.5" />
                      {result.user_name}
                    </span>
                    {result.organization_name && (
                      <span className="flex items-center gap-1">
                        <Building2 className="h-3.5 w-3.5" />
                        {result.organization_name}
                      </span>
                    )}
                    <span className="flex items-center gap-1">
                      <Calendar className="h-3.5 w-3.5" />
                      {result.created_at ? new Date(result.created_at).toLocaleString() : ""}
                    </span>
                  </div>
                </div>

                <div className="flex items-center gap-2">
                  <Button size="sm" variant="secondary" onClick={handleCopyLink} title="Copy Share Link">
                    {copiedLink ? <Check className="h-3.5 w-3.5 mr-1 text-emerald-500" /> : <Share2 className="h-3.5 w-3.5 mr-1" />}
                    {copiedLink ? "Link Copied" : "Share Link"}
                  </Button>
                  <Button size="sm" variant="primary" onClick={handleCopyOutput}>
                    {copied ? <Check className="h-3.5 w-3.5 mr-1 text-emerald-400" /> : <Copy className="h-3.5 w-3.5 mr-1" />}
                    {copied ? "Copied" : "Copy Output"}
                  </Button>
                </div>
              </div>
            </Card>

            {/* Content Tabs */}
            <Card className="p-0 border border-border overflow-hidden shadow-xs">
              <div className="flex items-center border-b border-border bg-muted/20 px-4 gap-2 text-xs">
                <button
                  onClick={() => setActiveTab("markdown")}
                  className={`flex items-center gap-1.5 border-b-2 py-3 px-3 font-medium transition-colors ${
                    activeTab === "markdown"
                      ? "border-foreground text-foreground"
                      : "border-transparent text-muted-foreground hover:text-foreground"
                  }`}
                >
                  <FileText className="h-3.5 w-3.5" />
                  Formatted Output
                </button>
                {result.output_json && (
                  <button
                    onClick={() => setActiveTab("json")}
                    className={`flex items-center gap-1.5 border-b-2 py-3 px-3 font-medium transition-colors ${
                      activeTab === "json"
                        ? "border-foreground text-foreground"
                        : "border-transparent text-muted-foreground hover:text-foreground"
                    }`}
                  >
                    <Code className="h-3.5 w-3.5" />
                    Structured JSON
                  </button>
                )}
                {result.input_summary && (
                  <button
                    onClick={() => setActiveTab("inputs")}
                    className={`flex items-center gap-1.5 border-b-2 py-3 px-3 font-medium transition-colors ${
                      activeTab === "inputs"
                        ? "border-foreground text-foreground"
                        : "border-transparent text-muted-foreground hover:text-foreground"
                    }`}
                  >
                    <FormInput className="h-3.5 w-3.5" />
                    Inputs Summary
                  </button>
                )}
              </div>

              <div className="p-6">
                {activeTab === "markdown" && (
                  <div className="prose prose-sm dark:prose-invert max-w-none text-xs leading-relaxed text-foreground whitespace-pre-wrap font-sans">
                    <ReactMarkdown>{result.output_text}</ReactMarkdown>
                  </div>
                )}

                {activeTab === "json" && result.output_json && (
                  <pre className="font-mono text-xs overflow-x-auto text-foreground p-4 bg-muted/40 rounded-lg border border-border">
                    {JSON.stringify(result.output_json, null, 2)}
                  </pre>
                )}

                {activeTab === "inputs" && result.input_summary && (
                  <div className="space-y-3">
                    {Object.entries(result.input_summary).map(([k, v]) => (
                      <div key={k} className="flex flex-col gap-1 rounded-lg bg-muted/30 p-3.5 border border-border text-xs">
                        <span className="font-semibold text-muted-foreground font-mono">{k}:</span>
                        <span className="text-foreground font-mono whitespace-pre-wrap">
                          {typeof v === "object" ? JSON.stringify(v, null, 2) : String(v)}
                        </span>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </Card>
          </div>
        )}
      </main>
    </div>
  );
}
