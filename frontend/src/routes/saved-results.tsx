import { useEffect, useState, useMemo } from "react";
import ReactMarkdown from "react-markdown";
import {
  BookmarkCheck,
  Search,
  Trash2,
  Copy,
  Check,
  Eye,
  FormInput,
  Workflow,
  Building2,
  User as UserIcon,
  Calendar,
  Code,
  FileText,
} from "lucide-react";
import { toast } from "sonner";
import { AppShell, PageHeader } from "@/components/app-shell";
import { Card, Badge, Button, Input } from "@/components/ui-kit";
import { apiDelete, apiGet } from "@/lib/api";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogFooter,
} from "@/components/ui/dialog";

export interface SavedResultItem {
  id: string;
  user_id: string;
  user_name: string;
  organization_id: string | null;
  organization_name: string | null;
  source_type: "form" | "flow";
  source_id: string;
  source_name: string;
  input_summary: Record<string, any> | null;
  output_text: string;
  output_json: Record<string, any> | any[] | null;
  created_at: string | null;
}

export default function SavedResultsPage() {
  const [items, setItems] = useState<SavedResultItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [search, setSearch] = useState("");
  const [filterType, setFilterType] = useState<"all" | "form" | "flow">("all");
  const [autoSaveEnabled, setAutoSaveEnabled] = useState(true);
  const [selectedResult, setSelectedResult] = useState<SavedResultItem | null>(null);
  const [activeTab, setActiveTab] = useState<"markdown" | "json" | "inputs">("markdown");
  const [copied, setCopied] = useState(false);
  const [deletingId, setDeletingId] = useState<string | null>(null);

  const fetchResults = async () => {
    try {
      setLoading(true);
      const params = new URLSearchParams();
      if (filterType !== "all") params.set("source_type", filterType);
      if (search.trim()) params.set("search", search.trim());

      const res = await apiGet<{
        items: SavedResultItem[];
        config?: { auto_save: boolean; org_admin_access: boolean };
      }>(`/saved-results?${params.toString()}`);

      setItems(res.items || []);
      if (res.config) {
        setAutoSaveEnabled(res.config.auto_save);
      }
    } catch (err: any) {
      toast.error(err.message || "Failed to load saved results.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    const timer = setTimeout(() => {
      fetchResults();
    }, 250);
    return () => clearTimeout(timer);
  }, [search, filterType]);

  const handleDelete = async (id: string, e?: React.MouseEvent) => {
    if (e) e.stopPropagation();
    if (!confirm("Are you sure you want to delete this saved result?")) return;

    try {
      setDeletingId(id);
      await apiDelete(`/saved-results/${id}`);
      toast.success("Saved result deleted.");
      setItems((prev) => prev.filter((item) => item.id !== id));
      if (selectedResult?.id === id) {
        setSelectedResult(null);
      }
    } catch (err: any) {
      toast.error(err.message || "Failed to delete saved result.");
    } finally {
      setDeletingId(null);
    }
  };

  const handleCopy = (text: string, e?: React.MouseEvent) => {
    if (e) e.stopPropagation();
    navigator.clipboard.writeText(text);
    setCopied(true);
    toast.success("Copied output to clipboard.");
    setTimeout(() => setCopied(false), 2000);
  };

  const formCount = useMemo(() => items.filter((i) => i.source_type === "form").length, [items]);
  const flowCount = useMemo(() => items.filter((i) => i.source_type === "flow").length, [items]);

  return (
    <AppShell>
      <PageHeader
        title="Saved Results"
        description="Archive of AI outputs generated from form and flow executions."
        actions={
          <div className="flex items-center gap-2">
            <Badge tone={autoSaveEnabled ? "neutral" : "outline"}>
              {autoSaveEnabled ? "Auto-Save Enabled" : "Explicit Save Mode"}
            </Badge>
          </div>
        }
      />

      {/* Filter and Search Toolbar */}
      <div className="mb-6 flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
        <div className="flex items-center gap-2">
          <button
            onClick={() => setFilterType("all")}
            className={`rounded-lg px-3 py-1.5 text-xs font-medium transition-colors ${
              filterType === "all"
                ? "bg-foreground text-background"
                : "bg-muted text-muted-foreground hover:text-foreground"
            }`}
          >
            All Results ({items.length})
          </button>
          <button
            onClick={() => setFilterType("form")}
            className={`flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-xs font-medium transition-colors ${
              filterType === "form"
                ? "bg-foreground text-background"
                : "bg-muted text-muted-foreground hover:text-foreground"
            }`}
          >
            <FormInput className="h-3.5 w-3.5" />
            Forms ({formCount})
          </button>
          <button
            onClick={() => setFilterType("flow")}
            className={`flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-xs font-medium transition-colors ${
              filterType === "flow"
                ? "bg-foreground text-background"
                : "bg-muted text-muted-foreground hover:text-foreground"
            }`}
          >
            <Workflow className="h-3.5 w-3.5" />
            Flows ({flowCount})
          </button>
        </div>

        <div className="relative w-full sm:w-64">
          <Search className="absolute left-3 top-2.5 h-4 w-4 text-muted-foreground" />
          <Input
            placeholder="Search outputs..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="pl-9 text-xs"
          />
        </div>
      </div>

      {/* Results Content */}
      {loading ? (
        <div className="py-12 text-center text-sm text-muted-foreground">Loading saved results...</div>
      ) : items.length === 0 ? (
        <Card className="py-12 text-center">
          <BookmarkCheck className="mx-auto h-8 w-8 text-muted-foreground/60" />
          <h3 className="mt-3 text-sm font-semibold">No saved results found</h3>
          <p className="mt-1 text-xs text-muted-foreground">
            {search ? "No outputs match your search filter." : "Execute a form or flow to save AI generated results."}
          </p>
        </Card>
      ) : (
        <div className="grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-3">
          {items.map((item) => (
            <Card
              key={item.id}
              className="group relative flex flex-col justify-between cursor-pointer transition-all hover:border-foreground/40 hover:shadow-sm"
              onClick={() => {
                setSelectedResult(item);
                setActiveTab("markdown");
              }}
            >
              <div>
                <div className="flex items-center justify-between gap-2">
                  <Badge tone={item.source_type === "form" ? "neutral" : "muted"}>
                    <span className="capitalize">{item.source_type}</span>
                  </Badge>
                  <span className="flex items-center gap-1 text-[11px] text-muted-foreground">
                    <Calendar className="h-3 w-3" />
                    {item.created_at ? new Date(item.created_at).toLocaleDateString() : ""}
                  </span>
                </div>

                <h3 className="mt-2.5 font-semibold text-sm tracking-tight text-foreground group-hover:text-foreground">
                  {item.source_name}
                </h3>

                <div className="mt-2 flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-muted-foreground">
                  <span className="flex items-center gap-1">
                    <UserIcon className="h-3 w-3" />
                    {item.user_name}
                  </span>
                  {item.organization_name && (
                    <span className="flex items-center gap-1">
                      <Building2 className="h-3 w-3" />
                      {item.organization_name}
                    </span>
                  )}
                </div>

                <div className="mt-3 line-clamp-4 rounded-lg bg-muted/50 p-3 font-mono text-xs text-foreground/90 whitespace-pre-wrap">
                  {item.output_text}
                </div>
              </div>

              <div className="mt-4 flex items-center justify-between border-t border-border pt-3">
                <Button
                  size="sm"
                  variant="ghost"
                  className="text-xs"
                  onClick={(e) => {
                    e.stopPropagation();
                    setSelectedResult(item);
                    setActiveTab("markdown");
                  }}
                >
                  <Eye className="h-3.5 w-3.5 mr-1" />
                  View Details
                </Button>

                <div className="flex items-center gap-1">
                  <Button
                    size="sm"
                    variant="ghost"
                    onClick={(e) => handleCopy(item.output_text, e)}
                    title="Copy Output"
                  >
                    <Copy className="h-3.5 w-3.5" />
                  </Button>
                  <Button
                    size="sm"
                    variant="ghost"
                    className="text-destructive hover:bg-destructive/10"
                    disabled={deletingId === item.id}
                    onClick={(e) => handleDelete(item.id, e)}
                    title="Delete Result"
                  >
                    <Trash2 className="h-3.5 w-3.5" />
                  </Button>
                </div>
              </div>
            </Card>
          ))}
        </div>
      )}

      {/* Result Detail Dialog */}
      <Dialog open={!!selectedResult} onOpenChange={(open) => !open && setSelectedResult(null)}>
        <DialogContent className="max-w-3xl max-h-[85vh] flex flex-col">
          {selectedResult && (
            <>
              <DialogHeader>
                <div className="flex items-center gap-2">
                  <Badge tone={selectedResult.source_type === "form" ? "neutral" : "muted"}>
                    <span className="capitalize">{selectedResult.source_type}</span>
                  </Badge>
                  <span className="text-xs text-muted-foreground">
                    Executed by {selectedResult.user_name}
                    {selectedResult.organization_name ? ` (${selectedResult.organization_name})` : ""}
                  </span>
                </div>
                <DialogTitle className="text-lg font-semibold mt-1">{selectedResult.source_name}</DialogTitle>
                <DialogDescription className="text-xs">
                  Created at {selectedResult.created_at ? new Date(selectedResult.created_at).toLocaleString() : ""}
                </DialogDescription>
              </DialogHeader>

              {/* Detail Tabs */}
              <div className="flex items-center border-b border-border gap-2 text-xs">
                <button
                  onClick={() => setActiveTab("markdown")}
                  className={`flex items-center gap-1.5 border-b-2 px-3 py-2 font-medium transition-colors ${
                    activeTab === "markdown"
                      ? "border-foreground text-foreground"
                      : "border-transparent text-muted-foreground hover:text-foreground"
                  }`}
                >
                  <FileText className="h-3.5 w-3.5" />
                  Formatted Output (Markdown)
                </button>
                {selectedResult.output_json && (
                  <button
                    onClick={() => setActiveTab("json")}
                    className={`flex items-center gap-1.5 border-b-2 px-3 py-2 font-medium transition-colors ${
                      activeTab === "json"
                        ? "border-foreground text-foreground"
                        : "border-transparent text-muted-foreground hover:text-foreground"
                    }`}
                  >
                    <Code className="h-3.5 w-3.5" />
                    Structured JSON
                  </button>
                )}
                {selectedResult.input_summary && (
                  <button
                    onClick={() => setActiveTab("inputs")}
                    className={`flex items-center gap-1.5 border-b-2 px-3 py-2 font-medium transition-colors ${
                      activeTab === "inputs"
                        ? "border-foreground text-foreground"
                        : "border-transparent text-muted-foreground hover:text-foreground"
                    }`}
                  >
                    <FormInput className="h-3.5 w-3.5" />
                    Input Summary
                  </button>
                )}
              </div>

              {/* Tab Content Body */}
              <div className="flex-1 overflow-y-auto p-4 bg-muted/30 rounded-lg my-2 border border-border">
                {activeTab === "markdown" && (
                  <div className="prose prose-sm dark:prose-invert max-w-none text-xs leading-relaxed text-foreground whitespace-pre-wrap">
                    <ReactMarkdown>{selectedResult.output_text}</ReactMarkdown>
                  </div>
                )}

                {activeTab === "json" && selectedResult.output_json && (
                  <pre className="font-mono text-xs overflow-x-auto text-foreground p-2">
                    {JSON.stringify(selectedResult.output_json, null, 2)}
                  </pre>
                )}

                {activeTab === "inputs" && selectedResult.input_summary && (
                  <div className="space-y-2 text-xs">
                    {Object.entries(selectedResult.input_summary).map(([k, v]) => (
                      <div key={k} className="flex flex-col gap-1 rounded bg-background p-2.5 border border-border">
                        <span className="font-semibold text-muted-foreground font-mono">{k}:</span>
                        <span className="text-foreground font-mono whitespace-pre-wrap">
                          {typeof v === "object" ? JSON.stringify(v, null, 2) : String(v)}
                        </span>
                      </div>
                    ))}
                  </div>
                )}
              </div>

              <DialogFooter className="flex items-center justify-between sm:justify-between">
                <Button
                  size="sm"
                  variant="ghost"
                  className="text-destructive hover:bg-destructive/10"
                  onClick={() => handleDelete(selectedResult.id)}
                >
                  <Trash2 className="h-3.5 w-3.5 mr-1" />
                  Delete Result
                </Button>

                <div className="flex items-center gap-2">
                  <Button size="sm" variant="secondary" onClick={() => handleCopy(selectedResult.output_text)}>
                    {copied ? <Check className="h-3.5 w-3.5 mr-1" /> : <Copy className="h-3.5 w-3.5 mr-1" />}
                    {copied ? "Copied" : "Copy Output"}
                  </Button>
                </div>
              </DialogFooter>
            </>
          )}
        </DialogContent>
      </Dialog>
    </AppShell>
  );
}
