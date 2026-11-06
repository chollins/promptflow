import { Link } from "react-router-dom";
import { useEffect, useMemo, useState } from "react";
import {
  Play,
  Plus,
  PencilLine,
  Trash2,
  Download,
  Upload,
  Sparkles,
  FormInput,
  Search,
  X,
  FileText,
  Clock,
  Layers,
  CheckCircle2,
} from "lucide-react";
import { AppShell, PageHeader } from "@/components/app-shell";
import { Button, Card, Input } from "@/components/ui-kit";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
} from "@/components/ui/dialog";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import { FormBuilder } from "@/components/FormBuilder";
import { PersonalFormRunnerModal } from "@/components/PersonalFormRunnerModal";
import { useCurrentUserId } from "@/lib/use-current-user-id";
import {
  getPersonalForms,
  savePersonalForm,
  deletePersonalForm,
  exportPersonalForms,
  importPersonalForms,
  type PersonalForm,
} from "@/lib/personal-forms";
import { apiGet } from "@/lib/api";
import { logActivity } from "@/lib/activity";
import { toast } from "sonner";

type OrgFormItem = {
  id: string;
  name: string;
  description?: string | null;
  version: string;
  fields: Array<{ id: string; type: string }>;
};

const DEFAULT_PERSONAL_FORM_JSON = {
  id: "my_form",
  name: "New Prompt Form",
  description: "A customized prompt form",
  version: "1.0",
  fields: [
    {
      id: "input_text",
      type: "textarea",
      label: "Input Content",
      description: "Enter your input content here",
      placeholder: "Paste or type content here...",
      required: true,
    },
  ],
  prompt: {
    system: "You are an expert AI assistant that processes user inputs with high quality and accuracy.",
    user: "Please process the following input:\n\n{{input_text}}",
  },
  model: {
    provider: "openai",
    name: "gpt-4o-mini",
    temperature: 0.7,
  },
};

export default function FormsPage() {
  const userId = useCurrentUserId();

  // Tab State
  const [activeTab, setActiveTab] = useState<"org" | "personal">("personal");

  // Organization Forms State
  const [orgForms, setOrgForms] = useState<OrgFormItem[]>([]);
  const [loadingOrg, setLoadingOrg] = useState(true);
  const [orgSearch, setOrgSearch] = useState("");

  // Personal Forms State
  const [personalForms, setPersonalForms] = useState<PersonalForm[]>([]);
  const [personalSearch, setPersonalSearch] = useState("");

  // Create / Edit Modal State
  const [editorOpen, setEditorOpen] = useState(false);
  const [editingFormId, setEditingFormId] = useState<string | null>(null);
  const [formName, setFormName] = useState("");
  const [formDescription, setFormDescription] = useState("");
  const [contentJson, setContentJson] = useState("");

  // Runner Modal State
  const [runnerOpen, setRunnerOpen] = useState(false);
  const [runningForm, setRunningForm] = useState<PersonalForm | null>(null);

  // Delete Confirmation State
  const [deleteConfirmId, setDeleteConfirmId] = useState<string | null>(null);

  // Load Organization Forms from backend
  useEffect(() => {
    let active = true;
    setLoadingOrg(true);
    apiGet<{ items: OrgFormItem[] }>("/forms")
      .then((data) => {
        if (active) setOrgForms(data.items || []);
      })
      .catch(() => {
        if (active) setOrgForms([]);
      })
      .finally(() => {
        if (active) setLoadingOrg(false);
      });

    return () => {
      active = false;
    };
  }, []);

  // Load Personal Forms from localStorage
  const refreshPersonalForms = () => {
    const list = getPersonalForms(userId);
    setPersonalForms(list);
  };

  useEffect(() => {
    refreshPersonalForms();
  }, [userId]);

  // Filtered Organization Forms
  const filteredOrgForms = useMemo(() => {
    const q = orgSearch.toLowerCase().trim();
    if (!q) return orgForms;
    return orgForms.filter(
      (f) =>
        f.name.toLowerCase().includes(q) ||
        (f.description && f.description.toLowerCase().includes(q))
    );
  }, [orgForms, orgSearch]);

  // Filtered Personal Forms
  const filteredPersonalForms = useMemo(() => {
    const q = personalSearch.toLowerCase().trim();
    if (!q) return personalForms;
    return personalForms.filter(
      (f) =>
        f.name.toLowerCase().includes(q) ||
        (f.description && f.description.toLowerCase().includes(q))
    );
  }, [personalForms, personalSearch]);

  // Helper to extract field count from content_json safely
  const getFieldCount = (form: PersonalForm): number => {
    try {
      const parsed = JSON.parse(form.content_json);
      return Array.isArray(parsed.fields) ? parsed.fields.length : 0;
    } catch {
      return 0;
    }
  };

  // Open Create Personal Form Modal
  const handleOpenCreateModal = () => {
    setEditingFormId(null);
    setFormName("New Prompt Form");
    setFormDescription("Custom prompt form saved in your browser storage");
    setContentJson(JSON.stringify(DEFAULT_PERSONAL_FORM_JSON, null, 2));
    setEditorOpen(true);
  };

  // Open Edit Personal Form Modal
  const handleOpenEditModal = (form: PersonalForm) => {
    setEditingFormId(form.id);
    setFormName(form.name);
    setFormDescription(form.description || "");
    setContentJson(form.content_json);
    setEditorOpen(true);
  };

  // Save Personal Form
  const handleSavePersonalForm = () => {
    if (!formName.trim()) {
      toast.error("Form name cannot be empty");
      return;
    }

    try {
      // Validate that contentJson is valid JSON
      const parsed = JSON.parse(contentJson);
      parsed.name = formName;
      if (formDescription) {
        parsed.description = formDescription;
      }
      const updatedContentJson = JSON.stringify(parsed, null, 2);

      const targetId =
        editingFormId ||
        `pf_local_${Date.now()}_${Math.random().toString(36).substring(2, 6)}`;

      const personalItem: PersonalForm = {
        id: targetId,
        name: formName,
        description: formDescription,
        updated_at: new Date().toISOString(),
        content_json: updatedContentJson,
      };

      savePersonalForm(userId, personalItem);
      refreshPersonalForms();
      setEditorOpen(false);
      toast.success(
        editingFormId
          ? "Personal form updated successfully!"
          : "Personal form created successfully!"
      );
      logActivity(
        editingFormId ? "updated personal form" : "created personal form",
        formName
      );
    } catch (err) {
      toast.error(
        err instanceof Error ? err.message : "Invalid JSON in form configuration"
      );
    }
  };

  // Delete Personal Form
  const handleDeletePersonalForm = () => {
    if (!deleteConfirmId) return;
    deletePersonalForm(userId, deleteConfirmId);
    refreshPersonalForms();
    setDeleteConfirmId(null);
    toast.success("Personal form deleted");
    logActivity("deleted personal form", deleteConfirmId);
  };

  // Single Form Export
  const handleExportSingleForm = (form: PersonalForm) => {
    const blob = new Blob([form.content_json || JSON.stringify(form, null, 2)], {
      type: "application/json",
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    const safeName = form.name.toLowerCase().replace(/[^a-z0-9_-]/g, "_") || "personal_form";
    a.download = `${safeName}.json`;
    a.click();
    URL.revokeObjectURL(url);
    toast.success(`Exported "${form.name}"`);
  };

  // Import Personal Forms
  const handleImportFile = async (file: File | null) => {
    if (!file) return;
    try {
      const text = await file.text();
      const count = importPersonalForms(userId, text);
      refreshPersonalForms();
      toast.success(`Successfully imported ${count} personal form(s)!`);
      logActivity("imported personal forms", `${count} forms`);
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Failed to import JSON");
    }
  };

  return (
    <AppShell>
      <div className="space-y-6">
        <PageHeader
          title="Prompt Forms"
          description="Build, generate, and run reusable prompt forms. Create your own personal forms or access assigned organization forms."
        />

        {/* Primary Tabs */}
        <Tabs
          value={activeTab}
          onValueChange={(val) => setActiveTab(val as "org" | "personal")}
          className="space-y-4"
        >
          <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3 border-b border-border pb-3">
            <TabsList className="grid grid-cols-2 w-full sm:w-[360px]">
              <TabsTrigger value="personal" className="gap-2">
                <Sparkles className="h-4 w-4 text-primary" />
                <span>My Forms</span>
                <span className="ml-1 rounded-full bg-muted px-2 py-0.5 text-xs text-muted-foreground font-semibold">
                  {personalForms.length}
                </span>
              </TabsTrigger>
              <TabsTrigger value="org" className="gap-2">
                <Layers className="h-4 w-4 text-muted-foreground" />
                <span>Organization Forms</span>
                <span className="ml-1 rounded-full bg-muted px-2 py-0.5 text-xs text-muted-foreground font-semibold">
                  {orgForms.length}
                </span>
              </TabsTrigger>
            </TabsList>

            {/* Top Action Bar (Context-dependent) */}
            {activeTab === "personal" && (
              <div className="flex items-center gap-2">
                <label className="inline-flex cursor-pointer items-center gap-1.5 rounded-md border border-border bg-background px-3 py-1.5 text-xs font-medium text-foreground transition-colors hover:bg-muted">
                  <Upload className="h-3.5 w-3.5" />
                  Import JSON
                  <input
                    type="file"
                    accept=".json,application/json"
                    className="hidden"
                    onChange={(e) => {
                      const file = e.target.files?.[0] ?? null;
                      void handleImportFile(file);
                      e.target.value = "";
                    }}
                  />
                </label>

                {personalForms.length > 0 && (
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => exportPersonalForms(userId)}
                    className="gap-1.5 text-xs h-8"
                  >
                    <Download className="h-3.5 w-3.5" />
                    Export All
                  </Button>
                )}

                <Button
                  onClick={handleOpenCreateModal}
                  className="gap-1.5 text-xs h-8 bg-foreground text-background hover:bg-foreground/90 font-medium shadow-xs"
                >
                  <Plus className="h-3.5 w-3.5" />
                  New Personal Form
                </Button>
              </div>
            )}
          </div>

          {/* TAB 1: PERSONAL FORMS */}
          <TabsContent value="personal" className="space-y-4">
            {/* Search Filter */}
            {personalForms.length > 0 && (
              <div className="relative max-w-md">
                <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                <Input
                  value={personalSearch}
                  onChange={(e) => setPersonalSearch(e.target.value)}
                  placeholder="Search my forms by name or description..."
                  className="pl-9 pr-8 h-9 text-xs"
                />
                {personalSearch && (
                  <button
                    onClick={() => setPersonalSearch("")}
                    className="absolute right-2.5 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground"
                  >
                    <X className="h-3.5 w-3.5" />
                  </button>
                )}
              </div>
            )}

            {/* Empty State */}
            {personalForms.length === 0 ? (
              <Card className="p-8 text-center border-dashed border-2 border-border/80 bg-muted/20">
                <div className="mx-auto flex h-12 w-12 items-center justify-center rounded-xl bg-primary/10 text-primary mb-3">
                  <Sparkles className="h-6 w-6" />
                </div>
                <h3 className="text-base font-semibold text-foreground">
                  No Personal Forms Yet
                </h3>
                <p className="mt-1.5 text-xs text-muted-foreground max-w-md mx-auto">
                  Create your own custom prompt forms or generate one automatically using
                  AI. Personal forms are saved privately in your browser storage and can be
                  executed anytime.
                </p>
                <div className="mt-5 flex items-center justify-center gap-3">
                  <Button
                    onClick={handleOpenCreateModal}
                    className="gap-1.5 text-xs bg-foreground text-background hover:bg-foreground/90 font-medium"
                  >
                    <Plus className="h-3.5 w-3.5" />
                    Create Your First Form
                  </Button>
                  <label className="inline-flex cursor-pointer items-center gap-1.5 rounded-md border border-border bg-background px-3 py-1.5 text-xs font-medium text-foreground transition-colors hover:bg-muted">
                    <Upload className="h-3.5 w-3.5" />
                    Import from JSON
                    <input
                      type="file"
                      accept=".json,application/json"
                      className="hidden"
                      onChange={(e) => {
                        const file = e.target.files?.[0] ?? null;
                        void handleImportFile(file);
                        e.target.value = "";
                      }}
                    />
                  </label>
                </div>
              </Card>
            ) : filteredPersonalForms.length === 0 ? (
              <Card className="p-8 text-center text-muted-foreground">
                <Search className="h-8 w-8 mx-auto mb-2 opacity-50" />
                <p className="text-sm font-medium">No personal forms match your search</p>
                <p className="text-xs mt-1">Try searching for a different keyword.</p>
              </Card>
            ) : (
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {filteredPersonalForms.map((form) => {
                  const fieldCount = getFieldCount(form);
                  const updatedDate = form.updated_at
                    ? new Date(form.updated_at).toLocaleDateString(undefined, {
                      month: "short",
                      day: "numeric",
                      year: "numeric",
                    })
                    : "Recently";

                  return (
                    <Card
                      key={form.id}
                      className="p-5 flex flex-col justify-between hover:border-primary/40 transition-all shadow-xs group"
                    >
                      <div className="space-y-2.5">
                        <div className="flex items-start justify-between gap-2">
                          <div className="font-semibold text-foreground text-sm line-clamp-1 group-hover:text-primary transition-colors">
                            {form.name}
                          </div>
                          <span className="text-[10px] px-2 py-0.5 rounded-full font-medium bg-primary/10 text-primary border border-primary/20 shrink-0">
                            Personal
                          </span>
                        </div>

                        <p className="text-xs text-muted-foreground line-clamp-2 min-h-[32px]">
                          {form.description || "No description provided."}
                        </p>

                        <div className="flex items-center gap-3 text-[11px] text-muted-foreground pt-1 border-t border-border/60">
                          <span className="inline-flex items-center gap-1">
                            <FormInput className="h-3 w-3" />
                            {fieldCount} {fieldCount === 1 ? "field" : "fields"}
                          </span>
                          <span>·</span>
                          <span className="inline-flex items-center gap-1">
                            <Clock className="h-3 w-3" />
                            {updatedDate}
                          </span>
                        </div>
                      </div>

                      {/* Card Action Buttons */}
                      <div className="flex items-center justify-between gap-2 pt-4 mt-2 border-t border-border">
                        <Button
                          size="sm"
                          onClick={() => {
                            setRunningForm(form);
                            setRunnerOpen(true);
                          }}
                          className="gap-1.5 h-8 text-xs bg-primary hover:bg-primary/90 text-primary-foreground font-medium flex-1 shadow-xs"
                          title="Execute personal form"
                        >
                          <Play className="h-3.5 w-3.5 fill-current" />
                          <span>Run</span>
                        </Button>

                        <div className="flex items-center gap-1">
                          <Button
                            variant="outline"
                            size="sm"
                            onClick={() => handleOpenEditModal(form)}
                            className="h-8 w-8 p-0 text-muted-foreground hover:text-foreground"
                            title="Edit form definition"
                          >
                            <PencilLine className="h-3.5 w-3.5" />
                          </Button>

                          <Button
                            variant="outline"
                            size="sm"
                            onClick={() => handleExportSingleForm(form)}
                            className="h-8 w-8 p-0 text-muted-foreground hover:text-foreground"
                            title="Export as JSON"
                          >
                            <Download className="h-3.5 w-3.5" />
                          </Button>

                          <Button
                            variant="outline"
                            size="sm"
                            onClick={() => setDeleteConfirmId(form.id)}
                            className="h-8 w-8 p-0 text-muted-foreground hover:text-destructive hover:bg-destructive/10"
                            title="Delete personal form"
                          >
                            <Trash2 className="h-3.5 w-3.5" />
                          </Button>
                        </div>
                      </div>
                    </Card>
                  );
                })}
              </div>
            )}
          </TabsContent>

          {/* TAB 2: ORGANIZATION FORMS */}
          <TabsContent value="org" className="space-y-4">
            {/* Search Filter */}
            {orgForms.length > 0 && (
              <div className="relative max-w-md">
                <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                <Input
                  value={orgSearch}
                  onChange={(e) => setOrgSearch(e.target.value)}
                  placeholder="Search organization forms..."
                  className="pl-9 pr-8 h-9 text-xs"
                />
                {orgSearch && (
                  <button
                    onClick={() => setOrgSearch("")}
                    className="absolute right-2.5 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground"
                  >
                    <X className="h-3.5 w-3.5" />
                  </button>
                )}
              </div>
            )}

            {loadingOrg ? (
              <div className="p-12 text-center text-muted-foreground flex flex-col items-center gap-3">
                <span className="h-6 w-6 animate-spin rounded-full border-2 border-foreground border-t-transparent" />
                <span className="text-xs">Loading organization forms...</span>
              </div>
            ) : filteredOrgForms.length === 0 ? (
              <Card className="p-8 text-center border-border bg-card">
                <div className="font-medium text-sm">No organization forms found</div>
                <p className="mt-1 text-xs text-muted-foreground">
                  Forms assigned to your organization flows will appear here.
                </p>
              </Card>
            ) : (
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {filteredOrgForms.map((form) => (
                  <Card
                    key={form.id}
                    className="p-5 flex flex-col justify-between hover:border-primary/40 transition-all shadow-xs group"
                  >
                    <div className="space-y-2">
                      <div className="flex items-start justify-between gap-2">
                        <div className="font-semibold text-foreground text-sm line-clamp-1 group-hover:text-primary transition-colors">
                          {form.name}
                        </div>
                        <span className="text-[10px] px-2 py-0.5 rounded-full font-medium bg-muted text-muted-foreground border border-border shrink-0">
                          Org
                        </span>
                      </div>

                      <p className="text-xs text-muted-foreground line-clamp-2 min-h-[32px]">
                        {form.description || "No description provided."}
                      </p>

                      <div className="flex items-center gap-2 text-[11px] text-muted-foreground pt-1 border-t border-border/60">
                        <span>
                          {form.fields.length} {form.fields.length === 1 ? "field" : "fields"}
                        </span>
                        <span>·</span>
                        <span>v{form.version}</span>
                      </div>
                    </div>

                    <div className="pt-4 mt-2 border-t border-border">
                      <Link
                        to={`/forms/${form.id}`}
                        className="inline-flex w-full h-8 items-center justify-center rounded-md bg-primary hover:bg-primary/90 px-3 text-xs font-medium text-primary-foreground transition-colors gap-1.5 shadow-xs"
                      >
                        <Play className="h-3.5 w-3.5 fill-current" />
                        Run Form
                      </Link>
                    </div>
                  </Card>
                ))}
              </div>
            )}
          </TabsContent>
        </Tabs>
      </div>

      {/* CREATE / EDIT PERSONAL FORM MODAL */}
      <Dialog open={editorOpen} onOpenChange={setEditorOpen}>
        <DialogContent className="w-[95vw] max-w-6xl max-h-[92vh] flex flex-col">
          <DialogHeader>
            <div className="flex items-center gap-2">
              <span className="text-[11px] px-2 py-0.5 rounded-full font-medium bg-primary/10 text-primary border border-primary/20">
                Personal Form
              </span>
              <DialogTitle>
                {editingFormId ? "Edit Personal Form" : "Create Personal Form"}
              </DialogTitle>
            </div>
            <DialogDescription className="text-xs">
              {editingFormId
                ? "Update your personal prompt form definition. Changes will be saved to your browser storage."
                : "Build your personal prompt form manually or use the AI Generator above to suggest one based on your prompt."}
            </DialogDescription>
          </DialogHeader>

          <div className="grid gap-4 mt-2 overflow-y-auto pr-1 flex-1">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
              <div className="space-y-1.5">
                <label className="text-xs font-medium text-foreground/80">Form Name</label>
                <Input
                  value={formName}
                  onChange={(e) => setFormName(e.target.value)}
                  placeholder="e.g. Sales Email Generator"
                />
              </div>

              <div className="space-y-1.5">
                <label className="text-xs font-medium text-foreground/80">Description</label>
                <Input
                  value={formDescription}
                  onChange={(e) => setFormDescription(e.target.value)}
                  placeholder="Optional summary of this personal form"
                />
              </div>
            </div>

            {/* Embedded FormBuilder with AI Prompt Generator Banner */}
            <FormBuilder
              contentJson={contentJson}
              onChangeContentJson={setContentJson}
              formName={formName}
              onFormNameChange={setFormName}
              formDescription={formDescription}
              onFormDescriptionChange={setFormDescription}
            />
          </div>

          <div className="flex items-center justify-between pt-4 border-t border-border mt-2">
            <span className="text-xs text-muted-foreground">
              Stored locally on this device under your account.
            </span>
            <div className="flex gap-2">
              <Button
                variant="secondary"
                onClick={() => setEditorOpen(false)}
                className="h-8 text-xs"
              >
                Cancel
              </Button>
              <Button
                onClick={handleSavePersonalForm}
                className="h-8 text-xs bg-foreground text-background hover:bg-foreground/90 font-medium"
              >
                {editingFormId ? "Update Form" : "Save Form"}
              </Button>
            </div>
          </div>
        </DialogContent>
      </Dialog>

      {/* RUNNER MODAL FOR PERSONAL FORM */}
      <PersonalFormRunnerModal
        personalForm={runningForm}
        open={runnerOpen}
        onOpenChange={setRunnerOpen}
      />

      {/* DELETE CONFIRMATION DIALOG */}
      <AlertDialog
        open={!!deleteConfirmId}
        onOpenChange={(open) => !open && setDeleteConfirmId(null)}
      >
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Delete this personal form?</AlertDialogTitle>
            <AlertDialogDescription>
              This will remove the form definition from your browser's local storage. This
              action cannot be undone. (You can export it first if you want a backup.)
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction
              onClick={handleDeletePersonalForm}
              className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
            >
              Delete Form
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </AppShell>
  );
}
