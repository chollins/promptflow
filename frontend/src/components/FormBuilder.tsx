import React, { useEffect, useMemo, useState } from "react";
import {
  Plus,
  Trash2,
  ArrowUp,
  ArrowDown,
  Sparkles,
  Settings2,
  Code,
  Layout,
  Tag,
  Copy,
  ChevronDown,
  ChevronUp,
  Loader2,
} from "lucide-react";
import { Button, Card, Input } from "@/components/ui-kit";
import { Textarea } from "@/components/ui/textarea";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { apiPost } from "@/lib/api";
import { toast } from "sonner";

export type BuilderField = {
  id: string;
  label: string;
  description?: string;
  placeholder?: string;
  type: "text" | "textarea" | "dropdown" | "radio" | "checkbox" | "multiselect" | "date" | "hidden";
  required?: boolean;
  default?: string;
  options?: string[];
  data_source?: {
    type: string;
    step_id?: string;
    path?: string;
  };
};

export type BuilderFormData = {
  id: string;
  name: string;
  description?: string;
  version: string;
  fields: BuilderField[];
  prompt: {
    system: string;
    user: string;
  };
  model: {
    provider: string;
    name: string;
    temperature: number;
  };
  output?: {
    type: string;
    schema?: any;
  } | null;
  execution?: {
    mode: string;
  } | null;
};

const DEFAULT_FORM_DATA: BuilderFormData = {
  id: "",
  name: "",
  description: "",
  version: "1.0",
  fields: [],
  prompt: {
    system: "You are a helpful AI assistant.",
    user: "",
  },
  model: {
    provider: "openai",
    name: "gpt-4o-mini",
    temperature: 0.7,
  },
  output: null,
  execution: { mode: "interactive" },
};

function parseJsonSafe(raw: string): BuilderFormData | null {
  try {
    const parsed = JSON.parse(raw);
    if (parsed && typeof parsed === "object") {
      return {
        id: parsed.id ?? "",
        name: parsed.name ?? "",
        description: parsed.description ?? "",
        version: parsed.version ?? "1.0",
        fields: Array.isArray(parsed.fields)
          ? parsed.fields.map((f: any) => ({
            id: f.id ?? "",
            label: f.label ?? "",
            description: f.description ?? "",
            placeholder: f.placeholder ?? f.description ?? "",
            type: f.type ?? "text",
            required: !!f.required,
            default: f.default ?? "",
            options: Array.isArray(f.options)
              ? f.options.map((opt: any) =>
                typeof opt === "string" ? opt : opt.label ?? opt.value ?? ""
              )
              : [],
            data_source: f.data_source ?? undefined,
          }))
          : [],
        prompt: {
          system: parsed.prompt?.system ?? "You are a helpful AI assistant.",
          user: parsed.prompt?.user ?? "",
        },
        model: {
          provider: parsed.model?.provider ?? "openai",
          name: parsed.model?.name ?? "gpt-4o-mini",
          temperature: typeof parsed.model?.temperature === "number" ? parsed.model.temperature : 0.7,
        },
        output: parsed.output ?? null,
        execution: parsed.execution ?? { mode: "interactive" },
      };
    }
  } catch {
    // Ignore JSON parse errors while typing
  }
  return null;
}

interface FormBuilderProps {
  contentJson: string;
  onChangeContentJson: (json: string) => void;
  formName?: string;
  onFormNameChange?: (name: string) => void;
  formDescription?: string;
  onFormDescriptionChange?: (desc: string) => void;
}

export function FormBuilder({
  contentJson,
  onChangeContentJson,
  formName,
  onFormNameChange,
  formDescription,
  onFormDescriptionChange,
}: FormBuilderProps) {
  const [activeTab, setActiveTab] = useState<"visual" | "json">("visual");
  const [visualSubTab, setVisualSubTab] = useState<"fields" | "prompt" | "model" | "output">("fields");
  const [collapsedFieldIds, setCollapsedFieldIds] = useState<Record<string, boolean>>({});
  const [jsonError, setJsonError] = useState<string | null>(null);
  const [aiPrompt, setAiPrompt] = useState("");
  const [generating, setGenerating] = useState(false);

  async function handleGenerateWithAi(customPrompt?: string) {
    const promptToUse = (customPrompt || aiPrompt).trim();
    if (!promptToUse) {
      toast.error("Please enter a description of the form to generate.");
      return;
    }
    setGenerating(true);
    try {
      const res = await apiPost<{ item: BuilderFormData }>("/forms/generate", {
        prompt: promptToUse,
      });
      if (res?.item) {
        const generated = res.item;
        setFormData(generated);
        const json = JSON.stringify(generated, null, 2);
        onChangeContentJson(json);
        if (onFormNameChange && generated.name) {
          onFormNameChange(generated.name);
        }
        if (onFormDescriptionChange && generated.description) {
          onFormDescriptionChange(generated.description);
        }
        setCollapsedFieldIds({});
        setVisualSubTab("fields");
        toast.success(`Generated "${generated.name}" with ${generated.fields?.length || 0} fields!`);
      }
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Failed to generate form with AI");
    } finally {
      setGenerating(false);
    }
  }

  function isFieldExpanded(fieldId: string) {
    return collapsedFieldIds[fieldId] !== true;
  }

  function toggleFieldExpanded(fieldId: string) {
    setCollapsedFieldIds((prev) => ({
      ...prev,
      [fieldId]: !prev[fieldId],
    }));
  }

  function expandAllFields() {
    setCollapsedFieldIds({});
  }

  function collapseAllFields() {
    const next: Record<string, boolean> = {};
    formData.fields.forEach((f) => {
      next[f.id] = true;
    });
    setCollapsedFieldIds(next);
  }

  // Local state parsed from JSON
  const [formData, setFormData] = useState<BuilderFormData>(() => {
    return parseJsonSafe(contentJson) ?? {
      ...DEFAULT_FORM_DATA,
      name: formName || "",
      description: formDescription || "",
    };
  });

  // Keep outer name/description aligned with formData
  useEffect(() => {
    if (formName !== undefined && formName !== formData.name) {
      setFormData((prev) => {
        const prevExpectedSlug = (prev.name || "")
          .toLowerCase()
          .replace(/[^a-z0-9]+/g, "-")
          .replace(/^-|-$/g, "");
        const newSlug = formName
          .toLowerCase()
          .replace(/[^a-z0-9]+/g, "-")
          .replace(/^-|-$/g, "");
        const shouldUpdateId =
          !prev.id || prev.id === prevExpectedSlug || prev.id.length <= 1;
        const next = {
          ...prev,
          name: formName,
          id: shouldUpdateId ? newSlug : prev.id,
        };
        const json = JSON.stringify(next, null, 2);
        onChangeContentJson(json);
        return next;
      });
    }
  }, [formName]);

  useEffect(() => {
    if (formDescription !== undefined && formDescription !== formData.description) {
      setFormData((prev) => ({ ...prev, description: formDescription }));
    }
  }, [formDescription]);

  // Synchronize formData updates to JSON string
  function updateFormData(updater: (prev: BuilderFormData) => BuilderFormData) {
    setFormData((prev) => {
      const next = updater(prev);
      const json = JSON.stringify(next, null, 2);
      onChangeContentJson(json);
      if (onFormNameChange && next.name !== formName) {
        onFormNameChange(next.name);
      }
      if (onFormDescriptionChange && next.description !== formDescription) {
        onFormDescriptionChange(next.description || "");
      }
      return next;
    });
  }

  // When raw JSON changes externally (e.g. from file upload)
  useEffect(() => {
    const parsed = parseJsonSafe(contentJson);
    if (parsed) {
      setFormData(parsed);
      setJsonError(null);
    } else {
      setJsonError("Invalid JSON");
    }
  }, [contentJson]);

  // Field Management Helpers
  function handleAddField() {
    const newIndex = formData.fields.length + 1;
    const newField: BuilderField = {
      id: `field_${newIndex}`,
      label: `Field ${newIndex}`,
      type: "text",
      description: "",
      placeholder: "",
      required: false,
      default: "",
      options: [],
    };
    setCollapsedFieldIds((prev) => ({
      ...prev,
      [newField.id]: false,
    }));
    updateFormData((prev) => ({
      ...prev,
      fields: [...prev.fields, newField],
    }));
  }

  function handleUpdateField(index: number, patch: Partial<BuilderField>) {
    updateFormData((prev) => {
      const fields = [...prev.fields];
      const current = fields[index];
      const updated = { ...current, ...patch };

      // If user typed description and not placeholder, or vice versa, sync them
      if (patch.description !== undefined && patch.placeholder === undefined) {
        updated.placeholder = patch.description;
      } else if (patch.placeholder !== undefined && patch.description === undefined) {
        updated.description = patch.placeholder;
      }

      fields[index] = updated;
      return { ...prev, fields };
    });
  }

  function handleRemoveField(index: number) {
    updateFormData((prev) => {
      const fields = prev.fields.filter((_, i) => i !== index);
      return { ...prev, fields };
    });
  }

  function handleMoveField(index: number, direction: "up" | "down") {
    updateFormData((prev) => {
      const fields = [...prev.fields];
      const targetIndex = direction === "up" ? index - 1 : index + 1;
      if (targetIndex < 0 || targetIndex >= fields.length) return prev;
      const temp = fields[index];
      fields[index] = fields[targetIndex];
      fields[targetIndex] = temp;
      return { ...prev, fields };
    });
  }

  // Insert {{field_id}} into user prompt
  function insertPromptVariable(fieldId: string) {
    const placeholder = `{{${fieldId}}}`;
    updateFormData((prev) => ({
      ...prev,
      prompt: {
        ...prev.prompt,
        user: prev.prompt.user ? `${prev.prompt.user} ${placeholder}` : placeholder,
      },
    }));
  }

  return (
    <div className="space-y-4">
      {/* Mode Selector: Visual Builder vs Raw JSON */}
      <div className="flex items-center justify-between border-b border-border pb-2">
        <div className="flex items-center gap-2">
          <Button
            type="button"
            size="sm"
            variant={activeTab === "visual" ? "default" : "outline"}
            onClick={() => setActiveTab("visual")}
            className="gap-1.5 h-8 text-xs font-medium"
          >
            <Layout className="h-3.5 w-3.5" /> Visual Builder
          </Button>
          <Button
            type="button"
            size="sm"
            variant={activeTab === "json" ? "default" : "outline"}
            onClick={() => setActiveTab("json")}
            className="gap-1.5 h-8 text-xs font-medium"
          >
            <Code className="h-3.5 w-3.5" /> Raw JSON
          </Button>
        </div>

        {activeTab === "visual" && (
          <span className="text-xs text-muted-foreground">
            {formData.fields.length} field{formData.fields.length === 1 ? "" : "s"} defined
          </span>
        )}
      </div>

      {activeTab === "json" ? (
        <div className="space-y-2">
          <div className="flex items-center justify-between text-xs text-muted-foreground">
            <span>Direct JSON Editor</span>
            {jsonError && <span className="text-destructive font-medium">{jsonError}</span>}
          </div>
          <Textarea
            value={contentJson}
            onChange={(e) => onChangeContentJson(e.target.value)}
            rows={16}
            className="font-mono text-xs"
            placeholder="Paste or edit form JSON here..."
          />
        </div>
      ) : (
        <div className="space-y-4">
          {/* AI Form Generator Banner */}
          <div className="rounded-xl border border-primary/25 bg-gradient-to-r from-primary/10 via-primary/5 to-transparent p-3.5 space-y-2.5 shadow-xs">
            <div className="flex items-center gap-2">
              <div className="flex h-7 w-7 items-center justify-center rounded-lg bg-primary/15 text-primary shrink-0">
                <Sparkles className="h-4 w-4" />
              </div>
              <div className="min-w-0 flex-1">
                <div className="text-xs font-semibold text-foreground flex items-center gap-1.5">
                  Generate Form with AI
                  <span className="text-[10px] px-1.5 py-0.2 rounded bg-primary/15 text-primary font-medium">
                    Beta
                  </span>
                </div>
                <p className="text-[11px] text-muted-foreground truncate">
                  Describe what you want to create, and AI will suggest all input fields, choices, and prompt templates.
                </p>
              </div>
            </div>

            <div className="flex items-center gap-2">
              <Input
                value={aiPrompt}
                onChange={(e) => setAiPrompt(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !e.shiftKey) {
                    e.preventDefault();
                    void handleGenerateWithAi();
                  }
                }}
                placeholder="e.g. Customer feedback survey with rating, product category, comments, and email..."
                className="h-8 text-xs bg-background/90 flex-1"
                disabled={generating}
              />
              <Button
                type="button"
                size="sm"
                onClick={() => void handleGenerateWithAi()}
                disabled={generating || !aiPrompt.trim()}
                className="h-8 text-xs gap-1.5 shrink-0 bg-primary hover:bg-primary/90 text-primary-foreground font-medium shadow-xs px-3"
              >
                {generating ? (
                  <>
                    <Loader2 className="h-3.5 w-3.5 animate-spin" />
                    Generating...
                  </>
                ) : (
                  <>
                    <Sparkles className="h-3.5 w-3.5" />
                    Generate
                  </>
                )}
              </Button>
            </div>

            {/* Quick Example Suggestions */}
            <div className="flex flex-wrap items-center gap-1.5 pt-0.5">
              <span className="text-[10px] text-muted-foreground font-medium shrink-0">Quick Ideas:</span>
              {[
                "Blog post writer with SEO keywords and tone",
                "Customer satisfaction survey with 1-5 rating",
                "Meeting action items & key decisions extractor",
                "Bug report with severity, repro steps & logs",
              ].map((example, exIdx) => (
                <button
                  key={exIdx}
                  type="button"
                  disabled={generating}
                  onClick={() => {
                    setAiPrompt(example);
                    void handleGenerateWithAi(example);
                  }}
                  className="text-[10px] px-2 py-0.5 rounded-full border border-border/70 bg-background/70 hover:bg-primary/10 hover:text-primary hover:border-primary/40 transition-colors text-muted-foreground cursor-pointer disabled:opacity-50"
                >
                  + {example}
                </button>
              ))}
            </div>
          </div>

          {/* Sub Navigation */}
          <div className="flex items-center gap-1.5 p-1 bg-muted/40 rounded-lg border border-border/50 text-xs">
            <button
              type="button"
              onClick={() => setVisualSubTab("fields")}
              className={`flex-1 py-1.5 px-3 rounded-md font-medium transition-colors text-center ${visualSubTab === "fields"
                  ? "bg-background text-foreground shadow-xs"
                  : "text-muted-foreground hover:text-foreground"
                }`}
            >
              1. Fields & Inputs ({formData.fields.length})
            </button>
            <button
              type="button"
              onClick={() => setVisualSubTab("prompt")}
              className={`flex-1 py-1.5 px-3 rounded-md font-medium transition-colors text-center ${visualSubTab === "prompt"
                  ? "bg-background text-foreground shadow-xs"
                  : "text-muted-foreground hover:text-foreground"
                }`}
            >
              2. Prompts & Templates
            </button>
            <button
              type="button"
              onClick={() => setVisualSubTab("model")}
              className={`flex-1 py-1.5 px-3 rounded-md font-medium transition-colors text-center ${visualSubTab === "model"
                  ? "bg-background text-foreground shadow-xs"
                  : "text-muted-foreground hover:text-foreground"
                }`}
            >
              3. Model Settings
            </button>
            <button
              type="button"
              onClick={() => setVisualSubTab("output")}
              className={`flex-1 py-1.5 px-3 rounded-md font-medium transition-colors text-center ${visualSubTab === "output"
                  ? "bg-background text-foreground shadow-xs"
                  : "text-muted-foreground hover:text-foreground"
                }`}
            >
              4. Structured Output
            </button>
          </div>

          {/* Form Identifier (Slug) */}
          <div className="flex items-center gap-3 px-3 py-2 rounded-lg bg-muted/20 border border-border/50 text-xs">
            <span className="font-semibold text-foreground/80 shrink-0">Runtime Identifier (ID / Slug):</span>
            <Input
              value={formData.id}
              onChange={(e) =>
                updateFormData((prev) => ({
                  ...prev,
                  id: e.target.value.toLowerCase().replace(/[^a-z0-9_-]/g, ""),
                }))
              }
              placeholder="e.g. meeting-summary"
              className="h-7 text-xs font-mono max-w-xs"
            />
            <span className="text-[11px] text-muted-foreground hidden sm:inline">
              Used for URLs and execution routing
            </span>
          </div>

          {/* TAB 1: FIELDS COMPOSER */}
          {visualSubTab === "fields" && (
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <div>
                  <h3 className="text-sm font-semibold text-foreground">Form Inputs</h3>
                  <p className="text-xs text-muted-foreground">
                    Add the input fields users will fill out when executing this form.
                  </p>
                </div>
                <div className="flex items-center gap-1.5">
                  {formData.fields.length > 1 && (
                    <>
                      <Button
                        type="button"
                        size="sm"
                        variant="outline"
                        onClick={expandAllFields}
                        className="h-8 text-xs px-2.5 text-muted-foreground hover:text-foreground"
                      >
                        Expand All
                      </Button>
                      <Button
                        type="button"
                        size="sm"
                        variant="outline"
                        onClick={collapseAllFields}
                        className="h-8 text-xs px-2.5 text-muted-foreground hover:text-foreground"
                      >
                        Collapse All
                      </Button>
                    </>
                  )}
                  <Button
                    type="button"
                    size="sm"
                    onClick={handleAddField}
                    className="gap-1.5 text-xs h-8"
                  >
                    <Plus className="h-3.5 w-3.5" /> Add Field
                  </Button>
                </div>
              </div>

              {formData.fields.length === 0 ? (
                <div className="border border-dashed border-border rounded-lg p-8 text-center bg-muted/20">
                  <Tag className="h-8 w-8 text-muted-foreground mx-auto mb-2 opacity-50" />
                  <div className="text-sm font-medium text-foreground">No input fields added yet</div>
                  <p className="text-xs text-muted-foreground mt-1 mb-4">
                    Forms need input fields to collect parameters before running prompts.
                  </p>
                  <Button type="button" size="sm" onClick={handleAddField} className="gap-1.5">
                    <Plus className="h-3.5 w-3.5" /> Add First Field
                  </Button>
                </div>
              ) : (
                <div className="space-y-3">
                  {formData.fields.map((field, index) => {
                    const isExpanded = isFieldExpanded(field.id);
                    const hasOptions =
                      field.type === "dropdown" ||
                      field.type === "radio" ||
                      field.type === "checkbox" ||
                      field.type === "multiselect";

                    return (
                      <div
                        key={field.id || index}
                        className="border border-border rounded-lg bg-card/60 p-4 space-y-3 transition-shadow hover:shadow-xs"
                      >
                        {/* Header bar (Clickable to collapse/expand) */}
                        <div
                          className="flex items-center justify-between gap-2 border-b border-border/50 pb-2.5 cursor-pointer select-none"
                          onClick={() => toggleFieldExpanded(field.id)}
                        >
                          <div className="flex items-center gap-2 min-w-0 flex-1">
                            <span className="flex items-center justify-center h-5 w-5 rounded-full bg-primary/10 text-primary text-xs font-semibold shrink-0">
                              {index + 1}
                            </span>
                            <span className="font-semibold text-sm text-foreground truncate">
                              {field.label || field.id || "Untitled Field"}
                            </span>
                            <span className="text-[11px] px-1.5 py-0.5 rounded bg-muted text-muted-foreground font-mono shrink-0">
                              {field.type}
                            </span>
                            {field.required && (
                              <span className="text-[10px] text-red-500 font-semibold uppercase shrink-0">
                                Required
                              </span>
                            )}
                            {/* Summary preview when collapsed */}
                            {!isExpanded && (field.placeholder || field.description) && (
                              <span className="text-xs text-muted-foreground truncate hidden sm:inline ml-2 border-l border-border/60 pl-2">
                                {field.placeholder || field.description}
                              </span>
                            )}
                          </div>

                          <div className="flex items-center gap-1" onClick={(e) => e.stopPropagation()}>
                            <Button
                              type="button"
                              size="sm"
                              variant="ghost"
                              disabled={index === 0}
                              onClick={() => handleMoveField(index, "up")}
                              className="h-7 w-7 p-0"
                              title="Move Up"
                            >
                              <ArrowUp className="h-3.5 w-3.5" />
                            </Button>
                            <Button
                              type="button"
                              size="sm"
                              variant="ghost"
                              disabled={index === formData.fields.length - 1}
                              onClick={() => handleMoveField(index, "down")}
                              className="h-7 w-7 p-0"
                              title="Move Down"
                            >
                              <ArrowDown className="h-3.5 w-3.5" />
                            </Button>
                            <Button
                              type="button"
                              size="sm"
                              variant="ghost"
                              onClick={() => toggleFieldExpanded(field.id)}
                              className="h-7 w-7 p-0"
                              title={isExpanded ? "Collapse" : "Expand"}
                            >
                              {isExpanded ? (
                                <ChevronUp className="h-3.5 w-3.5" />
                              ) : (
                                <ChevronDown className="h-3.5 w-3.5" />
                              )}
                            </Button>
                            <Button
                              type="button"
                              size="sm"
                              variant="ghost"
                              onClick={() => handleRemoveField(index)}
                              className="h-7 w-7 p-0 text-red-600 hover:text-red-700 hover:bg-red-50"
                              title="Delete Field"
                            >
                              <Trash2 className="h-3.5 w-3.5" />
                            </Button>
                          </div>
                        </div>

                        {/* Collapsible details */}
                        {isExpanded && (
                          <div className="grid grid-cols-1 md:grid-cols-2 gap-3 pt-1">
                            <div className="space-y-1">
                              <label className="text-xs font-medium text-foreground/80">
                                Variable ID (Prompt placeholder)
                              </label>
                              <Input
                                value={field.id}
                                onChange={(e) =>
                                  handleUpdateField(index, {
                                    id: e.target.value
                                      .toLowerCase()
                                      .replace(/[^a-z0-9_]/g, "_"),
                                  })
                                }
                                placeholder="e.g. meeting_notes, name"
                                className="h-8 text-xs font-mono"
                              />
                            </div>

                            <div className="space-y-1">
                              <label className="text-xs font-medium text-foreground/80">
                                Display Label
                              </label>
                              <Input
                                value={field.label}
                                onChange={(e) => handleUpdateField(index, { label: e.target.value })}
                                placeholder="e.g. Meeting Notes, Full Name"
                                className="h-8 text-xs"
                              />
                            </div>

                            <div className="space-y-1">
                              <label className="text-xs font-medium text-foreground/80">
                                Input Type
                              </label>
                              <Select
                                value={field.type}
                                onValueChange={(val: any) => handleUpdateField(index, { type: val })}
                              >
                                <SelectTrigger className="h-8 text-xs">
                                  <SelectValue />
                                </SelectTrigger>
                                <SelectContent>
                                  <SelectItem value="text">Text (Single-line Input Box)</SelectItem>
                                  <SelectItem value="textarea">Textarea (Multi-line Text Box)</SelectItem>
                                  <SelectItem value="dropdown">Dropdown (Single Selection)</SelectItem>
                                  <SelectItem value="radio">Radio Buttons (Single Choice)</SelectItem>
                                  <SelectItem value="checkbox">Checkbox (Toggle or Multi-select)</SelectItem>
                                  <SelectItem value="multiselect">Multi-select Choice List</SelectItem>
                                  <SelectItem value="date">Date Picker</SelectItem>
                                  <SelectItem value="hidden">Hidden Value</SelectItem>
                                </SelectContent>
                              </Select>
                            </div>

                            <div className="space-y-1">
                              <label className="text-xs font-medium text-foreground/80">
                                Description / Placeholder
                              </label>
                              <Input
                                value={field.description || field.placeholder || ""}
                                onChange={(e) =>
                                  handleUpdateField(index, {
                                    description: e.target.value,
                                    placeholder: e.target.value,
                                  })
                                }
                                placeholder="What's the field all about?"
                                className="h-8 text-xs"
                              />
                            </div>

                            <div className="space-y-1">
                              <label className="text-xs font-medium text-foreground/80">
                                Default Value (Optional)
                              </label>
                              <Input
                                value={field.default || ""}
                                onChange={(e) => handleUpdateField(index, { default: e.target.value })}
                                placeholder="Initial default value"
                                className="h-8 text-xs"
                              />
                            </div>

                            <div className="flex items-center gap-4 pt-4">
                              <label className="flex items-center gap-2 text-xs text-foreground cursor-pointer select-none">
                                <input
                                  type="checkbox"
                                  checked={!!field.required}
                                  onChange={(e) =>
                                    handleUpdateField(index, { required: e.target.checked })
                                  }
                                  className="h-4 w-4 rounded border-border text-primary"
                                />
                                Required Field
                              </label>
                            </div>

                            {/* Options Editor for Choice Fields */}
                            {hasOptions && (
                              <div className="col-span-1 md:col-span-2 space-y-2 border-t border-border/40 pt-3 mt-1">
                                <div className="flex items-center justify-between">
                                  <label className="text-xs font-semibold text-foreground/80">
                                    Options / Choices List
                                  </label>
                                  <Button
                                    type="button"
                                    size="sm"
                                    variant="outline"
                                    onClick={() => {
                                      const currentOpts = field.options || [];
                                      handleUpdateField(index, {
                                        options: [...currentOpts, `Option ${currentOpts.length + 1}`],
                                      });
                                    }}
                                    className="h-6 text-[11px] gap-1 px-2"
                                  >
                                    <Plus className="h-3 w-3" /> Add Choice
                                  </Button>
                                </div>

                                <div className="space-y-1.5 max-h-36 overflow-y-auto pr-1">
                                  {(field.options && field.options.length > 0 ? field.options : []).map(
                                    (opt, optIdx) => (
                                      <div key={optIdx} className="flex items-center gap-2">
                                        <Input
                                          value={opt}
                                          onChange={(e) => {
                                            const nextOpts = [...(field.options || [])];
                                            nextOpts[optIdx] = e.target.value;
                                            handleUpdateField(index, { options: nextOpts });
                                          }}
                                          placeholder={`Choice ${optIdx + 1}`}
                                          className="h-7 text-xs flex-1"
                                        />
                                        <Button
                                          type="button"
                                          size="sm"
                                          variant="ghost"
                                          onClick={() => {
                                            const nextOpts = (field.options || []).filter(
                                              (_, i) => i !== optIdx
                                            );
                                            handleUpdateField(index, { options: nextOpts });
                                          }}
                                          className="h-7 w-7 p-0 text-muted-foreground hover:text-destructive"
                                        >
                                          <Trash2 className="h-3 w-3" />
                                        </Button>
                                      </div>
                                    )
                                  )}
                                  {(!field.options || field.options.length === 0) && (
                                    <div className="text-xs text-muted-foreground italic">
                                      No options defined. Click "Add Choice" to add dropdown/radio values.
                                    </div>
                                  )}
                                </div>
                              </div>
                            )}

                            {/* Data Source Configuration (For Flow Step Output Dependency) */}
                            <div className="col-span-1 md:col-span-2 border-t border-border/40 pt-2 text-xs">
                              <details className="cursor-pointer text-muted-foreground hover:text-foreground">
                                <summary className="font-medium text-xs text-foreground/80">
                                  Dynamic Step Data Source (For multi-step flows)
                                </summary>
                                <div className="grid grid-cols-2 gap-2 mt-2 pt-2 border-t border-border/30">
                                  <div className="space-y-1">
                                    <label className="text-[11px] text-foreground/70">Previous Step ID</label>
                                    <Input
                                      value={field.data_source?.step_id || ""}
                                      onChange={(e) =>
                                        handleUpdateField(index, {
                                          data_source: {
                                            type: "step_output",
                                            step_id: e.target.value,
                                            path: field.data_source?.path || "",
                                          },
                                        })
                                      }
                                      placeholder="e.g. summary"
                                      className="h-7 text-xs font-mono"
                                    />
                                  </div>
                                  <div className="space-y-1">
                                    <label className="text-[11px] text-foreground/70">Output JSON Path</label>
                                    <Input
                                      value={field.data_source?.path || ""}
                                      onChange={(e) =>
                                        handleUpdateField(index, {
                                          data_source: {
                                            type: "step_output",
                                            step_id: field.data_source?.step_id || "",
                                            path: e.target.value,
                                          },
                                        })
                                      }
                                      placeholder="e.g. actions, points"
                                      className="h-7 text-xs font-mono"
                                    />
                                  </div>
                                </div>
                              </details>
                            </div>
                          </div>
                        )}
                      </div>
                    );
                  })}
                </div>
              )}
            </div>
          )}

          {/* TAB 2: PROMPTS COMPOSER */}
          {visualSubTab === "prompt" && (
            <div className="space-y-4">
              <div>
                <h3 className="text-sm font-semibold text-foreground">Prompt Instructions</h3>
                <p className="text-xs text-muted-foreground">
                  Define the System instructions and User prompt template sent to the model.
                </p>
              </div>

              {/* Variable Insertion Pills */}
              <div className="space-y-1.5 p-3 rounded-lg bg-muted/30 border border-border/50">
                <span className="text-xs font-semibold text-foreground/80 flex items-center gap-1.5">
                  <Sparkles className="h-3.5 w-3.5 text-primary" />
                  Available Form Variables (Click to insert into User Prompt):
                </span>
                <div className="flex flex-wrap gap-1.5 pt-1">
                  {formData.fields.length === 0 ? (
                    <span className="text-xs text-muted-foreground italic">
                      No fields created yet. Create fields in tab 1 to insert their placeholders here.
                    </span>
                  ) : (
                    formData.fields.map((f) => (
                      <button
                        key={f.id}
                        type="button"
                        onClick={() => insertPromptVariable(f.id)}
                        className="inline-flex items-center gap-1 px-2 py-1 rounded bg-primary/10 hover:bg-primary/20 text-primary text-xs font-mono font-medium border border-primary/20 transition-colors"
                        title={`Click to append {{${f.id}}} to user prompt`}
                      >
                        <span>+</span> {`{{${f.id}}}`}
                      </button>
                    ))
                  )}
                </div>
              </div>

              <div className="space-y-2">
                <label className="text-xs font-semibold text-foreground/80">
                  System Prompt (AI Persona & Guardrails)
                </label>
                <Textarea
                  value={formData.prompt.system}
                  onChange={(e) =>
                    updateFormData((prev) => ({
                      ...prev,
                      prompt: { ...prev.prompt, system: e.target.value },
                    }))
                  }
                  rows={3}
                  className="text-xs"
                  placeholder="e.g. You are a helpful assistant. Summarize supplied notes accurately."
                />
              </div>

              <div className="space-y-2">
                <label className="text-xs font-semibold text-foreground/80">
                  User Prompt Template (Supports &#123;&#123;field_id&#125;&#125;)
                </label>
                <Textarea
                  value={formData.prompt.user}
                  onChange={(e) =>
                    updateFormData((prev) => ({
                      ...prev,
                      prompt: { ...prev.prompt, user: e.target.value },
                    }))
                  }
                  rows={6}
                  className="font-mono text-xs"
                  placeholder="e.g. Please analyze the meeting notes: {{meeting_notes}} with a {{tone}} tone."
                />
              </div>
            </div>
          )}

          {/* TAB 3: MODEL SETTINGS */}
          {visualSubTab === "model" && (
            <div className="space-y-4">
              <div>
                <h3 className="text-sm font-semibold text-foreground">Model Configuration</h3>
                <p className="text-xs text-muted-foreground">
                  Specify the LLM provider, default model name, and generation temperature.
                </p>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="space-y-1.5">
                  <label className="text-xs font-medium text-foreground/80">Provider</label>
                  <Select
                    value={formData.model.provider}
                    onValueChange={(val) =>
                      updateFormData((prev) => ({
                        ...prev,
                        model: { ...prev.model, provider: val },
                      }))
                    }
                  >
                    <SelectTrigger className="h-8 text-xs">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="openai">OpenAI</SelectItem>
                      <SelectItem value="leonardo">Leonardo (Image Generation)</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <div className="space-y-1.5">
                  <label className="text-xs font-medium text-foreground/80">Model Name</label>
                  <Input
                    value={formData.model.name}
                    onChange={(e) =>
                      updateFormData((prev) => ({
                        ...prev,
                        model: { ...prev.model, name: e.target.value },
                      }))
                    }
                    placeholder="e.g. gpt-4o-mini, gpt-4o"
                    className="h-8 text-xs font-mono"
                  />
                </div>

                <div className="space-y-2 col-span-1 md:col-span-2">
                  <div className="flex items-center justify-between">
                    <label className="text-xs font-medium text-foreground/80">Temperature</label>
                    <span className="text-xs font-mono text-muted-foreground font-semibold">
                      {formData.model.temperature.toFixed(1)}
                    </span>
                  </div>
                  <input
                    type="range"
                    min="0"
                    max="2"
                    step="0.1"
                    value={formData.model.temperature}
                    onChange={(e) =>
                      updateFormData((prev) => ({
                        ...prev,
                        model: { ...prev.model, temperature: parseFloat(e.target.value) },
                      }))
                    }
                    className="w-full h-2 bg-muted rounded-lg appearance-none cursor-pointer accent-primary"
                  />
                  <div className="flex justify-between text-[10px] text-muted-foreground">
                    <span>0.0 (Precise / Factual)</span>
                    <span>0.7 (Balanced)</span>
                    <span>2.0 (Creative)</span>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* TAB 4: STRUCTURED OUTPUT */}
          {visualSubTab === "output" && (
            <div className="space-y-4">
              <div>
                <h3 className="text-sm font-semibold text-foreground">Structured JSON Output</h3>
                <p className="text-xs text-muted-foreground">
                  If this form outputs structured data (e.g. JSON object with array of items) to feed into subsequent flow steps.
                </p>
              </div>

              <div className="space-y-3">
                <label className="flex items-center gap-2 text-xs text-foreground cursor-pointer select-none">
                  <input
                    type="checkbox"
                    checked={!!formData.output}
                    onChange={(e) => {
                      if (e.target.checked) {
                        updateFormData((prev) => ({
                          ...prev,
                          output: {
                            type: "object",
                            schema: {
                              type: "object",
                              properties: {
                                result: { type: "string" },
                              },
                            },
                          },
                        }));
                      } else {
                        updateFormData((prev) => ({ ...prev, output: null }));
                      }
                    }}
                    className="h-4 w-4 rounded border-border text-primary"
                  />
                  Enable Structured JSON Output
                </label>

                {formData.output && (
                  <div className="space-y-2 pt-2">
                    <label className="text-xs font-medium text-foreground/80">
                      Output JSON Schema
                    </label>
                    <Textarea
                      value={
                        typeof formData.output.schema === "string"
                          ? formData.output.schema
                          : JSON.stringify(formData.output.schema ?? {}, null, 2)
                      }
                      onChange={(e) => {
                        try {
                          const parsed = JSON.parse(e.target.value);
                          updateFormData((prev) => ({
                            ...prev,
                            output: { type: "object", schema: parsed },
                          }));
                        } catch {
                          // Keep editing raw text
                        }
                      }}
                      rows={8}
                      className="font-mono text-xs"
                      placeholder={`{\n  "type": "object",\n  "properties": {\n    "actions": { "type": "array" }\n  }\n}`}
                    />
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

