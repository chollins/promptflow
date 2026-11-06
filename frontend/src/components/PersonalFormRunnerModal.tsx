import React, { useEffect, useMemo, useState } from "react";
import { Play, Sparkles, Copy, Check, AlertCircle, X, CheckCircle2 } from "lucide-react";
import { Button, Card, Input } from "@/components/ui-kit";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
} from "@/components/ui/dialog";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import DynamicForm, { type PromptForm } from "@/components/DynamicForm";
import { FormattedResultView } from "@/components/formatted-result-view";
import { apiGet, apiPost } from "@/lib/api";
import { logActivity } from "@/lib/activity";
import { toast } from "sonner";
import type { PersonalForm } from "@/lib/personal-forms";

interface PersonalFormRunnerModalProps {
  personalForm: PersonalForm | null;
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

export function PersonalFormRunnerModal({
  personalForm,
  open,
  onOpenChange,
}: PersonalFormRunnerModalProps) {
  const [formDef, setFormDef] = useState<PromptForm | null>(null);
  const [values, setValues] = useState<Record<string, string>>({});
  const [running, setRunning] = useState(false);
  const [result, setResult] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);
  const [modelsList, setModelsList] = useState<Array<{ id: string; name: string }>>([]);
  const [selectedModel, setSelectedModel] = useState<string>("gpt-4o-mini");

  // Parse form definition from personalForm content_json
  useEffect(() => {
    if (personalForm?.content_json) {
      try {
        const parsed = JSON.parse(personalForm.content_json);
        setFormDef(parsed);
        const initialVals: Record<string, string> = {};
        if (Array.isArray(parsed.fields)) {
          for (const f of parsed.fields) {
            initialVals[f.id] = f.type === "checkbox" ? "false" : (f.default ?? "");
          }
        }
        setValues(initialVals);
        setResult(null);
        setError(null);
        if (parsed.model?.name) {
          setSelectedModel(parsed.model.name);
        }
      } catch (err) {
        setError("Invalid form definition JSON");
      }
    } else {
      setFormDef(null);
      setValues({});
      setResult(null);
      setError(null);
    }
  }, [personalForm]);

  // Load available models for dropdown
  useEffect(() => {
    apiGet<{ models: Array<{ id: string; name: string }> }>("/models")
      .then((data) => {
        if (data.models && data.models.length > 0) {
          setModelsList(data.models);
        }
      })
      .catch(() => {
        // Fallback standard models
        setModelsList([
          { id: "gpt-4o-mini", name: "GPT-4o Mini" },
          { id: "gpt-4o", name: "GPT-4o" },
        ]);
      });
  }, []);

  async function handleExecute() {
    if (!formDef || !personalForm) return;
    setRunning(true);
    setError(null);
    try {
      const response = await apiPost<{
        prompt?: string;
        result: string;
        saved_result_id?: string;
      }>("/forms/adhoc/execute", {
        form_definition: formDef,
        values,
        model_override: selectedModel,
      });

      setResult(response.result);
      toast.success("Execution completed! Result saved to your history.");
      logActivity("executed personal form", personalForm.name);
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Execution failed";
      setError(msg);
      toast.error(msg);
    } finally {
      setRunning(false);
    }
  }

  function handleCopy() {
    if (!result) return;
    void navigator.clipboard.writeText(result);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
    toast.success("Copied result to clipboard");
  }

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="w-[95vw] max-w-4xl max-h-[90vh] flex flex-col p-6">
        <DialogHeader className="border-b border-border pb-3">
          <div className="flex items-center gap-2">
            <span className="text-[11px] px-2 py-0.5 rounded-full font-medium bg-primary/10 text-primary border border-primary/20">
              Personal Form
            </span>
            <DialogTitle className="text-xl font-bold truncate">
              {personalForm?.name || "Run Personal Form"}
            </DialogTitle>
          </div>
          {personalForm?.description && (
            <DialogDescription className="text-xs text-muted-foreground mt-1">
              {personalForm.description}
            </DialogDescription>
          )}
        </DialogHeader>

        <div className="overflow-y-auto flex-1 space-y-6 pr-1 pt-2">
          {error && (
            <div className="rounded-lg bg-destructive/10 border border-destructive/20 p-3 text-destructive text-xs flex items-center gap-2">
              <AlertCircle className="h-4 w-4 shrink-0" />
              <span>{error}</span>
            </div>
          )}

          {formDef && (
            <div className="space-y-4">
              <DynamicForm
                form={formDef}
                values={values}
                onValuesChange={setValues}
              />

              <div className="flex flex-wrap items-center justify-between gap-3 pt-4 border-t border-border">
                <div className="flex items-center gap-2">
                  <span className="text-xs font-medium text-foreground/80">Model:</span>
                  <Select value={selectedModel} onValueChange={setSelectedModel}>
                    <SelectTrigger className="h-8 text-xs min-w-[160px]">
                      <SelectValue placeholder="Select Model" />
                    </SelectTrigger>
                    <SelectContent>
                      {modelsList.map((m) => (
                        <SelectItem key={m.id} value={m.id}>
                          {m.name}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>

                <Button
                  onClick={() => void handleExecute()}
                  disabled={running}
                  className="gap-2 h-9 px-4 font-medium"
                >
                  <Play className="h-4 w-4 fill-current" />
                  {running ? "Generating..." : "Execute Form"}
                </Button>
              </div>
            </div>
          )}

          {/* Execution Result Output */}
          {result && (
            <div className="border border-border rounded-xl bg-card p-4 space-y-3 mt-4">
              <div className="flex items-center justify-between border-b border-border pb-2">
                <div className="flex items-center gap-1.5 text-xs font-semibold text-foreground">
                  <CheckCircle2 className="h-4 w-4 text-emerald-500" />
                  Generated Output
                </div>
                <Button
                  type="button"
                  variant="outline"
                  size="sm"
                  onClick={handleCopy}
                  className="h-7 text-xs gap-1.5"
                >
                  {copied ? (
                    <>
                      <Check className="h-3.5 w-3.5 text-emerald-500" /> Copied
                    </>
                  ) : (
                    <>
                      <Copy className="h-3.5 w-3.5" /> Copy
                    </>
                  )}
                </Button>
              </div>

              <div className="text-sm">
                <FormattedResultView content={result} />
              </div>
            </div>
          )}
        </div>
      </DialogContent>
    </Dialog>
  );
}

