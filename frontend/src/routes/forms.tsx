import { createFileRoute, Link } from "@tanstack/react-router";
import { useEffect, useState } from "react";
import { AppShell, PageHeader } from "@/components/app-shell";
import { Card } from "@/components/ui-kit";
import { apiGet } from "@/lib/api";

type FormItem = {
  id: string;
  name: string;
  description?: string | null;
  version: string;
  fields: Array<{ id: string; type: string }>;
};

export const Route = createFileRoute("/forms")({
  component: FormsPage,
});

function FormsPage() {
  const [items, setItems] = useState<FormItem[]>([]);

  useEffect(() => {
    apiGet<{ items: FormItem[] }>("/forms")
      .then((data) => setItems(data.items))
      .catch(() => setItems([]));
  }, []);

  return (
    <AppShell>
      <PageHeader
        title="Forms"
        description="Prompt forms are loaded from JSON and executed by the backend service."
      />
      <div className="grid grid-cols-1 gap-4">
        {items.length === 0 ? (
          <Card className="p-5">
            <div className="font-medium">No forms found</div>
            <p className="mt-2 text-sm text-muted-foreground">
              Add form JSON files to backend/forms to populate this page.
            </p>
          </Card>
        ) : (
          items.map((form) => (
            <Card key={form.id} className="p-5">
              <div className="flex items-start justify-between gap-4">
                <div>
                  <div className="font-medium">{form.name}</div>
                  <p className="mt-2 text-sm text-muted-foreground">
                    {form.description || "No description."}
                  </p>
                  <p className="mt-2 text-xs text-muted-foreground">
                    {form.fields.length} fields · v{form.version}
                  </p>
                </div>
                <Link
                  to="/forms/$formId"
                  params={{ formId: form.id }}
                  className="inline-flex h-10 items-center justify-center rounded-md border border-border bg-background px-4 text-sm font-medium text-foreground transition-colors hover:bg-muted"
                >
                  Open
                </Link>
              </div>
            </Card>
          ))
        )}
      </div>
    </AppShell>
  );
}
