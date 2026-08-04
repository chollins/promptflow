import { createFileRoute } from "@tanstack/react-router";
import { AppShell, PageHeader } from "@/components/app-shell";
import { FormRunner } from "@/components/form-runner";

export const Route = createFileRoute("/forms_/$formId")({
  component: FormDetailPage,
});

function FormDetailPage() {
  const { formId } = Route.useParams();

  return (
    <AppShell>
      <PageHeader
        title="Form runner"
        description="Render the selected PromptForm dynamically from the backend."
      />
      <FormRunner formId={formId} />
    </AppShell>
  );
}
