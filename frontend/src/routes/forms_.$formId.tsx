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
        title="Form Runner"
        description="Execute this reusable form. Management lives in the superadmin forms catalog."
      />
      <FormRunner key={formId} formId={formId} />
    </AppShell>
  );
}
