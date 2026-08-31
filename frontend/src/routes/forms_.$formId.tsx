import { useParams } from "react-router-dom";
import { AppShell, PageHeader } from "@/components/app-shell";
import { SuperadminFormRunner } from "@/components/SuperadminFormRunner";

export default function FormDetailPage() {
  const { formId } = useParams();

  return (
    <AppShell>
      <PageHeader
        title="Form Runner"
        description="Execute this reusable form."
      />
      <SuperadminFormRunner formId={formId} />
    </AppShell>
  );
}

