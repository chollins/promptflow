import { useParams } from "react-router-dom";
import { AppShell, PageHeader } from "@/components/app-shell";
import { SuperadminFormRunner } from "@/components/SuperadminFormRunner";
import { useCurrentUserId } from "@/lib/use-current-user-id";

export default function FormDetailPage() {
  const { formId } = useParams();
  const userId = useCurrentUserId();

  return (
    <AppShell>
      <PageHeader
        title="Form Runner"
        description="Execute this reusable form."
      />
      {/* key=userId forces a full remount when the authenticated user changes,
          clearing all diagnostic state so data permitted to one user cannot
          remain visible to the next user on the same tab. */}
      <SuperadminFormRunner key={userId ?? "loading"} formId={formId ?? ""} />
    </AppShell>
  );
}

