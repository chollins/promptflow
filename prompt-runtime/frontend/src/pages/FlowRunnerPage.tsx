import { useEffect, useState } from "react";
import { useParams } from "react-router-dom";

import { getErrorMessage } from "../api/forms";
import { getFlow } from "../api/flows";
import FlowRunner from "../components/FlowRunner";
import LoadingSpinner from "../components/LoadingSpinner";
import type { PromptFlow } from "../types/promptForm";

export default function FlowRunnerPage() {
  const { flowId } = useParams<{ flowId: string }>();
  const [flow, setFlow] = useState<PromptFlow | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const currentFlowId = flowId;
    if (!currentFlowId) {
      setError("Flow not found.");
      setLoading(false);
      return;
    }

    async function loadFlow() {
      try {
        setLoading(true);
        setError(null);
        const data = await getFlow(currentFlowId!);
        setFlow(data);
      } catch (err) {
        setError(getErrorMessage(err));
      } finally {
        setLoading(false);
      }
    }

    loadFlow();
  }, [flowId]);

  if (loading) {
    return <LoadingSpinner />;
  }

  if (error && !flow) {
    return (
      <div className="rounded-xl border border-red-200 bg-red-50 p-4 text-red-700">
        {error}
      </div>
    );
  }

  if (!flow) {
    return null;
  }

  return <FlowRunner flow={flow} />;
}
