import { useEffect, useState } from "react";

import { getErrorMessage } from "../api/forms";
import { getFlows } from "../api/flows";
import FlowCard from "../components/FlowCard";
import LoadingSpinner from "../components/LoadingSpinner";
import type { FlowSummary } from "../types/promptForm";

export default function Home() {
  const [flows, setFlows] = useState<FlowSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function loadFlows() {
      try {
        setLoading(true);
        setError(null);
        const data = await getFlows();
        setFlows(data);
      } catch (err) {
        setError(getErrorMessage(err));
      } finally {
        setLoading(false);
      }
    }

    loadFlows();
  }, []);

  if (loading) {
    return <LoadingSpinner />;
  }

  if (error) {
    return (
      <div className="rounded-xl border border-red-200 bg-red-50 p-4 text-red-700">
        {error}
      </div>
    );
  }

  if (flows.length === 0) {
    return <p className="py-12 text-center text-gray-600">No flows found.</p>;
  }

  return (
    <div className="grid gap-4">
      {flows.map((flow) => (
        <FlowCard key={flow.id} flow={flow} />
      ))}
    </div>
  );
}
