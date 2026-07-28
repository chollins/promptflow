import { Link } from "react-router-dom";

import type { FlowSummary } from "../types/promptForm";

interface Props {
  flow: FlowSummary;
}

export default function FlowCard({ flow }: Props) {
  return (
    <div className="bg-white border border-gray-200 rounded-xl shadow-sm p-5 flex items-start justify-between gap-4">
      <div className="space-y-1">
        <h2 className="text-lg font-semibold text-gray-900">{flow.name}</h2>
        {flow.description && <p className="text-gray-600">{flow.description}</p>}
        <p className="text-sm text-gray-500">Version {flow.version}</p>
      </div>
      <Link
        to={`/flows/${flow.id}`}
        className="shrink-0 rounded-lg bg-gray-900 px-4 py-2 text-white hover:bg-gray-800"
      >
        Open
      </Link>
    </div>
  );
}
