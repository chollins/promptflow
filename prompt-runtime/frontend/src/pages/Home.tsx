import { useEffect, useState } from "react";
import { Link } from "react-router-dom";

import { getErrorMessage, getForms } from "../api/forms";
import LoadingSpinner from "../components/LoadingSpinner";
import type { FormSummary } from "../types/promptForm";

export default function Home() {
  const [forms, setForms] = useState<FormSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function loadForms() {
      try {
        setLoading(true);
        setError(null);
        const data = await getForms();
        setForms(data);
      } catch (err) {
        setError(getErrorMessage(err));
      } finally {
        setLoading(false);
      }
    }

    loadForms();
  }, []);

  if (loading) {
    return <LoadingSpinner />;
  }

  if (error) {
    return (
      <div className="bg-red-50 border border-red-200 text-red-700 rounded-lg p-4">
        {error}
      </div>
    );
  }

  if (forms.length === 0) {
    return (
      <p className="text-gray-600 text-center py-12">No forms found.</p>
    );
  }

  return (
    <div className="grid gap-4">
      {forms.map((form) => (
        <div
          key={form.id}
          className="bg-white border rounded-lg shadow-sm p-4 flex items-start justify-between gap-4"
        >
          <div>
            <h2 className="text-lg font-semibold text-gray-900">{form.name}</h2>
            {form.description && (
              <p className="mt-1 text-gray-600">{form.description}</p>
            )}
          </div>
          <Link
            to={`/forms/${form.id}`}
            className="shrink-0 bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded"
          >
            Open
          </Link>
        </div>
      ))}
    </div>
  );
}
