import { useEffect, useState } from "react";

import { getErrorMessage, getForms } from "../api/forms";
import LoadingSpinner from "../components/LoadingSpinner";
import type { FormSummary } from "../types/promptForm";
import { Link } from "react-router-dom";

export default function FormsHome() {
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
      <div className="rounded-xl border border-red-200 bg-red-50 p-4 text-red-700">
        {error}
      </div>
    );
  }

  if (forms.length === 0) {
    return <p className="py-12 text-center text-gray-600">No forms found.</p>;
  }

  return (
    <div className="grid gap-4">
      {forms.map((form) => (
        <div key={form.id} className="bg-white border rounded-xl shadow-sm p-5 flex items-start justify-between gap-4">
          <div>
            <h2 className="text-lg font-semibold text-gray-900">{form.name}</h2>
            {form.description && <p className="mt-1 text-gray-600">{form.description}</p>}
            <p className="mt-2 text-sm text-gray-500">Version {form.version}</p>
          </div>
          <Link
            to={`/forms/${form.id}`}
            className="shrink-0 rounded-lg bg-gray-900 px-4 py-2 text-white hover:bg-gray-800"
          >
            Open
          </Link>
        </div>
      ))}
    </div>
  );
}
