import MarkdownPreview from "./MarkdownPreview";

interface Props {
  prompt: string;
  result: string;
  editedResult: string;
  onEditedResultChange: (value: string) => void;
  onContinue: () => void;
  onCancel: () => void;
  saving?: boolean;
  saveAs?: string;
}

export default function StepReview({
  prompt,
  result,
  editedResult,
  onEditedResultChange,
  onContinue,
  onCancel,
  saving = false,
  saveAs,
}: Props) {
  return (
    <div className="space-y-4">
      <div className="bg-white border border-gray-200 rounded-xl shadow-sm p-5 space-y-4">
        <div>
          <h3 className="text-lg font-semibold text-gray-900">Prompt</h3>
          <pre className="mt-2 whitespace-pre-wrap text-sm text-gray-700 bg-gray-50 rounded-lg p-4 border border-gray-200">
            {prompt}
          </pre>
        </div>

        <div>
          <div className="flex items-center justify-between gap-3">
            <h3 className="text-lg font-semibold text-gray-900">LLM Result</h3>
            {saveAs && (
              <span className="text-xs font-medium rounded-full bg-gray-100 px-3 py-1 text-gray-600">
                Saves to {saveAs}
              </span>
            )}
          </div>
          <textarea
            className="mt-2 w-full min-h-56 rounded-lg border border-gray-300 bg-white p-4 text-sm text-gray-800"
            value={editedResult}
            onChange={(e) => onEditedResultChange(e.target.value)}
          />
          <p className="mt-2 text-sm text-gray-500">
            Original result is shown below for reference.
          </p>
          <MarkdownPreview
            text={result}
            className="mt-2 rounded-lg border border-gray-200 bg-gray-50 p-4 text-sm text-gray-700 whitespace-pre-wrap"
          />
        </div>
      </div>

      <div className="flex items-center justify-end gap-3">
        <button
          type="button"
          onClick={onCancel}
          className="rounded-lg border border-gray-300 px-4 py-2 text-gray-700 hover:bg-gray-50"
        >
          Cancel
        </button>
        <button
          type="button"
          onClick={onContinue}
          disabled={saving}
          className="rounded-lg bg-gray-900 px-4 py-2 text-white hover:bg-gray-800 disabled:opacity-60"
        >
          {saving ? "Continuing..." : "Continue"}
        </button>
      </div>
    </div>
  );
}
