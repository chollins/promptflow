import type { ExecuteResponse } from "../types/promptForm";
import MarkdownPreview from "./MarkdownPreview";

interface Props {
  result: ExecuteResponse | null;
}

export default function ResultViewer({ result }: Props) {
  if (!result) {
    return null;
  }

  return (
    <div className="space-y-4">
      <div className="bg-white border rounded-lg shadow-sm p-4">
        <h2 className="text-lg font-semibold text-gray-900 mb-2">
          Generated Prompt
        </h2>
        <MarkdownPreview
          text={result.prompt}
          className="text-sm text-gray-700 whitespace-pre-wrap overflow-x-auto"
        />
      </div>

      <div className="bg-white border rounded-lg shadow-sm p-4">
        <h2 className="text-lg font-semibold text-gray-900 mb-2">LLM Result</h2>
        <MarkdownPreview
          text={result.result}
          className="text-sm text-gray-700 whitespace-pre-wrap overflow-x-auto space-y-2"
        />
      </div>
    </div>
  );
}
