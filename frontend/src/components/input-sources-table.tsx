import { formatDebugValue } from "@/components/debug-json-viewer";

type InputSource = {
  field_id: string;
  label: string;
  source_type: string;
  source_name: string;
  path: string;
  value: unknown;
};

export function InputSourcesTable({ sources }: { sources: InputSource[] }) {
  if (!sources.length) {
    return <div className="text-sm text-muted-foreground">No input source metadata available.</div>;
  }

  return (
    <div className="overflow-hidden rounded-lg border border-border">
      <table className="min-w-full divide-y divide-border text-sm">
        <thead className="bg-muted/40">
          <tr>
            <th className="px-3 py-2 text-left font-medium">Field</th>
            <th className="px-3 py-2 text-left font-medium">Source</th>
            <th className="px-3 py-2 text-left font-medium">Path</th>
            <th className="px-3 py-2 text-left font-medium">Value</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-border">
          {sources.map((source) => (
            <tr key={`${source.source_type}-${source.field_id}-${source.path}`}>
              <td className="px-3 py-2">
                <div className="font-medium">{source.label}</div>
                <div className="text-xs text-muted-foreground">{source.source_type}</div>
              </td>
              <td className="px-3 py-2">{source.source_name}</td>
              <td className="px-3 py-2 font-mono text-xs text-muted-foreground">{source.path}</td>
              <td className="px-3 py-2">
                <div className="max-h-24 overflow-auto rounded border border-border bg-background p-2 font-mono text-xs whitespace-pre-wrap">
                  {formatDebugValue(source.value)}
                </div>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
