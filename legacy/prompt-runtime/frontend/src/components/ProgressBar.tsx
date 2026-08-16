interface Props {
  current: number;
  total: number;
}

export default function ProgressBar({ current, total }: Props) {
  const percentage = total === 0 ? 0 : Math.min(100, Math.round((current / total) * 100));

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between text-sm text-gray-600">
        <span>
          Step {current} of {total}
        </span>
        <span>{percentage}%</span>
      </div>
      <div className="h-2 rounded-full bg-gray-200 overflow-hidden">
        <div
          className="h-full rounded-full bg-gray-900 transition-all"
          style={{ width: `${percentage}%` }}
        />
      </div>
    </div>
  );
}
