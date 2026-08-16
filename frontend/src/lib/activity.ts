export type ActivityLog = {
  id: string;
  action: string;
  target: string;
  timestamp: string;
};

const ACTIVITY_KEY = "promptflow_recent_activity";

export function getRecentActivity(): ActivityLog[] {
  if (typeof window === "undefined") return [];
  try {
    const data = localStorage.getItem(ACTIVITY_KEY);
    if (!data) return [];
    return JSON.parse(data);
  } catch {
    return [];
  }
}

export function logActivity(action: string, target: string) {
  if (typeof window === "undefined") return;
  const activity: ActivityLog = {
    id: typeof crypto !== "undefined" && "randomUUID" in crypto ? crypto.randomUUID() : Math.random().toString(36).substring(7),
    action,
    target,
    timestamp: new Date().toISOString(),
  };

  const existing = getRecentActivity();
  const updated = [activity, ...existing].slice(0, 10);
  localStorage.setItem(ACTIVITY_KEY, JSON.stringify(updated));
  window.dispatchEvent(new Event("promptflow-activity-updated"));
}

export function clearRecentActivity() {
  if (typeof window === "undefined") return;
  localStorage.removeItem(ACTIVITY_KEY);
  window.dispatchEvent(new Event("promptflow-activity-updated"));
}
