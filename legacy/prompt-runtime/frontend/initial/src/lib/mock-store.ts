// Simple mock store using localStorage so state can be switched
// and persists across route navigations.

export type Role = "admin" | "user" | "superadmin";

export interface MockUser {
  firstName: string;
  lastName: string;
  email: string;
  role: Role;
}

export const ORG = {
  name: "Digital Mixology",
  code: "DMX",
};

export interface Flow {
  id: string;
  name: string;
  enabled: boolean;
  description: string;
}

export const FLOWS: Flow[] = [
  { id: "f_lead", name: "Lead Research", enabled: true, description: "Automated prospect research pipeline." },
  { id: "f_deck", name: "Deck Generator", enabled: true, description: "Generate polished pitch decks from a brief." },
  { id: "f_comp", name: "Competitor Analysis", enabled: true, description: "Track and summarize competitor moves." },
  { id: "f_img", name: "Image Generation", enabled: false, description: "Create on-brand imagery from prompts." },
];

export interface OrgMember {
  name: string;
  email: string;
  role: "Admin" | "User";
  status: "Active" | "Invited";
}

export interface Organization {
  id: string;
  name: string;
  code: string;
  plan: "Free" | "Team" | "Enterprise";
  createdAt: string;
  members: OrgMember[];
  flowIds: string[];
}

export const ORGANIZATIONS: Organization[] = [
  {
    id: "org_dmx",
    name: "Digital Mixology",
    code: "DMX",
    plan: "Team",
    createdAt: "2025-02-14",
    flowIds: ["f_lead", "f_deck", "f_comp"],
    members: [
      { name: "John Doe", email: "john@email.com", role: "Admin", status: "Active" },
      { name: "Mary Jane", email: "mary@email.com", role: "User", status: "Active" },
      { name: "Alex Kim", email: "alex@email.com", role: "User", status: "Invited" },
    ],
  },
  {
    id: "org_north",
    name: "Northwind Labs",
    code: "NWL",
    plan: "Enterprise",
    createdAt: "2024-11-02",
    flowIds: ["f_lead", "f_comp", "f_img"],
    members: [
      { name: "Rita Blake", email: "rita@northwind.io", role: "Admin", status: "Active" },
      { name: "Sam Ortiz", email: "sam@northwind.io", role: "Admin", status: "Active" },
      { name: "Priya Shah", email: "priya@northwind.io", role: "User", status: "Active" },
      { name: "Owen Park", email: "owen@northwind.io", role: "User", status: "Invited" },
    ],
  },
  {
    id: "org_aurora",
    name: "Aurora Studio",
    code: "AUR",
    plan: "Free",
    createdAt: "2026-05-19",
    flowIds: ["f_deck"],
    members: [
      { name: "Nina Rossi", email: "nina@aurora.design", role: "Admin", status: "Active" },
    ],
  },
];

const KEY = "promptflow.mock.user";

export function getMockUser(): MockUser {
  if (typeof window === "undefined") {
    return { firstName: "John", lastName: "Doe", email: "john@email.com", role: "admin" };
  }
  try {
    const raw = localStorage.getItem(KEY);
    if (raw) return JSON.parse(raw) as MockUser;
  } catch {}
  const initial: MockUser = { firstName: "John", lastName: "Doe", email: "john@email.com", role: "admin" };
  localStorage.setItem(KEY, JSON.stringify(initial));
  return initial;
}

export function setMockUser(u: MockUser) {
  localStorage.setItem(KEY, JSON.stringify(u));
  window.dispatchEvent(new Event("promptflow:user-change"));
}

export function setMockRole(role: Role) {
  const u = getMockUser();
  setMockUser({ ...u, role });
}
