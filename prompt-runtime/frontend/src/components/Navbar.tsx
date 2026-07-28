import { Link, NavLink } from "react-router-dom";

const linkClass = ({ isActive }: { isActive: boolean }) =>
  [
    "rounded-lg px-3 py-2 text-sm font-medium transition-colors",
    isActive ? "bg-gray-900 text-white" : "text-gray-600 hover:text-gray-900 hover:bg-gray-100",
  ].join(" ");

export default function Navbar() {
  return (
    <nav className="bg-white border-b border-gray-200">
      <div className="max-w-5xl mx-auto p-6 flex items-center justify-between gap-6">
        <Link to="/flows" className="text-lg font-semibold text-gray-900">
          PromptFlow Runtime
        </Link>
        <div className="flex items-center gap-2">
          <NavLink to="/forms" className={linkClass}>
            PromptForms
          </NavLink>
          <NavLink to="/flows" className={linkClass}>
            PromptFlow
          </NavLink>
        </div>
      </div>
    </nav>
  );
}
