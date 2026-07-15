import { BrowserRouter, Link, Route, Routes } from "react-router-dom";

import Home from "./pages/Home";
import PromptFormPage from "./pages/PromptFormPage";

export default function App() {
  return (
    <BrowserRouter>
      <div className="min-h-screen bg-gray-100">
        <nav className="bg-white border-b">
          <div className="max-w-5xl mx-auto p-6 flex items-center gap-6">
            <span className="text-lg font-semibold text-gray-900">
              PromptFlow
            </span>
            <Link
              to="/"
              className="text-gray-600 hover:text-gray-900"
            >
              {/* Home */}
            </Link>
          </div>
        </nav>

        <main className="max-w-5xl mx-auto p-6 space-y-6">
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/forms/:id" element={<PromptFormPage />} />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  );
}
