import { BrowserRouter, Route, Routes } from "react-router-dom";

import FlowRunnerPage from "./pages/FlowRunnerPage";
import Home from "./pages/Home";
import FormsHome from "./pages/FormsHome";
import PromptFormPage from "./pages/PromptFormPage";
import Navbar from "./components/Navbar";

export default function App() {
  return (
    <BrowserRouter>
      <div className="min-h-screen bg-gray-100">
        <Navbar />

        <main className="max-w-5xl mx-auto p-6 space-y-6">
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/flows" element={<Home />} />
            <Route path="/forms" element={<FormsHome />} />
            <Route path="/forms/:id" element={<PromptFormPage />} />
            <Route path="/flows/:flowId" element={<FlowRunnerPage />} />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  );
}
