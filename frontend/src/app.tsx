import { Navigate, Route, Routes } from "react-router-dom";
import Landing from "@/routes/index";
import LoginPage from "@/routes/login";
import SignupPage from "@/routes/signup";
import ForgotPasswordPage from "@/routes/forgot-password";
import ResetPasswordPage from "@/routes/reset-password";
import VerifyOtpPage from "@/routes/verify-otp";
import CreateOrganizationPage from "@/routes/create-organization";
import DashboardPage from "@/routes/dashboard";
import FlowsPage from "@/routes/flows";
import FlowDetailPage from "@/routes/flows_.$flowId";
import FormsPage from "@/routes/forms";
import FormDetailPage from "@/routes/forms_.$formId";
import OrganizationPage from "@/routes/organization";
import SettingsPage from "@/routes/settings";
import UsersPage from "@/routes/users";
import AdminPage from "@/routes/admin";
import AdminFlowsPage from "@/routes/admin.flows";
import AdminFlowDetailPage from "@/routes/admin.flows_.$id";
import AdminFormsPage from "@/routes/admin.forms";
import AdminFormDetailPage from "@/routes/admin.forms_.$id";
import AdminManageFlowsPage from "@/routes/admin.manage-flows";
import AdminOrganizationsPage from "@/routes/admin.organizations";
import AdminOrganizationDetailPage from "@/routes/admin.organizations_.$id";

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<Landing />} />
      <Route path="/login" element={<LoginPage />} />
      <Route path="/signup" element={<SignupPage />} />
      <Route path="/forgot-password" element={<ForgotPasswordPage />} />
      <Route path="/reset-password" element={<ResetPasswordPage />} />
      <Route path="/verify-otp" element={<VerifyOtpPage />} />
      <Route path="/create-organization" element={<CreateOrganizationPage />} />

      <Route path="/dashboard" element={<DashboardPage />} />
      <Route path="/flows" element={<FlowsPage />} />
      <Route path="/flows/:flowId" element={<FlowDetailPage />} />
      <Route path="/forms" element={<FormsPage />} />
      <Route path="/forms/:formId" element={<FormDetailPage />} />
      <Route path="/organization" element={<OrganizationPage />} />
      <Route path="/settings" element={<SettingsPage />} />
      <Route path="/users" element={<UsersPage />} />
      <Route path="/admin" element={<AdminPage />} />
      <Route path="/admin/flows" element={<AdminFlowsPage />} />
      <Route path="/admin/flows/:id" element={<AdminFlowDetailPage />} />
      <Route path="/admin/forms" element={<AdminFormsPage />} />
      <Route path="/admin/forms/:id" element={<AdminFormDetailPage />} />
      <Route path="/admin/manage-flows" element={<AdminManageFlowsPage />} />
      <Route path="/admin/organizations" element={<AdminOrganizationsPage />} />
      <Route path="/admin/organizations/:id" element={<AdminOrganizationDetailPage />} />
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  );
}
