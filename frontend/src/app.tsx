import { Navigate, Route, Routes } from "react-router-dom";
import { AppShell } from "@/components/app-shell";
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

      <Route
        path="/dashboard"
        element={
          <AppShell>
            <DashboardPage />
          </AppShell>
        }
      />
      <Route
        path="/flows"
        element={
          <AppShell>
            <FlowsPage />
          </AppShell>
        }
      />
      <Route
        path="/flows/:flowId"
        element={
          <AppShell>
            <FlowDetailPage />
          </AppShell>
        }
      />
      <Route
        path="/forms"
        element={
          <AppShell>
            <FormsPage />
          </AppShell>
        }
      />
      <Route
        path="/forms/:formId"
        element={
          <AppShell>
            <FormDetailPage />
          </AppShell>
        }
      />
      <Route
        path="/organization"
        element={
          <AppShell>
            <OrganizationPage />
          </AppShell>
        }
      />
      <Route
        path="/settings"
        element={
          <AppShell>
            <SettingsPage />
          </AppShell>
        }
      />
      <Route
        path="/users"
        element={
          <AppShell>
            <UsersPage />
          </AppShell>
        }
      />
      <Route path="/admin" element={<AppShell><AdminPage /></AppShell>} />
      <Route path="/admin/flows" element={<AppShell><AdminFlowsPage /></AppShell>} />
      <Route path="/admin/flows/:id" element={<AppShell><AdminFlowDetailPage /></AppShell>} />
      <Route path="/admin/forms" element={<AppShell><AdminFormsPage /></AppShell>} />
      <Route path="/admin/forms/:id" element={<AppShell><AdminFormDetailPage /></AppShell>} />
      <Route path="/admin/manage-flows" element={<AppShell><AdminManageFlowsPage /></AppShell>} />
      <Route
        path="/admin/organizations"
        element={
          <AppShell>
            <AdminOrganizationsPage />
          </AppShell>
        }
      />
      <Route
        path="/admin/organizations/:id"
        element={
          <AppShell>
            <AdminOrganizationDetailPage />
          </AppShell>
        }
      />
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  );
}
