import { useEffect, useState } from "react";
import { getMockUser, type MockUser } from "@/lib/mock-store";

export function useMockUser(): MockUser {
  const [user, setUser] = useState<MockUser>(() => ({
    firstName: "John",
    lastName: "Doe",
    email: "john@email.com",
    role: "admin",
  }));

  useEffect(() => {
    setUser(getMockUser());
    const handler = () => setUser(getMockUser());
    window.addEventListener("promptflow:user-change", handler);
    window.addEventListener("storage", handler);
    return () => {
      window.removeEventListener("promptflow:user-change", handler);
      window.removeEventListener("storage", handler);
    };
  }, []);

  return user;
}
