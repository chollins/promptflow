import { useEffect, useState } from "react";
import { apiGet, hasSessionToken } from "@/lib/api";

export function useCurrentUserId(): string | null {
  const [userId, setUserId] = useState<string | null>(null);

  useEffect(() => {
    if (!hasSessionToken()) {
      setUserId(null);
      return;
    }

    let active = true;
    apiGet<{ user: { id: string } }>("/auth/me")
      .then((data) => {
        if (active && data?.user?.id) {
          setUserId(data.user.id);
        }
      })
      .catch(() => {
        if (active) setUserId(null);
      });

    return () => {
      active = false;
    };
  }, []);

  return userId;
}
