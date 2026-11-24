import { useEffect, useState } from "react";
import { apiGet } from "@/lib/api";

export function useCurrentUserId(): string | null {
  const [userId, setUserId] = useState<string | null>(null);

  useEffect(() => {
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
