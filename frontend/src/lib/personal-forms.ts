export interface PersonalForm {
  id: string;
  name: string;
  description?: string;
  updated_at: string;
  content_json: string;
}

const STORAGE_PREFIX = "promptflow_personal_forms";

function getStorageKey(userId?: string | null): string {
  const safeId = userId && userId.trim() ? userId.trim() : "default";
  return `${STORAGE_PREFIX}_${safeId}`;
}

export function getPersonalForms(userId?: string | null): PersonalForm[] {
  try {
    const raw = localStorage.getItem(getStorageKey(userId));
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed : [];
  } catch (err) {
    console.error("Failed to load personal forms from localStorage", err);
    return [];
  }
}

export function savePersonalForm(userId: string | null | undefined, form: PersonalForm): void {
  try {
    const key = getStorageKey(userId);
    const existing = getPersonalForms(userId);
    const index = existing.findIndex((item) => item.id === form.id);
    const updatedForm: PersonalForm = {
      ...form,
      updated_at: new Date().toISOString(),
    };

    let updatedList: PersonalForm[];
    if (index >= 0) {
      updatedList = [...existing];
      updatedList[index] = updatedForm;
    } else {
      updatedList = [updatedForm, ...existing];
    }

    localStorage.setItem(key, JSON.stringify(updatedList));
  } catch (err) {
    console.error("Failed to save personal form to localStorage", err);
    throw new Error("Could not save to browser storage. Check storage permissions.");
  }
}

export function deletePersonalForm(userId: string | null | undefined, formId: string): void {
  try {
    const key = getStorageKey(userId);
    const existing = getPersonalForms(userId);
    const updatedList = existing.filter((item) => item.id !== formId);
    localStorage.setItem(key, JSON.stringify(updatedList));
  } catch (err) {
    console.error("Failed to delete personal form from localStorage", err);
  }
}

export function exportPersonalForms(userId?: string | null): void {
  const items = getPersonalForms(userId);
  const blob = new Blob([JSON.stringify(items, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `my-personal-forms-${new Date().toISOString().slice(0, 10)}.json`;
  a.click();
  URL.revokeObjectURL(url);
}

export function importPersonalForms(userId: string | null | undefined, jsonContent: string): number {
  try {
    const parsed = JSON.parse(jsonContent);
    const itemsToImport: PersonalForm[] = [];

    if (Array.isArray(parsed)) {
      for (const item of parsed) {
        if (item && typeof item === "object") {
          // If it's already a PersonalForm wrapper
          if (item.id && item.content_json) {
            itemsToImport.push(item);
          } else if (item.name && item.fields) {
            // If it's a raw PromptForm JSON
            itemsToImport.push({
              id: item.id || `pf_local_${Date.now()}_${Math.random().toString(36).substring(2, 6)}`,
              name: item.name,
              description: item.description || "",
              updated_at: new Date().toISOString(),
              content_json: JSON.stringify(item, null, 2),
            });
          }
        }
      }
    } else if (parsed && typeof parsed === "object") {
      // Single PromptForm or wrapper
      if (parsed.id && parsed.content_json) {
        itemsToImport.push(parsed);
      } else if (parsed.name && parsed.fields) {
        itemsToImport.push({
          id: parsed.id || `pf_local_${Date.now()}_${Math.random().toString(36).substring(2, 6)}`,
          name: parsed.name,
          description: parsed.description || "",
          updated_at: new Date().toISOString(),
          content_json: JSON.stringify(parsed, null, 2),
        });
      }
    }

    if (itemsToImport.length === 0) {
      throw new Error("No valid form definitions found in JSON.");
    }

    const key = getStorageKey(userId);
    const existing = getPersonalForms(userId);
    const existingIds = new Set(existing.map((f) => f.id));

    const merged = [...existing];
    for (const item of itemsToImport) {
      let finalItem = item;
      if (existingIds.has(item.id)) {
        // Assign new ID to prevent collision
        finalItem = {
          ...item,
          id: `pf_local_${Date.now()}_${Math.random().toString(36).substring(2, 6)}`,
        };
      }
      merged.unshift(finalItem);
    }

    localStorage.setItem(key, JSON.stringify(merged));
    return itemsToImport.length;
  } catch (err) {
    throw new Error(err instanceof Error ? err.message : "Invalid JSON format");
  }
}

