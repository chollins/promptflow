import { useEffect, useState } from "react";

const FAVORITE_MODELS_STORAGE_KEY = "promptflow_favorite_models";
const FAVORITE_MODELS_EVENT = "promptflow-favorite-models-updated";

/**
 * Retrieve the list of favorite model IDs from localStorage.
 */
export function getFavoriteModelIds(): string[] {
  try {
    const raw = localStorage.getItem(FAVORITE_MODELS_STORAGE_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

/**
 * Save favorite model IDs to localStorage and dispatch custom update event.
 */
export function saveFavoriteModelIds(ids: string[]): void {
  try {
    localStorage.setItem(FAVORITE_MODELS_STORAGE_KEY, JSON.stringify(ids));
    window.dispatchEvent(new Event(FAVORITE_MODELS_EVENT));
  } catch {
    // ignore storage exceptions
  }
}

/**
 * Toggle favorite state for a model ID.
 * Returns true if now favorited, false if removed.
 */
export function toggleFavoriteModel(modelId: string): boolean {
  const current = getFavoriteModelIds();
  const exists = current.includes(modelId);
  const next = exists
    ? current.filter((id) => id !== modelId)
    : [...current, modelId];
  saveFavoriteModelIds(next);
  return !exists;
}

/**
 * Check if a model is currently favorited.
 */
export function isFavoriteModel(modelId: string): boolean {
  return getFavoriteModelIds().includes(modelId);
}

/**
 * React hook to access and toggle favorite models reactively.
 */
export function useFavoriteModels() {
  const [favoriteIds, setFavoriteIds] = useState<string[]>(() =>
    getFavoriteModelIds()
  );

  useEffect(() => {
    const update = () => {
      setFavoriteIds(getFavoriteModelIds());
    };

    window.addEventListener(FAVORITE_MODELS_EVENT, update);
    window.addEventListener("storage", update);

    return () => {
      window.removeEventListener(FAVORITE_MODELS_EVENT, update);
      window.removeEventListener("storage", update);
    };
  }, []);

  const isFavorite = (id: string) => favoriteIds.includes(id);

  const toggleFavorite = (id: string) => {
    return toggleFavoriteModel(id);
  };

  return { favoriteIds, isFavorite, toggleFavorite };
}

export const INITIAL_DEFAULT_MODELS = [
  { id: "gpt-4o", name: "GPT-4o", provider: "openai", category: "text" },
  { id: "gpt-4o-mini", name: "GPT-4o Mini", provider: "openai", category: "text" },
  { id: "claude-3-5-sonnet", name: "Claude 3.5 Sonnet", provider: "anthropic", category: "text" },
  { id: "gemini-1.5-pro", name: "Gemini 1.5 Pro", provider: "google", category: "text" },
];

/**
 * Prioritizes favorite models at the top of the list, preserving alphabetical order within groups.
 */
export function sortModelsWithFavorites<T extends { id: string; name: string }>(
  models: T[],
  favoriteIds: string[]
): T[] {
  const existingIds = new Set(models.map((m) => m.id));
  const missingFavorites = favoriteIds
    .filter((id) => !existingIds.has(id))
    .map((id) => ({ id, name: id, provider: "custom", category: "text" } as unknown as T));

  const all = [...missingFavorites, ...models];
  return all.sort((a, b) => {
    const aFav = favoriteIds.includes(a.id);
    const bFav = favoriteIds.includes(b.id);
    if (aFav && !bFav) return -1;
    if (!aFav && bFav) return 1;
    return a.name.localeCompare(b.name);
  });
}

/**
 * Returns the default model ID:
 * - If there are favorite models, returns the first favorite.
 * - If no favorites, defaults to "gpt-4o".
 */
export function getDefaultModelId(
  favoriteIds?: string[],
  availableModels?: Array<{ id: string }>
): string {
  const favs = favoriteIds ?? getFavoriteModelIds();
  if (favs.length > 0) {
    if (availableModels && availableModels.length > 0) {
      const match = favs.find((favId) =>
        availableModels.some((m) => m.id === favId)
      );
      if (match) return match;
    }
    return favs[0];
  }

  return "gpt-4o";
}

