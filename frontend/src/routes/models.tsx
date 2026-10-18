import { useEffect, useMemo, useState } from "react";
import {
  Cpu,
  Search,
  Star,
  FileText,
  Image as ImageIcon,
  Volume2,
  Sparkles,
  SlidersHorizontal,
} from "lucide-react";
import { AppShell, PageHeader } from "@/components/app-shell";
import { Button, Card, Input } from "@/components/ui-kit";
import { apiGet } from "@/lib/api";
import { useFavoriteModels } from "@/lib/favorite-models";
import { toast } from "sonner";

export type AIModelItem = {
  id: string;
  name: string;
  provider: string;
  category: "text" | "image" | "audio" | string;
  description?: string | null;
  is_active: boolean;
  recommended?: boolean;
};

export default function MemberModelsPage() {
  const [models, setModels] = useState<AIModelItem[]>([]);
  const [loading, setLoading] = useState(true);

  // Filters
  const [searchQuery, setSearchQuery] = useState("");
  const [categoryFilter, setCategoryFilter] = useState<string>("all");
  const [favoritesOnly, setFavoritesOnly] = useState<boolean>(false);

  const { favoriteIds, isFavorite, toggleFavorite } = useFavoriteModels();

  const fetchActiveModels = () => {
    setLoading(true);
    apiGet<{ models: AIModelItem[] }>("/models")
      .then((data) => {
        // Guarantee only active models are displayed
        const activeOnly = (data.models || []).filter(
          (m) => m.is_active !== false
        );
        setModels(activeOnly);
      })
      .catch((err) => {
        toast.error(`Failed to load AI models: ${err.message || err}`);
      })
      .finally(() => setLoading(false));
  };

  useEffect(() => {
    fetchActiveModels();
  }, []);

  const handleToggleFavorite = (model: AIModelItem) => {
    const isNowFav = toggleFavorite(model.id);
    if (isNowFav) {
      toast.success(`Added ${model.name} to favorites`);
    } else {
      toast.info(`Removed ${model.name} from favorites`);
    }
  };

  // Filtered and sorted models
  const filteredModels = useMemo(() => {
    return models
      .filter((m) => {
        // Category filter
        if (categoryFilter !== "all" && m.category !== categoryFilter) {
          return false;
        }

        // Favorites filter
        if (favoritesOnly && !isFavorite(m.id)) {
          return false;
        }

        // Search query
        if (searchQuery.trim()) {
          const q = searchQuery.toLowerCase();
          const matchName = m.name.toLowerCase().includes(q);
          const matchDesc = (m.description || "").toLowerCase().includes(q);
          const matchProvider = m.provider.toLowerCase().includes(q);
          const matchId = m.id.toLowerCase().includes(q);
          if (!matchName && !matchDesc && !matchProvider && !matchId) {
            return false;
          }
        }

        return true;
      })
      .sort((a, b) => {
        // Prioritize favorites first, then recommended, then alphabetical
        const aFav = isFavorite(a.id);
        const bFav = isFavorite(b.id);
        if (aFav && !bFav) return -1;
        if (!aFav && bFav) return 1;

        if (a.recommended && !b.recommended) return -1;
        if (!a.recommended && b.recommended) return 1;

        return a.name.localeCompare(b.name);
      });
  }, [models, categoryFilter, favoritesOnly, searchQuery, favoriteIds]);

  const stats = useMemo(() => {
    const total = models.length;
    const favCount = models.filter((m) => isFavorite(m.id)).length;
    const textCount = models.filter((m) => m.category === "text").length;
    const imageCount = models.filter((m) => m.category === "image").length;
    return { total, favCount, textCount, imageCount };
  }, [models, favoriteIds]);

  const getCategoryBadge = (category: string) => {
    switch (category) {
      case "text":
        return (
          <span className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-md text-xs font-medium bg-muted text-foreground border border-border">
            <FileText className="h-3 w-3 text-muted-foreground" />
            Text
          </span>
        );
      case "image":
        return (
          <span className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-md text-xs font-medium bg-muted text-foreground border border-border">
            <ImageIcon className="h-3 w-3 text-muted-foreground" />
            Image
          </span>
        );
      case "audio":
        return (
          <span className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-md text-xs font-medium bg-muted text-foreground border border-border">
            <Volume2 className="h-3 w-3 text-muted-foreground" />
            Audio
          </span>
        );
      default:
        return (
          <span className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-md text-xs font-medium bg-muted text-muted-foreground border border-border">
            {category}
          </span>
        );
    }
  };

  const getProviderBadge = (provider: string) => {
    return (
      <span className="inline-flex items-center px-1.5 py-0.5 rounded text-[11px] font-mono uppercase bg-muted text-muted-foreground border border-border">
        {provider}
      </span>
    );
  };

  return (
    <AppShell>
      <div className="space-y-6">
        <PageHeader
          title="AI Models"
          description="Browse available active AI models. Mark your favorites to prioritize them in flow and form runners."
        />

        {/* Stats cards */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
          <Card className="p-4 flex items-center gap-3">
            <div className="h-10 w-10 rounded-lg bg-muted border border-border text-foreground flex items-center justify-center shrink-0">
              <Cpu className="h-5 w-5" />
            </div>
            <div>
              <div className="text-2xl font-bold">{stats.total}</div>
              <div className="text-xs text-muted-foreground">Active Models</div>
            </div>
          </Card>

          <Card className="p-4 flex items-center gap-3">
            <div className="h-10 w-10 rounded-lg bg-muted border border-border text-foreground flex items-center justify-center shrink-0">
              <Star className="h-5 w-5 fill-foreground text-foreground" />
            </div>
            <div>
              <div className="text-2xl font-bold">{stats.favCount}</div>
              <div className="text-xs text-muted-foreground">My Favorites</div>
            </div>
          </Card>

          <Card className="p-4 flex items-center gap-3">
            <div className="h-10 w-10 rounded-lg bg-muted border border-border text-foreground flex items-center justify-center shrink-0">
              <FileText className="h-5 w-5" />
            </div>
            <div>
              <div className="text-2xl font-bold">{stats.textCount}</div>
              <div className="text-xs text-muted-foreground">Text Models</div>
            </div>
          </Card>

          <Card className="p-4 flex items-center gap-3">
            <div className="h-10 w-10 rounded-lg bg-muted border border-border text-foreground flex items-center justify-center shrink-0">
              <ImageIcon className="h-5 w-5" />
            </div>
            <div>
              <div className="text-2xl font-bold">{stats.imageCount}</div>
              <div className="text-xs text-muted-foreground">Image Models</div>
            </div>
          </Card>
        </div>

        {/* Filters and search bar */}
        <Card className="p-4 space-y-4">
          <div className="flex flex-col md:flex-row gap-3 items-stretch md:items-center justify-between">
            {/* Search Input */}
            <div className="relative flex-1">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
              <Input
                placeholder="Search models by name, provider, or description..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="pl-9 h-9"
              />
            </div>

            {/* Filter buttons */}
            <div className="flex items-center gap-2 flex-wrap">
              {/* Category Pills */}
              <div className="inline-flex rounded-lg border border-border bg-muted/30 p-0.5 text-xs">
                {(["all", "text", "image", "audio"] as const).map((cat) => (
                  <button
                    key={cat}
                    type="button"
                    onClick={() => setCategoryFilter(cat)}
                    className={`px-3 py-1 rounded-md font-medium capitalize transition-colors ${categoryFilter === cat
                      ? "bg-background text-foreground shadow-xs"
                      : "text-muted-foreground hover:text-foreground"
                      }`}
                  >
                    {cat}
                  </button>
                ))}
              </div>

              {/* Favorites only toggle */}
              <button
                type="button"
                onClick={() => setFavoritesOnly((prev) => !prev)}
                className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg border text-xs font-medium transition-colors ${favoritesOnly
                  ? "bg-foreground text-background border-foreground shadow-xs"
                  : "border-border bg-background text-muted-foreground hover:text-foreground"
                  }`}
              >
                <Star
                  className={`h-3.5 w-3.5 ${favoritesOnly ? "fill-background text-background" : "text-muted-foreground"
                    }`}
                />
                Favorites only
              </button>
            </div>
          </div>
        </Card>

        {/* Models Table */}
        <Card className="overflow-hidden border border-border">
          {loading ? (
            <div className="py-16 text-center text-sm text-muted-foreground">
              <Cpu className="h-8 w-8 animate-pulse mx-auto mb-3 opacity-40" />
              Loading available AI models...
            </div>
          ) : filteredModels.length === 0 ? (
            <div className="py-16 text-center space-y-3">
              <SlidersHorizontal className="h-8 w-8 text-muted-foreground/40 mx-auto" />
              <div className="text-base font-semibold">No models match your filters</div>
              <p className="text-xs text-muted-foreground max-w-sm mx-auto">
                Try adjusting your search query or removing the category/favorites filter.
              </p>
              {(searchQuery || categoryFilter !== "all" || favoritesOnly) && (
                <button
                  type="button"
                  onClick={() => {
                    setSearchQuery("");
                    setCategoryFilter("all");
                    setFavoritesOnly(false);
                  }}
                  className="text-xs text-primary underline hover:opacity-80"
                >
                  Clear all filters
                </button>
              )}
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-left text-sm">
                <thead className="bg-muted/40 text-xs uppercase tracking-wider text-muted-foreground border-b border-border">
                  <tr>
                    <th className="py-3 px-4 min-w-[200px]">Model</th>
                    <th className="py-3 px-4 w-28">Category</th>
                    <th className="py-3 px-4">Description</th>
                    <th className="py-3 px-4 w-44 text-right">Action</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-border">
                  {filteredModels.map((model) => {
                    const favorited = isFavorite(model.id);
                    return (
                      <tr
                        key={model.id}
                        className={`transition-colors hover:bg-muted/30 ${favorited ? "bg-muted/20" : ""
                          }`}
                      >
                        {/* Model name + ID + provider + optional badges */}
                        <td className="py-3 px-4">
                          <div className="flex items-center gap-2">
                            <span className="font-semibold text-foreground">
                              {model.name}
                            </span>
                            {getProviderBadge(model.provider)}
                            {model.recommended && (
                              <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-medium bg-muted text-foreground border border-border">
                                <Sparkles className="h-2.5 w-2.5 text-muted-foreground" />
                                Recommended
                              </span>
                            )}
                          </div>
                          <div className="text-xs text-muted-foreground font-mono mt-0.5">
                            {model.id}
                          </div>
                        </td>

                        {/* Category */}
                        <td className="py-3 px-4">
                          {getCategoryBadge(model.category)}
                        </td>

                        {/* Description */}
                        <td className="py-3 px-4 text-xs text-muted-foreground leading-relaxed max-w-md">
                          {model.description || (
                            <span className="italic opacity-60">
                              No description provided.
                            </span>
                          )}
                        </td>

                        {/* Action column with Add to Favorites button */}
                        <td className="py-3 px-4 text-right">
                          <Button
                            variant={favorited ? "secondary" : "outline"}
                            size="sm"
                            onClick={() => handleToggleFavorite(model)}
                            className={`gap-1.5 text-xs transition-colors cursor-pointer ${favorited
                              ? "bg-foreground text-white border-foreground hover:bg-foreground/90 hover:text-black"
                              : "border-border hover:bg-muted text-foreground"
                              }`}
                          >
                            <Star
                              className={`h-3.5 w-3.5 ${favorited ? "fill-background text-background" : "text-muted-foreground"
                                }`}
                            />
                            <span>{favorited ? "Favorited" : "Add to favorites"}</span>
                          </Button>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}
        </Card>
      </div>
    </AppShell>
  );
}
