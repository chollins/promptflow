import { useEffect, useMemo, useState } from "react";
import {
    Cpu,
    Sparkles,
    Plus,
    PencilLine,
    Trash2,
    Search,
    CheckCircle2,
    XCircle,
    Volume2,
    Image as ImageIcon,
    FileText,
    Star,
    SlidersHorizontal,
} from "lucide-react";
import { AppShell, PageHeader } from "@/components/app-shell";
import { Button, Card, Input } from "@/components/ui-kit";
import { Textarea } from "@/components/ui/textarea";
import {
    Dialog,
    DialogContent,
    DialogHeader,
    DialogTitle,
    DialogDescription,
} from "@/components/ui/dialog";
import {
    AlertDialog,
    AlertDialogAction,
    AlertDialogCancel,
    AlertDialogContent,
    AlertDialogDescription,
    AlertDialogFooter,
    AlertDialogHeader,
    AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import { apiDelete, apiGet, apiPost, apiPut } from "@/lib/api";
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

type ModelFormState = {
    id: string;
    name: string;
    provider: string;
    category: string;
    description: string;
    is_active: boolean;
    is_recommended: boolean;
};

const EMPTY_MODEL_FORM: ModelFormState = {
    id: "",
    name: "",
    provider: "openai",
    category: "text",
    description: "",
    is_active: true,
    is_recommended: false,
};

export default function AdminModelsPage() {
    const [models, setModels] = useState<AIModelItem[]>([]);
    const [loading, setLoading] = useState(true);
    const [saving, setSaving] = useState(false);
    const [modalOpen, setModalOpen] = useState(false);
    const [editingId, setEditingId] = useState<string | null>(null);
    const [form, setForm] = useState<ModelFormState>(EMPTY_MODEL_FORM);
    const [deleteId, setDeleteId] = useState<string | null>(null);

    // Filters
    const [searchQuery, setSearchQuery] = useState("");
    const [categoryFilter, setCategoryFilter] = useState<string>("all");
    const [statusFilter, setStatusFilter] = useState<string>("all");

    const fetchModels = () => {
        setLoading(true);
        apiGet<{ models: AIModelItem[] }>("/admin/models")
            .then((data) => {
                setModels(data.models || []);
            })
            .catch((err) => {
                toast.error(`Failed to load AI models: ${err.message || err}`);
            })
            .finally(() => setLoading(false));
    };

    useEffect(() => {
        fetchModels();
    }, []);

    const filteredModels = useMemo(() => {
        return models.filter((item) => {
            const matchesSearch =
                item.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
                item.id.toLowerCase().includes(searchQuery.toLowerCase()) ||
                item.provider.toLowerCase().includes(searchQuery.toLowerCase());

            const matchesCategory =
                categoryFilter === "all" || item.category === categoryFilter;

            const matchesStatus =
                statusFilter === "all" ||
                (statusFilter === "active" && item.is_active) ||
                (statusFilter === "inactive" && !item.is_active);

            return matchesSearch && matchesCategory && matchesStatus;
        });
    }, [models, searchQuery, categoryFilter, statusFilter]);

    const activeCount = useMemo(() => models.filter((m) => m.is_active).length, [models]);
    const textCount = useMemo(() => models.filter((m) => m.category === "text").length, [models]);
    const imageCount = useMemo(() => models.filter((m) => m.category === "image").length, [models]);
    const audioCount = useMemo(() => models.filter((m) => m.category === "audio").length, [models]);

    const handleOpenCreate = () => {
        setEditingId(null);
        setForm(EMPTY_MODEL_FORM);
        setModalOpen(true);
    };

    const handleOpenEdit = (item: AIModelItem) => {
        setEditingId(item.id);
        setForm({
            id: item.id,
            name: item.name,
            provider: item.provider,
            category: item.category,
            description: item.description || "",
            is_active: item.is_active,
            is_recommended: !!item.recommended,
        });
        setModalOpen(true);
    };

    const handleToggleStatus = async (item: AIModelItem) => {
        try {
            await apiPut(`/admin/models/${item.id}`, { is_active: !item.is_active });
            toast.success(
                `Model '${item.name}' status set to ${!item.is_active ? "Active" : "Inactive"}`
            );
            fetchModels();
        } catch (err: any) {
            toast.error(`Failed to update status: ${err.message || err}`);
        }
    };

    const handleSave = async () => {
        if (!form.id.trim()) {
            toast.error("Model ID is required.");
            return;
        }
        if (!form.name.trim()) {
            toast.error("Model Name is required.");
            return;
        }

        setSaving(true);
        try {
            if (editingId) {
                await apiPut(`/admin/models/${editingId}`, form);
                toast.success(`Model '${form.name}' updated successfully!`);
            } else {
                await apiPost("/admin/models", form);
                toast.success(`Model '${form.name}' created successfully!`);
            }
            setModalOpen(false);
            fetchModels();
        } catch (err: any) {
            toast.error(`Failed to save model: ${err.message || err}`);
        } finally {
            setSaving(false);
        }
    };

    const handleDelete = async () => {
        if (!deleteId) return;
        try {
            await apiDelete(`/admin/models/${deleteId}`);
            toast.success("Model removed successfully.");
            setDeleteId(null);
            fetchModels();
        } catch (err: any) {
            toast.error(`Failed to delete model: ${err.message || err}`);
        }
    };

    const getCategoryBadge = (cat: string) => {
        switch (cat) {
            case "text":
                return (
                    <span className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-md text-xs font-medium bg-blue-50 text-blue-700 border border-blue-200">
                        <FileText className="w-3.5 h-3.5" /> Text LLM
                    </span>
                );
            case "image":
                return (
                    <span className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-md text-xs font-medium bg-purple-50 text-purple-700 border border-purple-200">
                        <ImageIcon className="w-3.5 h-3.5" /> Image Model
                    </span>
                );
            case "audio":
                return (
                    <span className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-md text-xs font-medium bg-amber-50 text-amber-700 border border-amber-200">
                        <Volume2 className="w-3.5 h-3.5" /> Audio / Voice
                    </span>
                );
            default:
                return (
                    <span className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-md text-xs font-medium bg-zinc-100 text-zinc-700 border border-zinc-200">
                        <Sparkles className="w-3.5 h-3.5" /> {cat}
                    </span>
                );
        }
    };

    return (
        <AppShell>
            <div className="space-y-6 max-w-7xl mx-auto p-6 bg-zinc-50/50 min-h-screen">
                <PageHeader
                    title="AI Models Registry"
                    description="Manage registered LLMs, image generation models, and voice audio models. Control model availability and active status."
                    actions={
                        <Button
                            variant="primary"
                            onClick={handleOpenCreate}
                            className="gap-2 bg-black text-white hover:bg-zinc-800 shadow-sm border border-black font-medium"
                        >
                            <Plus className="h-4 w-4" /> Add AI Model
                        </Button>
                    }
                />

                {/* Stats Summary Bar - Clean White Theme */}
                <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                    <Card className="p-4 bg-white border-zinc-200/80 shadow-sm">
                        <div className="flex items-center justify-between">
                            <div>
                                <p className="text-xs font-semibold text-zinc-500 uppercase tracking-wider">Total Models</p>
                                <p className="text-2xl font-bold text-zinc-900 mt-1">{models.length}</p>
                            </div>
                            <div className="p-3 bg-zinc-100 text-zinc-900 rounded-xl border border-zinc-200">
                                <Cpu className="w-5 h-5" />
                            </div>
                        </div>
                    </Card>

                    <Card className="p-4 bg-white border-zinc-200/80 shadow-sm">
                        <div className="flex items-center justify-between">
                            <div>
                                <p className="text-xs font-semibold text-zinc-500 uppercase tracking-wider">Active Status</p>
                                <p className="text-2xl font-bold text-emerald-600 mt-1">{activeCount} / {models.length}</p>
                            </div>
                            <div className="p-3 bg-emerald-50 text-emerald-600 rounded-xl border border-emerald-100">
                                <CheckCircle2 className="w-5 h-5" />
                            </div>
                        </div>
                    </Card>

                    <Card className="p-4 bg-white border-zinc-200/80 shadow-sm">
                        <div className="flex items-center justify-between">
                            <div>
                                <p className="text-xs font-semibold text-zinc-500 uppercase tracking-wider">Generative LLMs</p>
                                <p className="text-2xl font-bold text-zinc-900 mt-1">{textCount}</p>
                            </div>
                            <div className="p-3 bg-blue-50 text-blue-600 rounded-xl border border-blue-100">
                                <FileText className="w-5 h-5" />
                            </div>
                        </div>
                    </Card>

                    <Card className="p-4 bg-white border-zinc-200/80 shadow-sm">
                        <div className="flex items-center justify-between">
                            <div>
                                <p className="text-xs font-semibold text-zinc-500 uppercase tracking-wider">Image & Audio</p>
                                <p className="text-2xl font-bold text-purple-600 mt-1">{imageCount + audioCount}</p>
                            </div>
                            <div className="p-3 bg-purple-50 text-purple-600 rounded-xl border border-purple-100">
                                <Sparkles className="w-5 h-5" />
                            </div>
                        </div>
                    </Card>
                </div>

                {/* Filter Controls Bar */}
                <Card className="p-4 bg-white border-zinc-200/80 shadow-sm space-y-4">
                    <div className="flex flex-col md:flex-row items-center justify-between gap-4">
                        {/* Search Input */}
                        <div className="relative w-full md:w-80">
                            <Search className="absolute left-3 top-2.5 h-4 w-4 text-zinc-400" />
                            <Input
                                placeholder="Search models or providers..."
                                value={searchQuery}
                                onChange={(e) => setSearchQuery(e.target.value)}
                                className="pl-9 bg-white border-zinc-200 text-zinc-900 text-sm focus:border-black focus:ring-black"
                            />
                        </div>

                        {/* Category Tabs */}
                        <div className="flex items-center gap-1 bg-zinc-100 p-1 rounded-lg border border-zinc-200/60 text-xs">
                            <span className="text-zinc-500 px-2 font-medium">Category:</span>
                            {(["all", "text", "image", "audio"] as const).map((cat) => (
                                <button
                                    key={cat}
                                    onClick={() => setCategoryFilter(cat)}
                                    className={`px-3 py-1.5 rounded-md font-medium capitalize transition-all ${categoryFilter === cat
                                            ? "bg-black text-white shadow-xs"
                                            : "text-zinc-600 hover:text-zinc-900 hover:bg-zinc-200/60"
                                        }`}
                                >
                                    {cat}
                                </button>
                            ))}
                        </div>

                        {/* Status Tabs */}
                        <div className="flex items-center gap-1 bg-zinc-100 p-1 rounded-lg border border-zinc-200/60 text-xs">
                            <span className="text-zinc-500 px-2 font-medium">Status:</span>
                            {(["all", "active", "inactive"] as const).map((st) => (
                                <button
                                    key={st}
                                    onClick={() => setStatusFilter(st)}
                                    className={`px-3 py-1.5 rounded-md font-medium capitalize transition-all ${statusFilter === st
                                            ? "bg-black text-white shadow-xs"
                                            : "text-zinc-600 hover:text-zinc-900 hover:bg-zinc-200/60"
                                        }`}
                                >
                                    {st}
                                </button>
                            ))}
                        </div>

                        {/* Quick Add Model Button */}
                        {/* <Button
              variant="primary"
              onClick={handleOpenCreate}
              className="gap-2 bg-black text-white hover:bg-zinc-800 shadow-sm border border-black font-medium text-xs h-9 px-3"
            >
              <Plus className="h-4 w-4" /> Add Model
            </Button> */}
                    </div>
                </Card>

                {/* Models Table */}
                <Card className="border-zinc-200/80 bg-white overflow-hidden shadow-sm">
                    {loading ? (
                        <div className="p-12 text-center text-zinc-500 flex flex-col items-center gap-3">
                            <span className="h-6 w-6 animate-spin rounded-full border-2 border-black border-t-transparent" />
                            <span>Loading AI Model Registry...</span>
                        </div>
                    ) : filteredModels.length === 0 ? (
                        <div className="p-12 text-center text-zinc-500">
                            <Cpu className="h-10 w-10 mx-auto mb-3 text-zinc-400" />
                            <p className="font-semibold text-zinc-800">No AI models found</p>
                            <p className="text-xs text-zinc-500 mt-1 mb-4">Try adjusting your search query or filters, or add a new model.</p>
                            <Button
                                variant="primary"
                                onClick={handleOpenCreate}
                                className="gap-2 bg-black text-white hover:bg-zinc-800 text-xs inline-flex"
                            >
                                <Plus className="h-4 w-4" /> Add AI Model
                            </Button>
                        </div>
                    ) : (
                        <div className="overflow-x-auto">
                            <table className="w-full text-left text-sm text-zinc-700">
                                <thead className="bg-zinc-50 text-xs font-semibold uppercase tracking-wider text-zinc-500 border-b border-zinc-200">
                                    <tr>
                                        <th className="px-6 py-4">Model Name & ID</th>
                                        <th className="px-6 py-4">Category</th>
                                        <th className="px-6 py-4">Provider</th>
                                        <th className="px-6 py-4">Status</th>
                                        <th className="px-6 py-4 text-right">Actions</th>
                                    </tr>
                                </thead>
                                <tbody className="divide-y divide-zinc-200/60">
                                    {filteredModels.map((item) => (
                                        <tr key={item.id} className="hover:bg-zinc-50/80 transition-colors">
                                            {/* Name & ID */}
                                            <td className="px-6 py-4">
                                                <div className="flex items-center gap-3">
                                                    <div className="p-2.5 rounded-lg bg-zinc-100 border border-zinc-200 text-zinc-900">
                                                        <Cpu className="w-4 h-4" />
                                                    </div>
                                                    <div>
                                                        <div className="flex items-center gap-2 font-semibold text-zinc-900">
                                                            {item.name}
                                                            {item.recommended && (
                                                                <span className="inline-flex items-center gap-1 text-[10px] px-2 py-0.5 rounded-full font-medium bg-amber-50 text-amber-800 border border-amber-200">
                                                                    <Star className="w-2.5 h-2.5 fill-amber-500 text-amber-500" /> Rec
                                                                </span>
                                                            )}
                                                        </div>
                                                        <div className="text-xs font-mono text-zinc-500 mt-0.5">{item.id}</div>
                                                        {item.description && (
                                                            <div className="text-xs text-zinc-500 mt-1 line-clamp-1 max-w-md">
                                                                {item.description}
                                                            </div>
                                                        )}
                                                    </div>
                                                </div>
                                            </td>

                                            {/* Category */}
                                            <td className="px-6 py-4">
                                                {getCategoryBadge(item.category)}
                                            </td>

                                            {/* Provider */}
                                            <td className="px-6 py-4">
                                                <span className="font-mono text-xs text-zinc-800 capitalize bg-zinc-100 px-2.5 py-1 rounded border border-zinc-200">
                                                    {item.provider}
                                                </span>
                                            </td>

                                            {/* Status Toggle Badge */}
                                            <td className="px-6 py-4">
                                                <button
                                                    onClick={() => handleToggleStatus(item)}
                                                    className={`inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-semibold transition-all cursor-pointer ${item.is_active
                                                            ? "bg-emerald-50 text-emerald-700 border border-emerald-200 hover:bg-emerald-100"
                                                            : "bg-zinc-100 text-zinc-500 border border-zinc-300 hover:bg-zinc-200"
                                                        }`}
                                                >
                                                    {item.is_active ? (
                                                        <>
                                                            <CheckCircle2 className="w-3.5 h-3.5 text-emerald-600" /> Active
                                                        </>
                                                    ) : (
                                                        <>
                                                            <XCircle className="w-3.5 h-3.5 text-zinc-400" /> Inactive
                                                        </>
                                                    )}
                                                </button>
                                            </td>

                                            {/* Actions */}
                                            <td className="px-6 py-4 text-right">
                                                <div className="flex items-center justify-end gap-2">
                                                    {/* Edit Button */}
                                                    <Button
                                                        variant="ghost"
                                                        size="sm"
                                                        onClick={() => handleOpenEdit(item)}
                                                        className="h-8 px-2.5 text-zinc-600 hover:text-zinc-900 hover:bg-zinc-100"
                                                        title="Edit Model"
                                                    >
                                                        <PencilLine className="h-4 w-4" />
                                                    </Button>

                                                    {/* Delete Button */}
                                                    <Button
                                                        variant="ghost"
                                                        size="sm"
                                                        onClick={() => setDeleteId(item.id)}
                                                        className="h-8 px-2.5 text-red-600 hover:text-red-700 hover:bg-red-50"
                                                        title="Delete Model"
                                                    >
                                                        <Trash2 className="h-4 w-4" />
                                                    </Button>
                                                </div>
                                            </td>
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>
                    )}
                </Card>

                {/* Create / Edit Model Dialog - Clean White */}
                <Dialog open={modalOpen} onOpenChange={setModalOpen}>
                    <DialogContent className="max-w-lg bg-white border-zinc-200 text-zinc-900 shadow-xl">
                        <DialogHeader>
                            <DialogTitle className="flex items-center gap-2 text-xl font-bold text-zinc-900">
                                <Cpu className="w-5 h-5 text-black" />
                                {editingId ? "Edit AI Model" : "Create New AI Model"}
                            </DialogTitle>
                            <DialogDescription className="text-zinc-500 text-xs">
                                Configure provider details, active status, and model metadata for runtime selection.
                            </DialogDescription>
                        </DialogHeader>

                        <div className="space-y-4 py-3 text-sm">
                            {/* Model ID */}
                            <div>
                                <label className="block text-xs font-semibold text-zinc-700 mb-1">
                                    Model ID <span className="text-red-500">*</span>
                                </label>
                                <Input
                                    placeholder="e.g. gpt-4o-mini, claude-3-5-sonnet, leonardo-phoenix"
                                    value={form.id}
                                    disabled={!!editingId}
                                    onChange={(e) => setForm({ ...form, id: e.target.value })}
                                    className="bg-zinc-50 border-zinc-200 font-mono text-xs text-zinc-900 focus:border-black focus:ring-black"
                                />
                            </div>

                            {/* Display Name */}
                            <div>
                                <label className="block text-xs font-semibold text-zinc-700 mb-1">
                                    Display Name <span className="text-red-500">*</span>
                                </label>
                                <Input
                                    placeholder="e.g. GPT-4o Mini, Claude 3.5 Sonnet"
                                    value={form.name}
                                    onChange={(e) => setForm({ ...form, name: e.target.value })}
                                    className="bg-white border-zinc-200 text-sm text-zinc-900 focus:border-black focus:ring-black"
                                />
                            </div>

                            {/* Provider & Category row */}
                            <div className="grid grid-cols-2 gap-4">
                                <div>
                                    <label className="block text-xs font-semibold text-zinc-700 mb-1">Provider</label>
                                    <select
                                        value={form.provider}
                                        onChange={(e) => setForm({ ...form, provider: e.target.value })}
                                        className="w-full h-10 px-3 rounded-lg bg-white border border-zinc-200 text-xs text-zinc-800 focus:outline-none focus:ring-1 focus:ring-black"
                                    >
                                        <option value="openai">OpenAI</option>
                                        <option value="anthropic">Anthropic</option>
                                        <option value="google">Google</option>
                                        <option value="leonardo">Leonardo.AI</option>
                                        <option value="elevenlabs">ElevenLabs</option>
                                        <option value="stability">Stability AI</option>
                                        <option value="custom">Custom Provider</option>
                                    </select>
                                </div>

                                <div>
                                    <label className="block text-xs font-semibold text-zinc-700 mb-1">Category</label>
                                    <select
                                        value={form.category}
                                        onChange={(e) => setForm({ ...form, category: e.target.value })}
                                        className="w-full h-10 px-3 rounded-lg bg-white border border-zinc-200 text-xs text-zinc-800 focus:outline-none focus:ring-1 focus:ring-black"
                                    >
                                        <option value="text">Text & Generative LLM</option>
                                        <option value="image">Picture & Image Model</option>
                                        <option value="audio">Audio & Speech Synthesis</option>
                                    </select>
                                </div>
                            </div>

                            {/* Description */}
                            <div>
                                <label className="block text-xs font-semibold text-zinc-700 mb-1">Description</label>
                                <Textarea
                                    placeholder="Brief description of model capabilities and use-cases..."
                                    value={form.description}
                                    onChange={(e) => setForm({ ...form, description: e.target.value })}
                                    className="bg-white border-zinc-200 text-xs text-zinc-900 min-h-[80px] focus:border-black focus:ring-black"
                                />
                            </div>

                            {/* Checkboxes for Status and Recommended */}
                            <div className="flex items-center justify-between p-3 rounded-xl bg-zinc-50 border border-zinc-200">
                                <label className="flex items-center gap-2 text-xs font-semibold text-zinc-800 cursor-pointer">
                                    <input
                                        type="checkbox"
                                        checked={form.is_active}
                                        onChange={(e) => setForm({ ...form, is_active: e.target.checked })}
                                        className="rounded border-zinc-300 text-black focus:ring-black h-4 w-4"
                                    />
                                    <span>Active Status (Available in UI)</span>
                                </label>

                                <label className="flex items-center gap-2 text-xs font-semibold text-amber-800 cursor-pointer">
                                    <input
                                        type="checkbox"
                                        checked={form.is_recommended}
                                        onChange={(e) => setForm({ ...form, is_recommended: e.target.checked })}
                                        className="rounded border-zinc-300 text-amber-600 focus:ring-amber-500 h-4 w-4"
                                    />
                                    <span>Recommended Badge</span>
                                </label>
                            </div>
                        </div>

                        <div className="flex items-center justify-end gap-2 pt-3 border-t border-zinc-200">
                            <Button
                                variant="ghost"
                                onClick={() => setModalOpen(false)}
                                className="text-zinc-600 hover:text-zinc-900 hover:bg-zinc-100"
                            >
                                Cancel
                            </Button>
                            <Button
                                variant="primary"
                                onClick={handleSave}
                                disabled={saving}
                                className="bg-black hover:bg-zinc-800 text-white gap-2 font-medium"
                            >
                                {saving && (
                                    <span className="h-4 w-4 animate-spin rounded-full border-2 border-white border-t-transparent" />
                                )}
                                {editingId ? "Save Changes" : "Create Model"}
                            </Button>
                        </div>
                    </DialogContent>
                </Dialog>

                {/* Delete Confirmation Modal */}
                <AlertDialog open={!!deleteId} onOpenChange={(open) => !open && setDeleteId(null)}>
                    <AlertDialogContent className="bg-white border-zinc-200 text-zinc-900 shadow-xl">
                        <AlertDialogHeader>
                            <AlertDialogTitle className="text-red-600">Delete AI Model?</AlertDialogTitle>
                            <AlertDialogDescription className="text-zinc-600 text-xs">
                                This action will remove model ID <span className="font-mono text-zinc-900 font-semibold">{deleteId}</span> from the active model registry catalog.
                            </AlertDialogDescription>
                        </AlertDialogHeader>
                        <AlertDialogFooter>
                            <AlertDialogCancel className="border-zinc-200 bg-zinc-100 text-zinc-700 hover:bg-zinc-200">
                                Cancel
                            </AlertDialogCancel>
                            <AlertDialogAction
                                onClick={handleDelete}
                                className="bg-red-600 hover:bg-red-700 text-white"
                            >
                                Delete Model
                            </AlertDialogAction>
                        </AlertDialogFooter>
                    </AlertDialogContent>
                </AlertDialog>
            </div>
        </AppShell>
    );
}
