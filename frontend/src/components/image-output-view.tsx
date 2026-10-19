import { useState, useId } from "react";
import {
  Download,
  ExternalLink,
  Copy,
  Check,
  Maximize2,
  X,
  Sparkles,
  Image as ImageIcon,
  AlertCircle,
  RefreshCw,
  Info,
} from "lucide-react";
import { toast } from "sonner";
import { cn } from "@/lib/utils";
import type { ImageOutputMetadata } from "@/lib/result-formatter";
import { Button } from "@/components/ui-kit";

interface ImageOutputViewProps {
  imageUrl: string;
  metadata?: ImageOutputMetadata | null;
  className?: string;
  title?: string;
  showDetails?: boolean;
}

export function ImageOutputView({
  imageUrl,
  metadata,
  className = "",
  title = "Generated Image Output",
  showDetails = true,
}: ImageOutputViewProps) {
  const [hasError, setHasError] = useState(false);
  const [isLoaded, setIsLoaded] = useState(false);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [copied, setCopied] = useState(false);
  const [showMetadata, setShowMetadata] = useState(false);
  const uniqueId = useId();

  const isMockUrl =
    metadata?.isMock ||
    imageUrl.includes("mock.leonardo.ai") ||
    imageUrl.includes("mock_");

  const promptText = metadata?.prompt;
  const modelName = metadata?.model;
  const aspectRatio = metadata?.aspectRatio;
  const dimensions = metadata?.dimensions;

  const handleCopyUrl = async () => {
    try {
      await navigator.clipboard.writeText(imageUrl);
      setCopied(true);
      toast.success("Image URL copied to clipboard");
      setTimeout(() => setCopied(false), 2000);
    } catch {
      toast.error("Failed to copy URL");
    }
  };

  const handleDownload = async () => {
    try {
      if (imageUrl.startsWith("data:")) {
        const link = document.createElement("a");
        link.href = imageUrl;
        link.download = `generated-image-${Date.now()}.png`;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        toast.success("Image downloaded");
        return;
      }

      const response = await fetch(imageUrl, { mode: "cors" });
      if (!response.ok) throw new Error("Network response was not ok");
      const blob = await response.blob();
      const blobUrl = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = blobUrl;
      link.download = `generated-image-${Date.now()}.png`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(blobUrl);
      toast.success("Image downloaded");
    } catch {
      // Fallback: Open URL in new window
      window.open(imageUrl, "_blank", "noopener,noreferrer");
      toast.info("Opened image in new tab for direct download");
    }
  };

  return (
    <div className={cn("rounded-xl border border-border bg-card overflow-hidden shadow-sm", className)}>
      {/* Header bar */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-border bg-muted/30">
        <div className="flex items-center gap-2">
          <div className="p-1.5 rounded-md bg-purple-500/10 text-purple-600 dark:text-purple-400">
            <ImageIcon className="h-4 w-4" />
          </div>
          <div>
            <div className="text-xs font-semibold uppercase tracking-wider text-foreground">
              {title}
            </div>
            {modelName && (
              <div className="text-[11px] text-muted-foreground flex items-center gap-1.5">
                <span>Model: {modelName}</span>
                {aspectRatio && <span>• {aspectRatio}</span>}
              </div>
            )}
          </div>
        </div>

        {/* Action buttons */}
        <div className="flex items-center gap-1.5">
          {imageUrl && !hasError && (
            <>
              <Button
                variant="ghost"
                size="sm"
                className="h-8 px-2 text-xs gap-1 text-muted-foreground hover:text-foreground"
                onClick={() => setIsFullscreen(true)}
                title="Expand Fullscreen"
              >
                <Maximize2 className="h-3.5 w-3.5" />
                <span className="hidden sm:inline">Enlarge</span>
              </Button>
              <Button
                variant="ghost"
                size="sm"
                className="h-8 px-2 text-xs gap-1 text-muted-foreground hover:text-foreground"
                onClick={handleDownload}
                title="Download Image"
              >
                <Download className="h-3.5 w-3.5" />
                <span className="hidden sm:inline">Download</span>
              </Button>
            </>
          )}

          {imageUrl && (
            <Button
              variant="ghost"
              size="sm"
              className="h-8 px-2 text-xs gap-1 text-muted-foreground hover:text-foreground"
              onClick={handleCopyUrl}
              title="Copy Image URL"
            >
              {copied ? <Check className="h-3.5 w-3.5 text-green-600" /> : <Copy className="h-3.5 w-3.5" />}
              <span className="hidden sm:inline">{copied ? "Copied" : "Copy URL"}</span>
            </Button>
          )}

          {imageUrl && (
            <Button
              variant="outline"
              size="sm"
              className="h-8 px-2 text-xs gap-1"
              onClick={() => window.open(imageUrl, "_blank", "noopener,noreferrer")}
              title="Open in new tab"
            >
              <ExternalLink className="h-3.5 w-3.5" />
              <span className="hidden sm:inline">Open</span>
            </Button>
          )}
        </div>
      </div>

      {/* Main Image Display */}
      <div className="p-4 flex flex-col items-center justify-center bg-muted/10 relative min-h-[220px]">
        {/* Loading skeleton */}
        {!isLoaded && !hasError && !isMockUrl && (
          <div className="absolute inset-0 flex flex-col items-center justify-center bg-muted/20 animate-pulse gap-2 text-xs text-muted-foreground">
            <Sparkles className="h-6 w-6 text-purple-500 animate-spin" />
            <span>Rendering image output...</span>
          </div>
        )}

        {/* Real Image Rendering */}
        {!hasError && !isMockUrl ? (
          <div className="relative group max-w-full flex justify-center">
            <img
              id={uniqueId}
              src={imageUrl}
              alt={promptText || "Generated Output Preview"}
              onLoad={() => setIsLoaded(true)}
              onError={() => setHasError(true)}
              className={cn(
                "rounded-lg border border-border/80 object-contain max-h-[460px] shadow-sm transition-opacity duration-300",
                isLoaded ? "opacity-100" : "opacity-0"
              )}
            />
            {/* Hover overlay hint */}
            {isLoaded && (
              <button
                type="button"
                onClick={() => setIsFullscreen(true)}
                className="absolute inset-0 bg-black/40 opacity-0 group-hover:opacity-100 transition-opacity rounded-lg flex items-center justify-center gap-2 text-white text-xs font-medium cursor-pointer"
              >
                <Maximize2 className="h-4 w-4" />
                <span>Click to enlarge</span>
              </button>
            )}
          </div>
        ) : (
          /* Mock or Offline/Fallback Card */
          <div className="w-full max-w-xl rounded-xl border border-dashed border-border bg-gradient-to-br from-purple-500/5 via-background to-blue-500/5 p-6 flex flex-col items-center text-center space-y-4">
            <div className="h-12 w-12 rounded-full bg-purple-500/10 flex items-center justify-center text-purple-600 dark:text-purple-400">
              <Sparkles className="h-6 w-6" />
            </div>

            <div className="space-y-1">
              <div className="text-sm font-semibold text-foreground flex items-center justify-center gap-1.5">
                <span>Generated Image Artifact</span>
                {isMockUrl && (
                  <span className="text-[10px] px-2 py-0.5 rounded-full bg-amber-500/10 text-amber-600 border border-amber-500/20 font-medium">
                    Mock Mode
                  </span>
                )}
              </div>
              <p className="text-xs text-muted-foreground max-w-md">
                {isMockUrl
                  ? "This image was generated in test/mock mode without an active Leonardo API key."
                  : "The image output could not be loaded directly into this view (it may be hosted on a restricted CDN or private network)."}
              </p>
            </div>

            {promptText && (
              <div className="w-full text-left bg-muted/40 rounded-lg p-3 border border-border/60 text-xs text-foreground space-y-1">
                <span className="font-semibold text-muted-foreground uppercase tracking-wider text-[10px]">
                  Prompt
                </span>
                <p className="italic leading-relaxed">{promptText}</p>
              </div>
            )}

            <div className="flex flex-wrap items-center justify-center gap-2 pt-1">
              <Button
                variant="outline"
                size="sm"
                onClick={() => window.open(imageUrl, "_blank", "noopener,noreferrer")}
                className="text-xs gap-1.5"
              >
                <ExternalLink className="h-3.5 w-3.5" />
                Open External URL
              </Button>
              <Button
                variant="secondary"
                size="sm"
                onClick={handleCopyUrl}
                className="text-xs gap-1.5"
              >
                {copied ? <Check className="h-3.5 w-3.5 text-green-600" /> : <Copy className="h-3.5 w-3.5" />}
                Copy URL
              </Button>
              {hasError && !isMockUrl && (
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => setHasError(false)}
                  className="text-xs gap-1.5"
                >
                  <RefreshCw className="h-3.5 w-3.5" />
                  Retry Load
                </Button>
              )}
            </div>
          </div>
        )}
      </div>

      {/* Details accordion/footer */}
      {showDetails && (promptText || modelName || aspectRatio || dimensions) && (
        <div className="border-t border-border bg-muted/20 px-4 py-2.5">
          <div className="flex items-center justify-between">
            <button
              type="button"
              onClick={() => setShowMetadata((prev) => !prev)}
              className="text-xs text-muted-foreground hover:text-foreground flex items-center gap-1.5 transition-colors"
            >
              <Info className="h-3.5 w-3.5" />
              <span>{showMetadata ? "Hide Generation Details" : "View Generation Details"}</span>
            </button>
            {aspectRatio && (
              <span className="text-[11px] font-mono px-2 py-0.5 rounded bg-muted border border-border text-muted-foreground">
                Ratio: {aspectRatio}
              </span>
            )}
          </div>

          {showMetadata && (
            <div className="mt-3 pt-3 border-t border-border/60 space-y-2 text-xs">
              {promptText && (
                <div>
                  <span className="font-medium text-foreground">Prompt: </span>
                  <span className="text-muted-foreground">{promptText}</span>
                </div>
              )}
              {metadata?.negativePrompt && (
                <div>
                  <span className="font-medium text-foreground">Negative Prompt: </span>
                  <span className="text-muted-foreground">{metadata.negativePrompt}</span>
                </div>
              )}
              <div className="grid grid-cols-2 sm:grid-cols-3 gap-2 pt-1 text-[11px] text-muted-foreground">
                {modelName && <div>Model: <span className="font-medium text-foreground">{modelName}</span></div>}
                {dimensions && <div>Dimensions: <span className="font-medium text-foreground">{dimensions}</span></div>}
                {metadata?.status && <div>Status: <span className="font-medium text-foreground">{metadata.status}</span></div>}
                {metadata?.generationId && <div className="truncate">Gen ID: <span className="font-mono text-foreground">{metadata.generationId}</span></div>}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Fullscreen Modal / Lightbox */}
      {isFullscreen && (
        <div
          className="fixed inset-0 z-50 bg-black/80 backdrop-blur-sm flex flex-col items-center justify-center p-4"
          onClick={() => setIsFullscreen(false)}
        >
          <div
            className="relative max-w-5xl max-h-[90vh] flex flex-col items-center"
            onClick={(e) => e.stopPropagation()}
          >
            <button
              type="button"
              onClick={() => setIsFullscreen(false)}
              className="absolute -top-10 right-0 p-1.5 rounded-full bg-white/20 hover:bg-white/30 text-white transition-colors"
              title="Close Fullscreen"
            >
              <X className="h-5 w-5" />
            </button>
            <img
              src={imageUrl}
              alt={promptText || "Generated Output Preview"}
              className="max-h-[85vh] max-w-full rounded-lg object-contain shadow-2xl"
            />
            {promptText && (
              <p className="mt-3 text-xs text-white/80 text-center max-w-2xl truncate">
                {promptText}
              </p>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
