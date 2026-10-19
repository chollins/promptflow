/**
 * Utility functions to clean, parse, and format LLM output.
 * Converts raw JSON and markdown code blocks into clean, human-readable markdown.
 */

export function formatKey(key: string): string {
  return key
    .replace(/([a-z])([A-Z])/g, "$1 $2")
    .replace(/[_-]+/g, " ")
    .replace(/^\w/, (c) => c.toUpperCase())
    .trim();
}

/**
 * Recursively converts a JSON value (object, array, primitive) into clean Markdown.
 */
export function jsonToReadableMarkdown(data: any, depth = 0): string {
  if (typeof data === "string") return data;
  if (typeof data !== "object" || data === null) return String(data);

  // Array handling
  if (Array.isArray(data)) {
    if (data.length === 0) return "*(Empty list)*";

    // If array of primitives (strings, numbers)
    const allPrimitives = data.every(
      (item) => typeof item !== "object" || item === null
    );
    if (allPrimitives) {
      return data.map((item) => `- ${item}`).join("\n");
    }

    // Array of objects
    return data
      .map((item, idx) => {
        if (typeof item === "object" && item !== null) {
          // Look for an obvious identifier/name key
          const entries = Object.entries(item);
          const nameEntry = entries.find(([k]) =>
            ["name", "title", "label", "stretch", "exercise", "step", "item"].includes(
              k.toLowerCase()
            )
          );

          const title = nameEntry ? String(nameEntry[1]) : `Item ${idx + 1}`;
          const otherEntries = entries.filter(([k]) => k !== nameEntry?.[0]);

          if (otherEntries.length === 0) {
            return `- **${title}**`;
          }

          const details = otherEntries
            .map(([k, v]) => {
              const label = formatKey(k);
              if (Array.isArray(v)) {
                return `  - **${label}:** ${v.join(", ")}`;
              }
              if (typeof v === "object" && v !== null) {
                return `  - **${label}:** ${JSON.stringify(v)}`;
              }
              return `  - **${label}:** ${v}`;
            })
            .join("\n");

          return `- **${title}**\n${details}`;
        }
        return `- ${item}`;
      })
      .join("\n\n");
  }

  // Object handling
  const lines: string[] = [];
  const entries = Object.entries(data);

  for (const [key, value] of entries) {
    const formattedTitle = formatKey(key);

    if (value === null || value === undefined) {
      lines.push(`**${formattedTitle}:** *(None)*`);
    } else if (Array.isArray(value)) {
      if (value.length === 0) {
        lines.push(`**${formattedTitle}:** *(None)*`);
      } else if (
        value.every((v) => typeof v !== "object" || v === null)
      ) {
        lines.push(`### ${formattedTitle}`);
        lines.push(value.map((v) => `- ${v}`).join("\n"));
      } else {
        lines.push(`### ${formattedTitle}`);
        lines.push(jsonToReadableMarkdown(value, depth + 1));
      }
    } else if (typeof value === "object") {
      lines.push(`### ${formattedTitle}`);
      lines.push(jsonToReadableMarkdown(value, depth + 1));
    } else {
      lines.push(`**${formattedTitle}:** ${value}`);
    }
  }

  return lines.join("\n\n");
}

export interface ImageOutputMetadata {
  url: string;
  prompt?: string;
  negativePrompt?: string;
  model?: string;
  aspectRatio?: string;
  dimensions?: string;
  generationId?: string;
  status?: string;
  isMock?: boolean;
}

export interface ProcessedLlmResult {
  hasJson: boolean;
  readableMarkdown: string;
  rawJsonText?: string;
  imageUrl?: string | null;
  imageMetadata?: ImageOutputMetadata | null;
  originalText: string;
}

export function isLikelyImageUrl(str: string): boolean {
  if (!str || typeof str !== "string") return false;
  const s = str.trim();
  if (
    s.startsWith("http://") ||
    s.startsWith("https://") ||
    s.startsWith("data:image/") ||
    s.startsWith("/outputs/") ||
    s.startsWith("/images/") ||
    s.startsWith("/api/outputs/")
  ) {
    return true;
  }
  return /\.(png|jpg|jpeg|webp|gif|svg)(\?.*)?$/i.test(s);
}

export function extractImageMetadata(obj: any): ImageOutputMetadata | null {
  if (!obj || typeof obj !== "object") return null;

  let url: string | null = null;
  if (typeof obj.url === "string" && isLikelyImageUrl(obj.url)) url = obj.url;
  else if (typeof obj.image_url === "string" && isLikelyImageUrl(obj.image_url)) url = obj.image_url;
  else if (typeof obj.imageUrl === "string" && isLikelyImageUrl(obj.imageUrl)) url = obj.imageUrl;
  else if (typeof obj.image === "string" && isLikelyImageUrl(obj.image)) url = obj.image;
  else if (typeof obj.src === "string" && isLikelyImageUrl(obj.src)) url = obj.src;

  // DALL-E format: { data: [{ url: "..." }] }
  if (!url && Array.isArray(obj.data) && obj.data.length > 0) {
    const item = obj.data[0];
    if (typeof item?.url === "string" && isLikelyImageUrl(item.url)) {
      url = item.url;
    }
  }

  // Leonardo format: { images: [{ url: "..." }] } or { images: ["..."] }
  if (!url && Array.isArray(obj.images) && obj.images.length > 0) {
    const item = obj.images[0];
    if (typeof item === "string" && isLikelyImageUrl(item)) {
      url = item;
    } else if (typeof item?.url === "string" && isLikelyImageUrl(item.url)) {
      url = item.url;
    }
  }

  // Nested output or result
  if (!url && obj.output && typeof obj.output === "object") {
    const nested = extractImageMetadata(obj.output);
    if (nested) return nested;
  }
  if (!url && obj.result && typeof obj.result === "object") {
    const nested = extractImageMetadata(obj.result);
    if (nested) return nested;
  }

  // Check artifacts array
  if (!url && Array.isArray(obj.artifacts)) {
    const art = obj.artifacts.find((a: any) => typeof a === "string" && isLikelyImageUrl(a));
    if (art) url = art;
  }

  const isImageType = obj.type === "image" || obj.output_type === "image" || !!url;
  if (!url && !isImageType) {
    return null;
  }

  return {
    url: url || "",
    prompt: obj.prompt || obj.clean_prompt || obj.description || undefined,
    negativePrompt: obj.negative_prompt || obj.negativePrompt || undefined,
    model: obj.model || undefined,
    aspectRatio: obj.aspect_ratio || obj.aspectRatio || obj.ar || undefined,
    dimensions: obj.dimensions || (obj.width && obj.height ? `${obj.width}x${obj.height}` : undefined),
    generationId: obj.generation_id || obj.generationId || undefined,
    status: obj.status || undefined,
    isMock: typeof url === "string" && (url.includes("mock.leonardo.ai") || url.includes("mock_")),
  };
}

/**
 * Extracts JSON from LLM text (including markdown code blocks ```json ... ```)
 * and returns both human-readable markdown and raw JSON for toggling.
 */
export function processLlmResult(text: string): ProcessedLlmResult {
  if (!text || typeof text !== "string") {
    return {
      hasJson: false,
      readableMarkdown: "",
      originalText: text || "",
    };
  }

  const trimmed = text.trim();
  let foundMetadata: ImageOutputMetadata | null = null;

  // 1. Direct raw JSON check (with or without markdown code fences ```json ... ```)
  let strippedFences = trimmed;
  if (/^```(?:json)?\s*/i.test(strippedFences)) {
    strippedFences = strippedFences.replace(/^```(?:json)?\s*/i, "");
    if (/```\s*$/i.test(strippedFences)) {
      strippedFences = strippedFences.replace(/```\s*$/i, "");
    }
    strippedFences = strippedFences.trim();
  }

  if (
    (strippedFences.startsWith("{") && strippedFences.endsWith("}")) ||
    (strippedFences.startsWith("[") && strippedFences.endsWith("]"))
  ) {
    try {
      const parsed = JSON.parse(strippedFences);
      if (typeof parsed === "object" && parsed !== null) {
        foundMetadata = extractImageMetadata(parsed);
      }
      return {
        hasJson: true,
        readableMarkdown: jsonToReadableMarkdown(parsed),
        rawJsonText: JSON.stringify(parsed, null, 2),
        imageUrl: foundMetadata?.url || null,
        imageMetadata: foundMetadata,
        originalText: text,
      };
    } catch {
      // not valid JSON, proceed
    }
  }

  // 2. Check for fenced code blocks containing JSON
  const codeBlockRegex = /```(?:json)?\s*([\s\S]*?)\s*```/gi;
  let match: RegExpExecArray | null;
  const matches: Array<{ full: string; content: string; parsed: any }> = [];

  while ((match = codeBlockRegex.exec(text)) !== null) {
    const rawContent = match[1].trim();
    if (
      (rawContent.startsWith("{") && rawContent.endsWith("}")) ||
      (rawContent.startsWith("[") && rawContent.endsWith("]"))
    ) {
      try {
        const parsed = JSON.parse(rawContent);
        matches.push({ full: match[0], content: rawContent, parsed });
        if (!foundMetadata && typeof parsed === "object" && parsed !== null) {
          foundMetadata = extractImageMetadata(parsed);
        }
      } catch {
        // Not valid JSON in this block
      }
    }
  }

  if (matches.length > 0) {
    let replacedText = text;
    for (const m of matches) {
      const readable = jsonToReadableMarkdown(m.parsed);
      replacedText = replacedText.replace(m.full, readable);
    }

    return {
      hasJson: true,
      readableMarkdown: replacedText.trim(),
      rawJsonText:
        matches.length === 1
          ? JSON.stringify(matches[0].parsed, null, 2)
          : text,
      imageUrl: foundMetadata?.url || null,
      imageMetadata: foundMetadata,
      originalText: text,
    };
  }

  // 3. Fallback: Search for embedded JSON object or array
  const jsonMatch = trimmed.match(/(\{[\s\S]*\}|\[[\s\S]*\])/);
  if (jsonMatch) {
    try {
      const parsed = JSON.parse(jsonMatch[1]);
      if (typeof parsed === "object" && parsed !== null) {
        foundMetadata = extractImageMetadata(parsed);
      }
      const readable = jsonToReadableMarkdown(parsed);
      const replacedText = trimmed.replace(jsonMatch[1], readable);
      return {
        hasJson: true,
        readableMarkdown: replacedText.trim(),
        rawJsonText: JSON.stringify(parsed, null, 2),
        imageUrl: foundMetadata?.url || null,
        imageMetadata: foundMetadata,
        originalText: text,
      };
    } catch {
      // ignore
    }
  }

  // 4. Check for markdown image: ![alt](url)
  const mdMatch = trimmed.match(/!\[(.*?)\]\((https?:\/\/[^\s\)]+|data:image\/[^\s\)]+|\/[^\s\)]+)\)/);
  if (mdMatch) {
    foundMetadata = {
      url: mdMatch[2],
      prompt: mdMatch[1] || undefined,
      isMock: mdMatch[2].includes("mock.leonardo.ai"),
    };
    return {
      hasJson: false,
      readableMarkdown: text,
      imageUrl: foundMetadata.url,
      imageMetadata: foundMetadata,
      originalText: text,
    };
  }

  // 5. Check for direct image link in text
  const bareMatch = trimmed.match(/(https?:\/\/[^\s"']+\.(?:png|jpg|jpeg|webp|gif|svg)(?:\?[^\s"']*)?)/i);
  if (bareMatch) {
    foundMetadata = {
      url: bareMatch[1],
      isMock: bareMatch[1].includes("mock.leonardo.ai"),
    };
    return {
      hasJson: false,
      readableMarkdown: text,
      imageUrl: foundMetadata.url,
      imageMetadata: foundMetadata,
      originalText: text,
    };
  }

  // Standard plain text / markdown
  return {
    hasJson: false,
    readableMarkdown: text,
    imageUrl: null,
    imageMetadata: null,
    originalText: text,
  };
}
