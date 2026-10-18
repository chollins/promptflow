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

export interface ProcessedLlmResult {
  hasJson: boolean;
  readableMarkdown: string;
  rawJsonText?: string;
  imageUrl?: string | null;
  originalText: string;
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
  let foundImageUrl: string | null = null;

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
        foundImageUrl = parsed.url || parsed.image_url || null;
      }
      return {
        hasJson: true,
        readableMarkdown: jsonToReadableMarkdown(parsed),
        rawJsonText: JSON.stringify(parsed, null, 2),
        imageUrl: foundImageUrl,
        originalText: text,
      };
    } catch {
      // not valid JSON, proceed
    }
  }

  // 2. Check for fenced code blocks containing JSON
  // Regex matches ```json ... ``` or ``` ... ```
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
        if (!foundImageUrl && typeof parsed === "object" && parsed !== null) {
          foundImageUrl = parsed.url || parsed.image_url || null;
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
      imageUrl: foundImageUrl,
      originalText: text,
    };
  }

  // 3. Fallback: Search for embedded JSON object or array
  const jsonMatch = trimmed.match(/(\{[\s\S]*\}|\[[\s\S]*\])/);
  if (jsonMatch) {
    try {
      const parsed = JSON.parse(jsonMatch[1]);
      if (typeof parsed === "object" && parsed !== null) {
        foundImageUrl = parsed.url || parsed.image_url || null;
      }
      const readable = jsonToReadableMarkdown(parsed);
      const replacedText = trimmed.replace(jsonMatch[1], readable);
      return {
        hasJson: true,
        readableMarkdown: replacedText.trim(),
        rawJsonText: JSON.stringify(parsed, null, 2),
        imageUrl: foundImageUrl,
        originalText: text,
      };
    } catch {
      // ignore
    }
  }

  // 4. Check for direct image link if no json image found
  if (trimmed.match(/^https?:\/\/.*\.(png|jpg|jpeg|webp)$/i)) {
    foundImageUrl = trimmed;
  }

  // Standard plain text / markdown
  return {
    hasJson: false,
    readableMarkdown: text,
    imageUrl: foundImageUrl,
    originalText: text,
  };
}
