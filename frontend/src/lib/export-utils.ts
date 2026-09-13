import { Document, Packer, Paragraph, TextRun, HeadingLevel, Table, TableRow, TableCell, WidthType, BorderStyle } from "docx";
import { toast } from "sonner";

/**
 * Downloads a file in browser
 */
export function downloadFile(filename: string, blob: Blob) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

/**
 * Export text as Markdown (.md)
 */
export function exportToMarkdown(title: string, text: string) {
  try {
    const filename = `${sanitizeFilename(title)}.md`;
    const blob = new Blob([text], { type: "text/markdown;charset=utf-8;" });
    downloadFile(filename, blob);
    toast.success(`Exported as ${filename}`);
  } catch (err: any) {
    toast.error(err.message || "Failed to export Markdown");
  }
}

/**
 * Export text as CSV (.csv)
 * Parses markdown tables if present, otherwise splits content into clean CSV rows.
 */
export function exportToCsv(title: string, text: string) {
  try {
    const filename = `${sanitizeFilename(title)}.csv`;
    const lines = text.split("\n");
    const csvRows: string[][] = [];

    // Check if output contains markdown table syntax
    const tableLines = lines.filter((l) => l.trim().startsWith("|") && l.trim().endsWith("|"));
    if (tableLines.length >= 2) {
      for (const line of tableLines) {
        // Ignore separator line | --- | --- |
        if (line.includes("---")) continue;
        const cells = line
          .split("|")
          .slice(1, -1)
          .map((c) => c.trim());
        csvRows.push(cells);
      }
    } else {
      // Standard text: Header + Content rows
      csvRows.push(["Title", "Content"]);
      // Group text by double newlines or non-empty lines
      const paragraphs = text.split(/\n\n+/).map((p) => p.trim()).filter(Boolean);
      if (paragraphs.length > 0) {
        paragraphs.forEach((p, idx) => {
          csvRows.push([`Section ${idx + 1}`, p]);
        });
      } else {
        csvRows.push([title, text]);
      }
    }

    const csvContent = csvRows
      .map((row) =>
        row
          .map((cell) => {
            const escaped = cell.replace(/"/g, '""');
            return `"${escaped}"`;
          })
          .join(",")
      )
      .join("\n");

    const blob = new Blob(["\uFEFF" + csvContent], { type: "text/csv;charset=utf-8;" });
    downloadFile(filename, blob);
    toast.success(`Exported as ${filename}`);
  } catch (err: any) {
    toast.error(err.message || "Failed to export CSV");
  }
}

/**
 * Export text as formatted DOCX (.docx)
 * Converts markdown formatting (headings, lists, bold/italic, tables) into DOCX elements.
 */
export async function exportToDocx(title: string, text: string) {
  try {
    const filename = `${sanitizeFilename(title)}.docx`;
    const docChildren: (Paragraph | Table)[] = [];

    // Add Document Title Header
    docChildren.push(
      new Paragraph({
        text: title,
        heading: HeadingLevel.TITLE,
        spacing: { after: 200 },
      })
    );

    const lines = text.split("\n");
    let inTable = false;
    let tableBuffer: string[] = [];

    const flushTable = () => {
      if (tableBuffer.length < 2) {
        tableBuffer = [];
        inTable = false;
        return;
      }

      const rows: TableRow[] = [];
      for (const line of tableBuffer) {
        if (line.includes("---")) continue; // skip table divider line
        const cells = line
          .split("|")
          .slice(1, -1)
          .map((c) => c.trim());
        
        rows.push(
          new TableRow({
            children: cells.map(
              (cellText) =>
                new TableCell({
                  children: [new Paragraph({ children: parseInlineText(cellText) })],
                  width: { size: 100 / cells.length, type: WidthType.PERCENTAGE },
                })
            ),
          })
        );
      }

      if (rows.length > 0) {
        docChildren.push(
          new Table({
            rows,
            width: { size: 100, type: WidthType.PERCENTAGE },
            borders: {
              top: { style: BorderStyle.SINGLE, size: 1, color: "CCCCCC" },
              bottom: { style: BorderStyle.SINGLE, size: 1, color: "CCCCCC" },
              left: { style: BorderStyle.SINGLE, size: 1, color: "CCCCCC" },
              right: { style: BorderStyle.SINGLE, size: 1, color: "CCCCCC" },
              insideHorizontal: { style: BorderStyle.SINGLE, size: 1, color: "E5E5E5" },
              insideVertical: { style: BorderStyle.SINGLE, size: 1, color: "E5E5E5" },
            },
          })
        );
        docChildren.push(new Paragraph({ spacing: { after: 120 } }));
      }
      tableBuffer = [];
      inTable = false;
    };

    for (let i = 0; i < lines.length; i++) {
      const line = lines[i];
      const trimmed = line.trim();

      // Check if line is part of a markdown table
      if (trimmed.startsWith("|") && trimmed.endsWith("|")) {
        inTable = true;
        tableBuffer.push(trimmed);
        continue;
      } else if (inTable) {
        flushTable();
      }

      if (!trimmed) {
        docChildren.push(new Paragraph({ spacing: { after: 100 } }));
        continue;
      }

      // Headings
      if (trimmed.startsWith("# ")) {
        docChildren.push(
          new Paragraph({
            text: trimmed.replace(/^#\s+/, ""),
            heading: HeadingLevel.HEADING_1,
            spacing: { before: 240, after: 120 },
          })
        );
      } else if (trimmed.startsWith("## ")) {
        docChildren.push(
          new Paragraph({
            text: trimmed.replace(/^##\s+/, ""),
            heading: HeadingLevel.HEADING_2,
            spacing: { before: 200, after: 100 },
          })
        );
      } else if (trimmed.startsWith("### ")) {
        docChildren.push(
          new Paragraph({
            text: trimmed.replace(/^###\s+/, ""),
            heading: HeadingLevel.HEADING_3,
            spacing: { before: 160, after: 80 },
          })
        );
      } else if (trimmed.startsWith("- ") || trimmed.startsWith("* ")) {
        // Bullet list
        const bulletText = trimmed.replace(/^[-*]\s+/, "");
        docChildren.push(
          new Paragraph({
            children: parseInlineText(bulletText),
            bullet: { level: 0 },
            spacing: { after: 60 },
          })
        );
      } else if (/^\d+\.\s+/.test(trimmed)) {
        // Numbered list item
        const numText = trimmed.replace(/^\d+\.\s+/, "");
        docChildren.push(
          new Paragraph({
            children: parseInlineText(numText),
            spacing: { after: 60 },
          })
        );
      } else {
        // Regular paragraph with inline formatting
        docChildren.push(
          new Paragraph({
            children: parseInlineText(line),
            spacing: { after: 100 },
          })
        );
      }
    }

    if (inTable) {
      flushTable();
    }

    const doc = new Document({
      sections: [
        {
          properties: {},
          children: docChildren,
        },
      ],
    });

    const blob = await Packer.toBlob(doc);
    downloadFile(filename, blob);
    toast.success(`Exported formatted document as ${filename}`);
  } catch (err: any) {
    console.error("Docx export error:", err);
    toast.error(err.message || "Failed to export DOCX document");
  }
}

/**
 * Parses inline bold (**text**) and italic (*text*) markdown into TextRun objects
 */
function parseInlineText(text: string): TextRun[] {
  const runs: TextRun[] = [];
  // Tokenize bold (**...**) and italic (*...*)
  const regex = /(\*\*.*?\*\*|\*.*?\*|`.*?`)/g;
  const parts = text.split(regex);

  for (const part of parts) {
    if (!part) continue;
    if (part.startsWith("**") && part.endsWith("**")) {
      runs.push(new TextRun({ text: part.slice(2, -2), bold: true }));
    } else if (part.startsWith("*") && part.endsWith("*")) {
      runs.push(new TextRun({ text: part.slice(1, -1), italics: true }));
    } else if (part.startsWith("`") && part.endsWith("`")) {
      runs.push(new TextRun({ text: part.slice(1, -1), font: "Courier New" }));
    } else {
      runs.push(new TextRun({ text: part }));
    }
  }

  return runs;
}

function sanitizeFilename(name: string): string {
  return name.toLowerCase().replace(/[^a-z0-9]/gi, "_").replace(/_+/g, "_").slice(0, 50) || "result";
}
