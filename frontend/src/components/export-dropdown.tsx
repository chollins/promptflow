import { Button } from "@/components/ui-kit";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { exportToCsv, exportToDocx, exportToMarkdown } from "@/lib/export-utils";
import { Download, FileCode, FileSpreadsheet, FileText, ChevronDown } from "lucide-react";

interface ExportDropdownProps {
  title: string;
  text: string;
  size?: "sm" | "md";
  variant?: "primary" | "secondary" | "ghost" | "outline";
  className?: string;
}

export function ExportDropdown({
  title,
  text,
  size = "sm",
  variant = "outline",
  className = "",
}: ExportDropdownProps) {
  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button size={size} variant={variant} className={`gap-1.5 text-xs ${className}`}>
          <Download className="h-3.5 w-3.5" />
          Export as
          <ChevronDown className="h-3 w-3 opacity-60 ml-0.5" />
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" className="w-40">
        <DropdownMenuItem
          className="gap-2 text-xs cursor-pointer"
          onClick={() => exportToDocx(title, text)}
        >
          <FileText className="h-3.5 w-3.5 text-blue-600" />
          Word (.docx)
        </DropdownMenuItem>
        <DropdownMenuItem
          className="gap-2 text-xs cursor-pointer"
          onClick={() => exportToMarkdown(title, text)}
        >
          <FileCode className="h-3.5 w-3.5 text-purple-600" />
          Markdown (.md)
        </DropdownMenuItem>
        <DropdownMenuItem
          className="gap-2 text-xs cursor-pointer"
          onClick={() => exportToCsv(title, text)}
        >
          <FileSpreadsheet className="h-3.5 w-3.5 text-emerald-600" />
          CSV (.csv)
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  );
}
