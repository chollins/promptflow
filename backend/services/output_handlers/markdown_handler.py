from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .base import BaseOutputHandler, OutputResult

_PLACEHOLDER_RE = re.compile(r"\{\{([\w\.]+)\}\}")


def transform_json_to_markdown(data: dict[str, Any], template: str) -> str:
    """Transforms structured JSON data into Markdown using template placeholders."""
    result = template
    for match in _PLACEHOLDER_RE.finditer(template):
        key = match.group(1)
        if key in data:
            val = data[key]
            replacement = json.dumps(val, indent=2) if isinstance(val, (dict, list)) else str(val)
            result = result.replace(f"{{{{{key}}}}}", replacement)
    return result


class MarkdownOutputHandler(BaseOutputHandler):
    """Output handler for persisting Markdown artifacts and executing JSON-to-Markdown template transformations."""

    @property
    def name(self) -> str:
        return "markdown"

    def process(
        self,
        raw_output: Any,
        options: dict[str, Any] | None = None,
        save_as: str = "output",
        output_dir: Path | None = None,
    ) -> OutputResult:
        options = options or {}
        try:
            markdown_content = ""
            template = options.get("template") or options.get("transformation_template")

            if template and isinstance(raw_output, (dict, str)):
                data_dict = raw_output
                if isinstance(raw_output, str):
                    try:
                        data_dict = json.loads(raw_output)
                    except json.JSONDecodeError:
                        data_dict = {"text": raw_output}
                if isinstance(data_dict, dict):
                    markdown_content = transform_json_to_markdown(data_dict, template)
                else:
                    markdown_content = str(raw_output)
            elif isinstance(raw_output, str):
                markdown_content = raw_output
            elif isinstance(raw_output, dict):
                # Fallback format dict as markdown codeblock or key-value list
                markdown_content = "\n".join(f"**{k.replace('_', ' ').title()}**: {v}" for k, v in raw_output.items())
            else:
                markdown_content = str(raw_output)

            artifacts: list[str] = []
            if output_dir:
                output_dir.mkdir(parents=True, exist_ok=True)
                file_path = output_dir / f"{save_as}.md"
                file_path.write_text(markdown_content, encoding="utf-8")
                artifacts.append(str(file_path))

            return OutputResult(
                handler_name=self.name,
                success=True,
                output_type="markdown",
                artifacts=artifacts,
                data=markdown_content,
            )
        except Exception as exc:
            return OutputResult(
                handler_name=self.name,
                success=False,
                output_type="markdown",
                error=str(exc),
            )
