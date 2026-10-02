from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .base import BaseOutputHandler, OutputResult


class JSONOutputHandler(BaseOutputHandler):
    """Output handler for persisting structured JSON artifacts."""

    @property
    def name(self) -> str:
        return "json"

    def process(
        self,
        raw_output: Any,
        options: dict[str, Any] | None = None,
        save_as: str = "output",
        output_dir: Path | None = None,
    ) -> OutputResult:
        options = options or {}
        try:
            if isinstance(raw_output, str):
                try:
                    parsed_data = json.loads(raw_output)
                except json.JSONDecodeError:
                    parsed_data = {"result": raw_output}
            else:
                parsed_data = raw_output

            formatted_json = json.dumps(parsed_data, indent=2)

            artifacts: list[str] = []
            if output_dir:
                output_dir.mkdir(parents=True, exist_ok=True)
                file_path = output_dir / f"{save_as}.json"
                file_path.write_text(formatted_json, encoding="utf-8")
                artifacts.append(str(file_path))

            return OutputResult(
                handler_name=self.name,
                success=True,
                output_type="json",
                artifacts=artifacts,
                data=parsed_data,
            )
        except Exception as exc:
            return OutputResult(
                handler_name=self.name,
                success=False,
                output_type="json",
                error=str(exc),
            )
