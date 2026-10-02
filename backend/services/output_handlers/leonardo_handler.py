from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .base import BaseOutputHandler, OutputResult
from ..adapters.leonardo_adapter import LeonardoAdapter, LeonardoAdapterError


class LeonardoOutputHandler(BaseOutputHandler):
    """Output handler for generating image artifacts via Leonardo.AI adapter."""

    def __init__(self, adapter: LeonardoAdapter | None = None):
        self.adapter = adapter or LeonardoAdapter()

    @property
    def name(self) -> str:
        return "leonardo"

    def process(
        self,
        raw_output: Any,
        options: dict[str, Any] | None = None,
        save_as: str = "output",
        output_dir: Path | None = None,
    ) -> OutputResult:
        options = options or {}
        try:
            image_spec: dict[str, Any] = {}
            if isinstance(raw_output, str):
                try:
                    image_spec = json.loads(raw_output)
                except json.JSONDecodeError:
                    image_spec = {"type": "image", "prompt": raw_output}
            elif isinstance(raw_output, dict):
                image_spec = raw_output

            if not isinstance(image_spec, dict):
                raise LeonardoAdapterError("Expected dict or JSON string for image generation payload.")

            result = self.adapter.generate_image(
                image_spec,
                output_dir=output_dir,
                save_as=save_as,
            )

            return OutputResult(
                handler_name=self.name,
                success=True,
                output_type="image",
                artifacts=result.get("artifacts", []),
                data=result,
            )
        except Exception as exc:
            return OutputResult(
                handler_name=self.name,
                success=False,
                output_type="image",
                error=str(exc),
            )
