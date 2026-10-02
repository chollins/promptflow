from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from .base import BaseOutputHandler, OutputResult
from .json_handler import JSONOutputHandler
from .markdown_handler import MarkdownOutputHandler
from .leonardo_handler import LeonardoOutputHandler

logger = logging.getLogger(__name__)


class OutputHandlerRegistry:
    """Registry and factory for resolving and executing PromptFlow output handlers."""

    def __init__(self):
        self._handlers: dict[str, BaseOutputHandler] = {}
        # Register default built-in handlers
        self.register_handler(JSONOutputHandler())
        self.register_handler(MarkdownOutputHandler())
        leonardo = LeonardoOutputHandler()
        self.register_handler(leonardo)
        # Alias "image" handler to Leonardo handler
        self._handlers["image"] = leonardo

    def register_handler(self, handler: BaseOutputHandler) -> None:
        self._handlers[handler.name.lower()] = handler
        logger.info("Registered output handler: '%s'", handler.name)

    def get_handler(self, name: str) -> BaseOutputHandler | None:
        return self._handlers.get(name.lower())

    def list_handlers(self) -> list[str]:
        return sorted(list(self._handlers.keys()))

    def process_output(
        self,
        handler_name: str,
        raw_output: Any,
        options: dict[str, Any] | None = None,
        save_as: str = "output",
        output_dir: Path | None = None,
    ) -> OutputResult:
        handler = self.get_handler(handler_name)
        if not handler:
            return OutputResult(
                handler_name=handler_name,
                success=False,
                output_type=handler_name,
                error=f"Output handler '{handler_name}' is not registered.",
            )
        return handler.process(raw_output, options=options, save_as=save_as, output_dir=output_dir)


# Global default registry instance
default_registry = OutputHandlerRegistry()
