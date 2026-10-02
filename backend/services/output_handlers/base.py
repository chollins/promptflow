from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class OutputResult:
    handler_name: str
    success: bool
    output_type: str
    artifacts: list[str] = field(default_factory=list)  # File paths or artifact URLs
    data: Any = None
    error: str | None = None


class BaseOutputHandler(ABC):
    """Abstract base class for PromptFlow output handlers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier name for this output handler."""
        pass

    @abstractmethod
    def process(
        self,
        raw_output: Any,
        options: dict[str, Any] | None = None,
        save_as: str = "output",
        output_dir: Path | None = None,
    ) -> OutputResult:
        """Processes raw flow output and returns structured OutputResult."""
        pass
