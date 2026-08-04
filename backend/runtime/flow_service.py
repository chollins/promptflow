from __future__ import annotations

import logging
from pathlib import Path

from pydantic import ValidationError

from .schemas.prompt_flow import PromptFlow

logger = logging.getLogger(__name__)
FLOWS_DIR = Path(__file__).resolve().parent.parent / "flows"


class FlowNotFoundError(Exception):
    pass


class InvalidFlowError(Exception):
    pass


def get_flow(flow_id: str) -> PromptFlow:
    file_path = FLOWS_DIR / f"{flow_id}.flow.json"
    if not file_path.is_file():
        raise FlowNotFoundError(f"Flow '{flow_id}' not found")
    try:
        return PromptFlow.model_validate_json(file_path.read_text(encoding="utf-8"))
    except ValidationError as exc:
        raise InvalidFlowError(f"Invalid PromptFlow JSON for '{flow_id}': {exc}") from exc


def get_all_flows() -> list[PromptFlow]:
    flows: list[PromptFlow] = []
    for file_path in sorted(FLOWS_DIR.glob("*.flow.json")):
        try:
            flows.append(PromptFlow.model_validate_json(file_path.read_text(encoding="utf-8")))
        except Exception as exc:
            logger.warning("Skipping invalid flow '%s': %s", file_path.name, exc)
    return flows

