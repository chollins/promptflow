"""Load and validate PromptFlow JSON definitions."""

from __future__ import annotations

import logging
from pathlib import Path

from pydantic import ValidationError

from schemas.prompt_flow import PromptFlow

logger = logging.getLogger(__name__)

FLOWS_DIR = Path(__file__).resolve().parent.parent / "flows"


class FlowNotFoundError(Exception):
    """Raised when a flow JSON file does not exist."""


class InvalidFlowError(Exception):
    """Raised when a flow JSON file fails validation."""


def get_flow(flow_id: str) -> PromptFlow:
    """Load a PromptFlow by id from flows/{flow_id}.flow.json."""
    file_path = FLOWS_DIR / f"{flow_id}.flow.json"

    if not file_path.is_file():
        raise FlowNotFoundError(f"Flow '{flow_id}' not found")

    try:
        flow = PromptFlow.model_validate_json(
            file_path.read_text(encoding="utf-8")
        )
    except ValidationError as exc:
        raise InvalidFlowError(
            f"Invalid PromptFlow JSON for '{flow_id}': {exc}"
        ) from exc
    except OSError as exc:
        raise InvalidFlowError(
            f"Unable to read flow '{flow_id}': {exc}"
        ) from exc

    logger.info("Loaded flow id=%s name=%s", flow.id, flow.name)
    return flow


def get_all_flows() -> list[PromptFlow]:
    """Load all valid flows from the flows directory."""
    flows: list[PromptFlow] = []

    for file_path in sorted(FLOWS_DIR.glob("*.flow.json")):
        try:
            flow = PromptFlow.model_validate_json(
                file_path.read_text(encoding="utf-8")
            )
            flows.append(flow)
        except Exception as exc:
            logger.warning("Skipping invalid flow '%s': %s", file_path.name, exc)

    return flows
