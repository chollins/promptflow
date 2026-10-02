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
    candidates = [
        Path(flow_id),
        Path("prompts") / flow_id,
        Path(__file__).resolve().parent.parent / "prompts" / flow_id,
        FLOWS_DIR / f"{flow_id}.flow.json",
        FLOWS_DIR / f"{flow_id}.md",
        Path("prompts") / f"{flow_id}.md",
        Path(__file__).resolve().parent.parent / "prompts" / f"{flow_id}.md",
    ]

    target_path: Path | None = None
    for cand in candidates:
        if cand.is_file():
            target_path = cand
            break

    if not target_path:
        raise FlowNotFoundError(f"Flow '{flow_id}' not found")

    if target_path.suffix.lower() == ".json" or target_path.name.endswith(".flow.json"):
        try:
            return PromptFlow.model_validate_json(target_path.read_text(encoding="utf-8"))
        except ValidationError as exc:
            raise InvalidFlowError(f"Invalid PromptFlow JSON for '{flow_id}': {exc}") from exc

    if target_path.suffix.lower() in (".md", ".markdown"):
        try:
            from services.markdown_prompt_parser import parse_markdown_prompts
            from .form_service import register_runtime_form
            from .schemas.prompt_flow import FlowStep
        except ImportError:
            from ..services.markdown_prompt_parser import parse_markdown_prompts
            from .form_service import register_runtime_form
            from .schemas.prompt_flow import FlowStep

        try:
            forms = parse_markdown_prompts(target_path)
            steps: list[FlowStep] = []
            for i, form in enumerate(forms, start=1):
                register_runtime_form(form)
                steps.append(
                    FlowStep(
                        id=f"step_{i}_{form.id}",
                        sequence=i,
                        name=form.name,
                        prompt_form_id=form.id,
                        input_bindings={},
                    )
                )

            flow_name = target_path.stem.replace("_", " ").replace("-", " ").title()
            return PromptFlow(
                id=target_path.stem,
                version="1.0",
                name=flow_name,
                description=f"Flow created from markdown file '{target_path.name}'",
                steps=steps,
            )
        except Exception as exc:
            raise InvalidFlowError(f"Failed to load Markdown prompt flow from '{target_path}': {exc}") from exc

    raise InvalidFlowError(f"Unsupported file format for flow '{flow_id}'")


def get_all_flows() -> list[PromptFlow]:
    flows: list[PromptFlow] = []
    for file_path in sorted(FLOWS_DIR.glob("*.flow.json")):
        try:
            flows.append(PromptFlow.model_validate_json(file_path.read_text(encoding="utf-8")))
        except Exception as exc:
            logger.warning("Skipping invalid flow '%s': %s", file_path.name, exc)
    return flows

