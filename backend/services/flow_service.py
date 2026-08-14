from __future__ import annotations

import json
import logging
from pathlib import Path

from pydantic import ValidationError

from extensions import db
from models import Flow
from .schemas.prompt_flow import PromptFlow

logger = logging.getLogger(__name__)
FLOWS_DIR = Path(__file__).resolve().parent.parent / "flows"


class FlowNotFoundError(Exception):
    pass


class InvalidFlowError(Exception):
    pass


def _load_flow_from_text(flow_id: str, raw_content: str) -> PromptFlow:
    try:
        return PromptFlow.model_validate_json(raw_content)
    except ValidationError as exc:
        raise InvalidFlowError(f"Invalid PromptFlow JSON for '{flow_id}': {exc}") from exc


def _load_flow_from_file(flow_id: str, file_path: str) -> PromptFlow:
    path = Path(file_path)
    if not path.is_absolute():
        path = (FLOWS_DIR.parent / file_path).resolve()
    if not path.is_file():
        raise FlowNotFoundError(f"Flow '{flow_id}' not found")
    try:
        return _load_flow_from_text(flow_id, path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise InvalidFlowError(f"Unable to read flow '{flow_id}': {exc}") from exc


def load_flow_definition(flow: Flow) -> PromptFlow:
    steps = sorted(flow.form_steps, key=lambda step: step.step_number)
    if not steps:
        if flow.content_json:
            return _load_flow_from_text(flow.slug, flow.content_json)
        return _load_flow_from_file(flow.slug, flow.file_path)

    return PromptFlow(
        id=flow.slug,
        version="1.0",
        name=flow.name,
        description=flow.description or "",
        steps=[
            {
                "id": step.form.slug.replace("-", "_") if step.form else step.form_id,
                "sequence": step.step_number,
                "name": step.form.name if step.form else step.form_id,
                "prompt_form_id": step.form.slug if step.form else step.form_id,
                "input_bindings": {},
                "dynamic_fields": [],
                "review": {"required": step.is_required, "editable": True},
                "output": None,
                "next": None,
            }
            for step in steps
        ],
    )


def get_flow(flow_id: str) -> PromptFlow:
    flow = Flow.query.filter(
        (Flow.id == flow_id) | (Flow.slug == flow_id)
    ).first()
    if not flow:
        file_path = FLOWS_DIR / f"{flow_id}.flow.json"
        if file_path.is_file():
            return _load_flow_from_file(flow_id, str(file_path))
        raise FlowNotFoundError(f"Flow '{flow_id}' not found")
    return load_flow_definition(flow)


def get_all_flows() -> list[PromptFlow]:
    flows: list[PromptFlow] = []
    seen_ids: set[str] = set()

    for record in Flow.query.order_by(Flow.name.asc()).all():
        try:
            flows.append(load_flow_definition(record))
            seen_ids.add(record.slug)
        except Exception as exc:
            logger.warning("Skipping invalid db flow '%s': %s", record.slug, exc)

    for file_path in sorted(FLOWS_DIR.glob("*.flow.json")):
        slug = file_path.stem.replace(".flow", "")
        if slug in seen_ids:
            continue
        try:
            flows.append(_load_flow_from_file(slug, str(file_path)))
        except Exception as exc:
            logger.warning("Skipping invalid flow '%s': %s", file_path.name, exc)

    return flows


def _slugify(name: str) -> str:
    slug = "".join(ch.lower() if ch.isalnum() else "-" for ch in name).strip("-")
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug or "flow"


def _unique_slug(base: str, flow_id: str | None = None) -> str:
    slug = base
    counter = 2
    while True:
        query = Flow.query.filter_by(slug=slug)
        if flow_id:
            query = query.filter(Flow.id != flow_id)
        if query.first() is None:
            return slug
        slug = f"{base}-{counter}"
        counter += 1


def create_flow(
    *,
    name: str,
    description: str | None,
    content_json: str,
    is_active: bool = True,
) -> Flow:
    flow_def = _load_flow_from_text("new-flow", content_json)
    slug = _unique_slug(_slugify(name))
    file_path = f"flows/{slug}.flow.json"
    flow = Flow(
        name=name,
        slug=slug,
        description=description,
        content_json=json.dumps(flow_def.model_dump(), indent=2),
        file_path=file_path,
        is_active=is_active,
    )
    db.session.add(flow)
    db.session.commit()
    return flow


def update_flow(
    flow_id: str,
    *,
    name: str,
    description: str | None,
    content_json: str,
    is_active: bool,
) -> Flow:
    flow = Flow.query.filter((Flow.id == flow_id) | (Flow.slug == flow_id)).first()
    if not flow:
        raise FlowNotFoundError(f"Flow '{flow_id}' not found")

    flow_def = _load_flow_from_text(flow_id, content_json)
    flow.name = name
    flow.description = description
    flow.content_json = json.dumps(flow_def.model_dump(), indent=2)
    flow.is_active = is_active
    flow.slug = _unique_slug(_slugify(name), flow.id)
    db.session.commit()
    return flow


def delete_flow(flow_id: str) -> None:
    flow = Flow.query.filter((Flow.id == flow_id) | (Flow.slug == flow_id)).first()
    if not flow:
        raise FlowNotFoundError(f"Flow '{flow_id}' not found")
    db.session.delete(flow)
    db.session.commit()
