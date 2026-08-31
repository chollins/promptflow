from __future__ import annotations

from datetime import datetime
import logging
import re

from pydantic import BaseModel, Field

from .form_service import get_form
from .prompt_executor import LLMExecutionError, execute_prompt
from .prompt_service import render_prompt

logger = logging.getLogger(__name__)
_PLACEHOLDER_RE = re.compile(r"\{\{(\w+)\}\}")


class FormExecuteResponse(BaseModel):
    form_id: str
    prompt: str | None = None
    result: str
    values: dict = Field(default_factory=dict)
    debug: dict | None = None


def _find_unresolved_placeholders(rendered_prompt: str) -> list[str]:
    return sorted(set(_PLACEHOLDER_RE.findall(rendered_prompt)))


def _render_prompt_with_validation(template: str, values: dict[str, str]) -> str:
    rendered = render_prompt(template, values)
    unresolved = _find_unresolved_placeholders(rendered)
    if unresolved:
        raise LLMExecutionError(f"Unresolved prompt placeholders: {', '.join(unresolved)}")
    return rendered


def execute_form(
    form_id: str,
    values: dict | None = None,
    *,
    diagnostic_capabilities: frozenset[str] | None = None,
) -> FormExecuteResponse:
    if diagnostic_capabilities is None:
        diagnostic_capabilities = frozenset()
        
    started_at = datetime.utcnow()
    form = get_form(form_id)
    merged_values = {key: str(value) for key, value in (values or {}).items()}
    rendered_system_prompt = _render_prompt_with_validation(form.prompt.system, merged_values)
    rendered_prompt = _render_prompt_with_validation(form.prompt.user, merged_values)
    result = execute_prompt(
        system_prompt=rendered_system_prompt,
        user_prompt=rendered_prompt,
        model=form.model.name,
        temperature=form.model.temperature,
    )
    completed_at = datetime.utcnow()
    debug = {}
    
    if "input_sources" in diagnostic_capabilities:
        debug["input_sources"] = [
            {
                "field_id": field.id,
                "label": field.label,
                "source_type": "Current Form Input",
                "source_name": "Current Form Input",
                "path": f"values.{field.id}",
                "value": merged_values.get(field.id),
            }
            for field in form.fields
        ]
    if "prompts" in diagnostic_capabilities:
        debug["prompt_template"] = {
            "system": form.prompt.system,
            "user": form.prompt.user,
        }
        debug["resolved_prompt"] = {
            "system": rendered_system_prompt,
            "user": rendered_prompt,
        }
    if "model" in diagnostic_capabilities:
        debug["model_configuration"] = {
            "provider": form.model.provider,
            "name": form.model.name,
            "temperature": form.model.temperature,
        }
    if "output_schema" in diagnostic_capabilities:
        debug["output_schema"] = {
            "type": "object",
            "properties": {
                field.id: {
                    "label": field.label,
                    "type": field.type,
                    "required": field.required,
                    "description": field.description,
                    "default": field.default,
                    "options": field.options,
                }
                for field in form.fields
            },
            "required": [field.id for field in form.fields if field.required],
        }
    if "raw_response" in diagnostic_capabilities:
        debug["raw_response"] = result
    if "execution" in diagnostic_capabilities:
        debug["execution_details"] = {
            "form_id": form.id,
            "started_at": started_at.isoformat() + "Z",
            "completed_at": completed_at.isoformat() + "Z",
            "duration_ms": round((completed_at - started_at).total_seconds() * 1000, 2),
            "status": "completed",
        }
        debug["runtime_state"] = {
            "status": "completed",
            "current_form_id": form.id,
            "values": merged_values,
        }
        
    logger.info("Form executed form=%s", form_id)
    return FormExecuteResponse(
        form_id=form_id,
        prompt=rendered_prompt if "prompts" in diagnostic_capabilities else None,
        result=result,
        values=merged_values,
        debug=debug if debug else None,
    )
