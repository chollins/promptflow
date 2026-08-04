from __future__ import annotations

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
    prompt: str
    result: str
    values: dict = Field(default_factory=dict)


def _find_unresolved_placeholders(rendered_prompt: str) -> list[str]:
    return sorted(set(_PLACEHOLDER_RE.findall(rendered_prompt)))


def _render_prompt_with_validation(template: str, values: dict[str, str]) -> str:
    rendered = render_prompt(template, values)
    unresolved = _find_unresolved_placeholders(rendered)
    if unresolved:
        raise LLMExecutionError(f"Unresolved prompt placeholders: {', '.join(unresolved)}")
    return rendered


def execute_form(form_id: str, values: dict | None = None) -> FormExecuteResponse:
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
    logger.info("Form executed form=%s", form_id)
    return FormExecuteResponse(form_id=form_id, prompt=rendered_prompt, result=result, values=merged_values)
