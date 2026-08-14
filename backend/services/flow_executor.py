from __future__ import annotations

import json
import logging
import re
import time
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, Field

from .context_service import ExecutionContext
from .flow_service import get_flow
from .form_service import get_form
from .prompt_executor import LLMExecutionError, execute_prompt
from .prompt_service import render_prompt
from .schemas.prompt_flow import FlowStep, OutputSettings, PromptFlow

logger = logging.getLogger(__name__)
OUTPUTS_DIR = Path(__file__).resolve().parent.parent / "outputs"
_PLACEHOLDER_RE = re.compile(r"\{\{([\w\.]+)\}\}")


class FlowStepNotFoundError(Exception):
    pass


class FlowStepResult(BaseModel):
    id: str
    name: str
    sequence: int
    prompt: str
    result: str
    completed: bool = True
    next: str | None = None
    output: dict | None = None


class FlowExecuteResponse(BaseModel):
    context: dict = Field(default_factory=dict)
    steps: list[FlowStepResult] = Field(default_factory=list)
    debug: dict | None = None


def _find_step(flow: PromptFlow, step_id: str) -> FlowStep:
    for step in flow.steps:
        if step.id == step_id:
            return step
    raise FlowStepNotFoundError(f"Step '{step_id}' not found in flow '{flow.id}'")


def _build_step_values(
    step: FlowStep,
    user_values: dict,
    context: ExecutionContext,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    values: dict[str, object] = dict(user_values or {})
    input_sources: list[dict[str, object]] = [
        {
            "field_id": key,
            "label": key,
            "source_type": "Current Form Input",
            "source_name": "Current Form Input",
            "path": f"values.{key}",
            "value": str(value),
        }
        for key, value in (user_values or {}).items()
    ]
    for key, value in context.all().items():
        values[key] = value
        input_sources.append(
            {
                "field_id": key,
                "label": key,
                "source_type": "Context Value",
                "source_name": "Execution Context",
                "path": f"context.{key}",
                "value": str(value),
            }
        )
    for prompt_var, context_key in step.input_bindings.items():
        bound = context.get(context_key)
        if bound is not None:
            values[prompt_var] = bound
            input_sources.append(
                {
                    "field_id": prompt_var,
                    "label": prompt_var,
                    "source_type": "Bound Context",
                    "source_name": context_key,
                    "path": f"context.{context_key}",
                    "value": str(bound),
                }
            )
        else:
            logger.warning(
                "Missing context key for binding flow_step=%s prompt_var=%s context_key=%s",
                step.id,
                prompt_var,
                context_key,
            )
    return values, input_sources


def _find_unresolved_placeholders(rendered_prompt: str) -> list[str]:
    return sorted(set(_PLACEHOLDER_RE.findall(rendered_prompt)))


def _render_prompt_with_validation(template: str, values: dict[str, object]) -> str:
    rendered = render_prompt(template, values)
    unresolved = _find_unresolved_placeholders(rendered)
    if unresolved:
        raise LLMExecutionError(f"Unresolved prompt placeholders: {', '.join(unresolved)}")
    return rendered


def _save_output(flow_id: str, output: OutputSettings, result: str) -> None:
    output_dir = OUTPUTS_DIR / flow_id
    output_dir.mkdir(parents=True, exist_ok=True)
    for fmt in output.formats:
        if fmt == "json":
            (output_dir / f"{output.save_as}.json").write_text(
                json.dumps({"save_as": output.save_as, "result": result}, indent=2),
                encoding="utf-8",
            )
        elif fmt == "markdown":
            (output_dir / f"{output.save_as}.md").write_text(result, encoding="utf-8")


def _execute_step(
    flow: PromptFlow,
    step: FlowStep,
    user_values: dict,
    context: ExecutionContext,
    *,
    include_debug: bool = False,
) -> tuple[FlowStepResult, dict | None]:
    started = time.perf_counter()
    started_at = datetime.utcnow()
    form = get_form(step.prompt_form_id)
    values, input_sources = _build_step_values(step, user_values, context)
    rendered_system_prompt = _render_prompt_with_validation(form.prompt.system, values)
    rendered_prompt = _render_prompt_with_validation(form.prompt.user, values)
    result = execute_prompt(
        system_prompt=rendered_system_prompt,
        user_prompt=rendered_prompt,
        model=form.model.name,
        temperature=form.model.temperature,
    )
    completed_at = datetime.utcnow()

    # Parse structured LLM output when the form declares an output schema
    parsed_result: object = result
    if form.output:
        try:
            clean_result = result.strip()
            if clean_result.startswith("```json"):
                clean_result = clean_result[7:]
            elif clean_result.startswith("```"):
                clean_result = clean_result[3:]
            if clean_result.endswith("```"):
                clean_result = clean_result[:-3]
            clean_result = clean_result.strip()

            parsed_result = json.loads(clean_result)
            if not isinstance(parsed_result, (dict, list)):
                raise ValueError("Expected an object or array.")
        except (json.JSONDecodeError, ValueError) as exc:
            logger.warning(
                "LLM output is not valid JSON for step=%s: %s (raw: %.200s)",
                step.id, exc, result,
            )
            parsed_result = result  # fall back to raw string

    # Store step result in context so subsequent forms can reference it
    # via data_source: { type: "step_output", step_id: "<step.id>", path: "<key>" }
    context.store_step_result(
        step_id=step.id,
        user_values=user_values or {},
        raw_result=result,
        parsed_result=parsed_result,
    )

    if step.output:
        context.set(step.output.save_as, result)
        _save_output(flow.id, step.output, result)
    logger.info(
        "Step completed flow=%s step=%s in %.1f ms",
        flow.id,
        step.id,
        (time.perf_counter() - started) * 1000,
    )
    step_result = FlowStepResult(
        id=step.id,
        name=step.name,
        sequence=step.sequence,
        prompt=rendered_prompt,
        result=result,
        completed=True,
        next=step.next,
        output=step.output.model_dump() if step.output else None,
    )
    debug = None
    if include_debug:
        debug = {
            "input_sources": input_sources,
            "prompt_template": {
                "system": form.prompt.system,
                "user": form.prompt.user,
            },
            "resolved_prompt": {
                "system": rendered_system_prompt,
                "user": rendered_prompt,
            },
            "model_configuration": {
                "provider": form.model.provider,
                "name": form.model.name,
                "temperature": form.model.temperature,
            },
            "output_schema": {
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
            },
            "raw_response": result,
            "execution_details": {
                "flow_id": flow.id,
                "step_id": step.id,
                "started_at": started_at.isoformat() + "Z",
                "completed_at": completed_at.isoformat() + "Z",
                "duration_ms": round((completed_at - started_at).total_seconds() * 1000, 2),
                "status": "completed",
                "retry_count": 0,
                "validation_status": "passed",
            },
            "runtime_state": {
                "status": "completed",
                "current_step": step.id,
                "context": context.all(),
            },
        }
    return step_result, debug


def execute_flow(
    flow_id: str,
    values: dict | None = None,
    context: dict | None = None,
    step_id: str | None = None,
    *,
    include_debug: bool = False,
) -> FlowExecuteResponse:
    flow = get_flow(flow_id)
    step = _find_step(flow, step_id) if step_id else sorted(flow.steps, key=lambda item: item.sequence)[0]
    execution_context = ExecutionContext(context)
    step_result, debug = _execute_step(flow, step, values or {}, execution_context, include_debug=include_debug)
    return FlowExecuteResponse(context=execution_context.all(), steps=[step_result], debug=debug)
