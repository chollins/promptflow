"""Orchestrate PromptFlow steps using existing prompt execution services."""

from __future__ import annotations

import logging
import json
import time
import re
from pathlib import Path

from pydantic import BaseModel, Field

from schemas.prompt_flow import FlowStep, OutputSettings, PromptFlow
from services.context_service import ExecutionContext
from services.flow_service import FlowNotFoundError, InvalidFlowError, get_flow
from services.form_service import FormNotFoundError, InvalidFormError, get_form
from services.prompt_executor import (
    LLMConfigurationError,
    LLMExecutionError,
    execute_prompt,
)
from services.prompt_service import render_prompt

logger = logging.getLogger(__name__)

OUTPUTS_DIR = Path(__file__).resolve().parent.parent / "outputs"
_PLACEHOLDER_RE = re.compile(r"\{\{(\w+)\}\}")


class FlowStepNotFoundError(Exception):
    """Raised when a requested flow step does not exist."""


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


def _find_step(flow: PromptFlow, step_id: str) -> FlowStep:
    for step in flow.steps:
        if step.id == step_id:
            return step
    raise FlowStepNotFoundError(
        f"Step '{step_id}' not found in flow '{flow.id}'"
    )


def _build_step_values(
    step: FlowStep,
    user_values: dict,
    context: ExecutionContext,
) -> dict[str, str]:
    """Merge user values, then context values, then input bindings."""
    values: dict[str, str] = {
        key: str(value) for key, value in user_values.items()
    }

    for key, value in context.all().items():
        values[key] = str(value)

    for prompt_var, context_key in step.input_bindings.items():
        bound = context.get(context_key)
        if bound is not None:
            values[prompt_var] = str(bound)
        else:
            logger.warning(
                "Missing context key for binding flow_step=%s prompt_var=%s context_key=%s",
                step.id,
                prompt_var,
                context_key,
            )

    return values


def _find_unresolved_placeholders(rendered_prompt: str) -> list[str]:
    return sorted(set(_PLACEHOLDER_RE.findall(rendered_prompt)))


def _render_prompt_with_validation(template: str, values: dict[str, str]) -> str:
    rendered = render_prompt(template, values)
    unresolved = _find_unresolved_placeholders(rendered)
    if unresolved:
        message = f"Unresolved prompt placeholders: {', '.join(unresolved)}"
        logger.warning("%s", message)
        raise LLMExecutionError(message)
    return rendered


def _save_output(
    flow_id: str,
    output: OutputSettings,
    result: str,
) -> None:
    """Persist step output to backend/outputs/{flow_id}/."""
    output_dir = OUTPUTS_DIR / flow_id
    output_dir.mkdir(parents=True, exist_ok=True)

    for fmt in output.formats:
        if fmt == "json":
            file_path = output_dir / f"{output.save_as}.json"
            payload = {"save_as": output.save_as, "result": result}
            file_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            logger.info("Saved JSON output to %s", file_path)
        elif fmt == "markdown":
            file_path = output_dir / f"{output.save_as}.md"
            file_path.write_text(result, encoding="utf-8")
            logger.info("Saved markdown output to %s", file_path)


def _execute_step(
    flow: PromptFlow,
    step: FlowStep,
    user_values: dict,
    context: ExecutionContext,
) -> FlowStepResult:
    """Run a single flow step and update the execution context."""
    logger.info(
        "Step started flow=%s step=%s sequence=%s",
        flow.id,
        step.id,
        step.sequence,
    )
    started = time.perf_counter()

    form = get_form(step.prompt_form_id)
    values = _build_step_values(step, user_values, context)
    logger.info(
        "Execution values flow=%s step=%s request_values=%s context=%s bindings=%s merged_values=%s",
        flow.id,
        step.id,
        user_values,
        context.all(),
        step.input_bindings,
        values,
    )
    rendered_system_prompt = _render_prompt_with_validation(form.prompt.system, values)
    rendered_prompt = _render_prompt_with_validation(form.prompt.user, values)

    try:
        result = execute_prompt(
            system_prompt=rendered_system_prompt,
            user_prompt=rendered_prompt,
            model=form.model.name,
            temperature=form.model.temperature,
        )
    except (LLMConfigurationError, LLMExecutionError):
        logger.exception(
            "Step failed flow=%s step=%s",
            flow.id,
            step.id,
        )
        raise

    if step.output:
        context.set(step.output.save_as, result)
        try:
            _save_output(flow.id, step.output, result)
        except OSError as exc:
            logger.exception(
                "Failed to save output flow=%s step=%s",
                flow.id,
                step.id,
            )
            raise LLMExecutionError(f"Failed to save output: {exc}") from exc

    elapsed_ms = (time.perf_counter() - started) * 1000
    logger.info(
        "Step completed flow=%s step=%s in %.1f ms",
        flow.id,
        step.id,
        elapsed_ms,
    )

    return FlowStepResult(
        id=step.id,
        name=step.name,
        sequence=step.sequence,
        prompt=rendered_prompt,
        result=result,
        completed=True,
        next=step.next,
        output=step.output.model_dump() if step.output else None,
    )


def execute_flow(
    flow_id: str,
    values: dict | None = None,
    context: dict | None = None,
    step_id: str | None = None,
) -> FlowExecuteResponse:
    """
    Execute a single PromptFlow step in guided mode.

    Loads the flow, runs the requested step or the first step, updates context,
    and returns. Does not automatically continue to subsequent steps.
    """
    flow = get_flow(flow_id)
    if step_id is None:
        if not flow.steps:
            raise FlowStepNotFoundError(f"Flow '{flow.id}' has no steps")
        step = sorted(flow.steps, key=lambda item: item.sequence)[0]
    else:
        step = _find_step(flow, step_id)
    execution_context = ExecutionContext(context)

    if flow.runtime.mode == "automatic":
        # TODO: run all remaining steps sequentially without pauses.
        logger.info(
            "Automatic mode requested for flow=%s; executing single step only",
            flow.id,
        )

    step_result = _execute_step(flow, step, values or {}, execution_context)

    return FlowExecuteResponse(
        context=execution_context.all(),
        steps=[step_result],
    )


__all__ = [
    "FlowExecuteResponse",
    "FlowStepNotFoundError",
    "FlowStepResult",
    "execute_flow",
    "FlowNotFoundError",
    "InvalidFlowError",
    "FormNotFoundError",
    "InvalidFormError",
    "LLMConfigurationError",
    "LLMExecutionError",
]
