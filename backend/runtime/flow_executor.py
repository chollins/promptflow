from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path

from pydantic import BaseModel, Field

from .context_service import ExecutionContext
from .flow_service import get_flow
from .form_service import get_form
from .prompt_executor import LLMConfigurationError, LLMExecutionError, execute_prompt
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
    prompt: str | None = None
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


def _resolve_path(context_dict: dict, path: str) -> object | None:
    parts = path.split(".")
    current = context_dict
    for part in parts:
        if isinstance(current, dict) and part in current:
            current = current[part]
        elif isinstance(current, list) and part.isdigit() and int(part) < len(current):
            current = current[int(part)]
        else:
            return None
    return current


def _build_step_values(step: FlowStep, user_values: dict, context: ExecutionContext) -> dict[str, str]:
    values = {key: str(value) for key, value in (user_values or {}).items()}
    for key, value in context.all().items():
        values[key] = str(value)
    for prompt_var, context_key in step.input_bindings.items():
        if context_key.startswith("steps."):
            bound = _resolve_path(context.all(), context_key)
        else:
            bound = context.get(context_key)
        if bound is not None:
            values[prompt_var] = str(bound) if not isinstance(bound, (dict, list)) else json.dumps(bound)
        else:
            logger.warning("Missing context key for binding flow_step=%s prompt_var=%s context_key=%s", step.id, prompt_var, context_key)
    return values


def _find_unresolved_placeholders(rendered_prompt: str) -> list[str]:
    return sorted(set(_PLACEHOLDER_RE.findall(rendered_prompt)))


def _render_prompt_with_validation(template: str, values: dict[str, str]) -> str:
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


def _execute_step(flow: PromptFlow, step: FlowStep, user_values: dict, context: ExecutionContext, diagnostic_capabilities: frozenset[str] | None = None) -> tuple[FlowStepResult, dict | None]:
    if diagnostic_capabilities is None:
        diagnostic_capabilities = frozenset()
    started = time.perf_counter()
    form = get_form(step.prompt_form_id)
    values = _build_step_values(step, user_values, context)
    rendered_system_prompt = _render_prompt_with_validation(form.prompt.system, values)
    rendered_prompt = _render_prompt_with_validation(form.prompt.user, values)
    result = execute_prompt(
        system_prompt=rendered_system_prompt,
        user_prompt=rendered_prompt,
        model=form.model.name,
        temperature=form.model.temperature,
    )
    duration_ms = (time.perf_counter() - started) * 1000
    parsed_result = result
    if form.output:
        try:
            parsed_result = json.loads(result)
            if not isinstance(parsed_result, (dict, list)):
                raise ValueError("Expected an object or array.")
        except (json.JSONDecodeError, ValueError) as exc:
            raise LLMExecutionError(f"LLM output did not match expected structure: {exc}")

    context_vars = context.all()
    if "steps" not in context_vars:
        context_vars["steps"] = {}
    context_vars["steps"][step.id] = {
        "input": values,
        "output": parsed_result,
        "status": "completed",
        "error": None
    }
    # Sync back to context
    for k, v in context_vars.items():
        context.set(k, v)

    if step.output:
        context.set(step.output.save_as, result)
        _save_output(flow.id, step.output, result)
    logger.info("Step completed flow=%s step=%s in %.1f ms", flow.id, step.id, duration_ms)
    
    step_result = FlowStepResult(
        id=step.id,
        name=step.name,
        sequence=step.sequence,
        prompt=rendered_prompt if "prompts" in diagnostic_capabilities else None,
        result=result,
        completed=True,
        next=step.next,
        output=step.output.model_dump() if step.output else None,
    )

    debug_info = {}
    
    if "input_sources" in diagnostic_capabilities:
        input_sources = []
        for field in form.fields:
            if field.id in user_values:
                val = user_values[field.id]
                input_sources.append({
                    "field_id": field.id,
                    "label": field.label,
                    "source_type": "Current Form Input",
                    "source_name": "Current Form Input",
                    "path": f"values.{field.id}",
                    "value": val,
                })
            elif field.id in context.all():
                val = context.get(field.id)
                input_sources.append({
                    "field_id": field.id,
                    "label": field.label,
                    "source_type": "Previous Step Output",
                    "source_name": "Context",
                    "path": f"context.{field.id}",
                    "value": val,
                })

        for prompt_var, context_key in step.input_bindings.items():
            if context_key.startswith("steps."):
                bound = _resolve_path(context.all(), context_key)
                has_key = bound is not None
            else:
                bound = context.get(context_key)
                has_key = context_key in context.all()
            if has_key:
                # Find if there is a corresponding field
                field_label = prompt_var
                for f in form.fields:
                    if f.id == prompt_var:
                        field_label = f.label
                        break
                input_sources.append({
                    "field_id": prompt_var,
                    "label": field_label,
                    "source_type": "Previous Step Output",
                    "source_name": "Context Binding",
                    "path": f"context.{context_key}",
                    "value": bound,
                })
        debug_info["input_sources"] = input_sources
        
    if "prompts" in diagnostic_capabilities:
        debug_info["prompt_template"] = {
            "system": form.prompt.system,
            "user": form.prompt.user,
        }
        debug_info["resolved_prompt"] = {
            "system": rendered_system_prompt,
            "user": rendered_prompt,
        }
        
    if "model" in diagnostic_capabilities:
        debug_info["model_configuration"] = {
            "provider": form.model.provider,
            "name": form.model.name,
            "temperature": form.model.temperature,
        }
        
    if "output_schema" in diagnostic_capabilities:
        output_schema = None
        if step.output:
            output_schema = {
                "type": "object",
                "save_as": step.output.save_as,
                "formats": step.output.formats
            }
        debug_info["output_schema"] = output_schema
        
    if "raw_response" in diagnostic_capabilities:
        debug_info["raw_response"] = result
        
    if "execution" in diagnostic_capabilities:
        debug_info["execution_details"] = {
            "duration_ms": int(duration_ms),
        }

    return step_result, (debug_info if debug_info else None)


def execute_flow(flow_id: str, values: dict | None = None, context: dict | None = None, step_id: str | None = None, diagnostic_capabilities: frozenset[str] | None = None) -> FlowExecuteResponse:
    flow = get_flow(flow_id)
    step = _find_step(flow, step_id) if step_id else sorted(flow.steps, key=lambda item: item.sequence)[0]
    execution_context = ExecutionContext(context)
    step_result, debug_info = _execute_step(flow, step, values or {}, execution_context, diagnostic_capabilities)
    
    if debug_info:
        completed = []
        pending = []
        found_current = False
        for s in sorted(flow.steps, key=lambda item: item.sequence):
            if s.id == step.id:
                found_current = True
                completed.append(s.id)
            elif not found_current:
                completed.append(s.id)
            else:
                pending.append(s.id)
                
        debug_info["runtime_state"] = {
            "status": "completed",
            "current_step": step.id,
            "completed_steps": completed,
            "pending_steps": pending,
        }
        
    return FlowExecuteResponse(context=execution_context.all(), steps=[step_result], debug=debug_info)

