from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from runtime.flow_executor import execute_flow, _find_step, _execute_step, ExecutionContext
from runtime.flow_service import FlowNotFoundError, InvalidFlowError, get_flow
from runtime.form_service import FormNotFoundError, get_form
from runtime.schemas.prompt_flow import PromptFlow
from runtime.schemas.prompt_form import PromptForm

logger = logging.getLogger(__name__)


class FileRunnerError(Exception):
    """Base exception for file runner errors."""
    pass


class FileRunnerValidationError(FileRunnerError):
    """Raised when input files, JSON format, or required fields fail validation."""
    pass


def _validate_input_json(input_path: str | Path) -> dict[str, Any]:
    path = Path(input_path)
    if not path.is_file():
        raise FileRunnerValidationError(f"Input file not found: '{input_path}'")
    try:
        content = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise FileRunnerValidationError(f"Unable to read input file '{input_path}': {exc}") from exc

    try:
        data = json.loads(content)
    except json.JSONDecodeError as exc:
        raise FileRunnerValidationError(f"Invalid JSON in input file '{input_path}': {exc}") from exc

    if not isinstance(data, dict):
        raise FileRunnerValidationError(f"Input JSON in '{input_path}' must be a JSON object (dictionary).")

    return data


def _validate_required_fields(flow: PromptFlow, input_values: dict[str, Any]) -> None:
    missing_fields: list[str] = []
    for step in sorted(flow.steps, key=lambda s: s.sequence):
        try:
            form = get_form(step.prompt_form_id)
        except (FormNotFoundError, InvalidFlowError):
            continue
        for field in form.fields:
            if field.required and field.type != "hidden":
                # Check if field is satisfied by input_values or previous step outputs/bindings
                is_bound = field.id in step.input_bindings or any(
                    k == field.id or k.endswith(f".{field.id}") for k in step.input_bindings.values()
                )
                if not is_bound and (field.id not in input_values or input_values[field.id] is None or str(input_values[field.id]).strip() == ""):
                    if field.id not in missing_fields:
                        missing_fields.append(field.id)

    if missing_fields:
        raise FileRunnerValidationError(f"Missing required fields for flow '{flow.id}': {', '.join(missing_fields)}")


def run_flow_from_file(
    flow: str | Path,
    input_path: str | Path | None = None,
    output_path: str | Path | None = None,
    model_override: dict | None = None,
    input_values: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Executes a Flow using local JSON input file or input dict and returns structured output dict.
    Does NOT use the web application database for execution persistence.
    """
    # 1. Load & Validate Input JSON or Dict
    if input_values is None:
        if input_path is not None:
            input_values = _validate_input_json(input_path)
        else:
            input_values = {}
    else:
        input_values = dict(input_values)

    # 2. Load Flow definition
    flow_id_or_path = str(flow)
    try:
        prompt_flow = get_flow(flow_id_or_path)
    except (FlowNotFoundError, InvalidFlowError) as exc:
        raise FileRunnerValidationError(str(exc)) from exc

    # 3. Validate Required Fields
    _validate_required_fields(prompt_flow, input_values)

    # 4. Execute Flow Steps
    sorted_steps = sorted(prompt_flow.steps, key=lambda s: s.sequence)
    execution_context = ExecutionContext(dict(input_values))
    completed_steps = []

    for step in sorted_steps:
        step_result, _ = _execute_step(
            prompt_flow,
            step,
            user_values=input_values,
            context=execution_context,
            model_override=model_override,
        )
        completed_steps.append(step_result.model_dump())

    final_output = completed_steps[-1]["result"] if completed_steps else ""
    try:
        final_output_parsed = json.loads(final_output)
    except (json.JSONDecodeError, TypeError):
        final_output_parsed = final_output

    result_payload = {
        "flow_id": prompt_flow.id,
        "flow_name": prompt_flow.name,
        "status": "completed",
        "output": final_output_parsed,
        "context": execution_context.all(),
        "steps": completed_steps,
    }

    # 5. Persist JSON Output File if specified
    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        try:
            out_p.write_text(json.dumps(result_payload, indent=2), encoding="utf-8")
            logger.info("Saved flow output to '%s'", out_p)
        except OSError as exc:
            raise FileRunnerError(f"Failed to write output JSON to '{output_path}': {exc}") from exc

    return result_payload
