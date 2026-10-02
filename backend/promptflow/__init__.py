from __future__ import annotations

from file_runner import run_flow_from_file, FileRunnerError, FileRunnerValidationError


def run(
    flow: str,
    input: str,
    output: str | None = None,
    model: str | dict | None = None,
) -> dict:
    """
    Programmatic Python API for executing Flows from Jupyter notebooks or scripts.
    """
    model_override = None
    if isinstance(model, str):
        model_override = {"provider": "openai", "name": model, "temperature": 0.7}
    elif isinstance(model, dict):
        model_override = model

    return run_flow_from_file(
        flow=flow,
        input_path=input,
        output_path=output,
        model_override=model_override,
    )
