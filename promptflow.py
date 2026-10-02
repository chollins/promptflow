from __future__ import annotations

import sys
from pathlib import Path

# Ensure backend directory is in sys.path
backend_dir = Path(__file__).resolve().parent / "backend"
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from backend.file_runner import run_flow_from_file, FileRunnerError, FileRunnerValidationError


def run(
    flow: str | Path,
    input: str | Path,
    output: str | Path | None = None,
    model: str | dict | None = None,
) -> dict:
    """
    Programmatic Python API for executing Flows from Jupyter notebooks or scripts.

    Example usage in Jupyter:
        import promptflow

        result = promptflow.run(
            flow="client_assessment",
            input="./inputs/company.json",
            output="./outputs/company.json"
        )
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
