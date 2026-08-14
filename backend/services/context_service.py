from __future__ import annotations

import json
import logging

logger = logging.getLogger(__name__)


class ExecutionContext:
    def __init__(self, initial: dict | None = None) -> None:
        self.variables: dict = dict(initial or {})
        if "steps" not in self.variables:
            self.variables["steps"] = {}

    def set(self, key: str, value: object) -> None:
        self.variables[key] = value

    def get(self, key: str) -> object | None:
        return self.variables.get(key)

    def all(self) -> dict:
        return dict(self.variables)

    def store_step_result(self, step_id: str, user_values: dict, raw_result: str, parsed_result: object) -> None:
        """Store a step's input and parsed output into context.steps[step_id]."""
        if "steps" not in self.variables:
            self.variables["steps"] = {}
        self.variables["steps"][step_id] = {
            "input": dict(user_values),
            "output": parsed_result,
            "raw": raw_result,
            "status": "completed",
            "error": None,
        }
        logger.debug(
            "Stored step result step_id=%s output_type=%s output_keys=%s",
            step_id,
            type(parsed_result).__name__,
            list(parsed_result.keys()) if isinstance(parsed_result, dict) else "N/A",
        )
