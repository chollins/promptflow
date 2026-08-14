from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)
_PLACEHOLDER = re.compile(r"\{\{([\w\.]+)\}\}")


def render_prompt(template: str, values: dict) -> str:
    def _resolve_path(path: str) -> object:
        parts = path.split(".")
        current: object = values
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            elif hasattr(current, part):
                current = getattr(current, part)
            else:
                return None
        return current

    def _replace(match: re.Match[str]) -> str:
        key = match.group(1)
        val = _resolve_path(key)
        if val is not None:
            if isinstance(val, (dict, list)):
                import json
                return json.dumps(val)
            return str(val)
        return match.group(0)

    rendered = _PLACEHOLDER.sub(_replace, template)
    logger.info("Rendered prompt:\n%s", rendered)
    return rendered
