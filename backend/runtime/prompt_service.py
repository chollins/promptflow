from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)
_PLACEHOLDER = re.compile(r"\{\{(\w+)\}\}")


def render_prompt(template: str, values: dict) -> str:
    def _replace(match: re.Match[str]) -> str:
        key = match.group(1)
        return str(values[key]) if key in values else match.group(0)

    rendered = _PLACEHOLDER.sub(_replace, template)
    logger.info("Rendered prompt:\n%s", rendered)
    return rendered

