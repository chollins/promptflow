"""Render prompt templates by substituting {{variable}} placeholders."""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

_PLACEHOLDER = re.compile(r"\{\{(\w+)\}\}")


def render_prompt(template: str, values: dict) -> str:
    """Replace {{key}} placeholders with values. Missing keys stay as-is."""

    def _replace(match: re.Match[str]) -> str:
        key = match.group(1)
        if key not in values:
            return match.group(0)
        return str(values[key])

    rendered = _PLACEHOLDER.sub(_replace, template)
    logger.info("Rendered prompt:\n%s", rendered)
    return rendered
