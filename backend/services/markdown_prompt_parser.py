from __future__ import annotations

import re
from pathlib import Path
from typing import Sequence

from .schemas.prompt_form import FieldSchema, ModelSettings, Prompt, PromptForm


class MarkdownPromptParseError(Exception):
    """Raised when parsing a Markdown prompt file fails due to syntax or validation errors."""
    pass


PROMPT_MARKER_RE = re.compile(r"<!--\s*PROMPT:\s*([a-zA-Z0-9_\-]+)\s*-->", re.IGNORECASE)
SYSTEM_MARKER_RE = re.compile(r"<!--\s*SYSTEM:\s*(.*?)\s*-->", re.IGNORECASE | re.DOTALL)
PLACEHOLDER_RE = re.compile(r"\{\{([a-zA-Z0-9_\.]+)\}\}")


def parse_markdown_prompts(content_or_path: str | Path) -> list[PromptForm]:
    """
    Parses a Markdown string or file containing multiple prompts separated by HTML comments.

    Expected format:
    <!-- PROMPT: prompt_id -->
    <!-- SYSTEM: Optional system prompt text -->
    User prompt text containing {{variable}} placeholders.

    Returns a list of PromptForm objects compatible with PromptFlow runtime.
    """
    raw_content = ""
    if isinstance(content_or_path, Path):
        if not content_or_path.is_file():
            raise MarkdownPromptParseError(f"File not found: '{content_or_path}'")
        try:
            raw_content = content_or_path.read_text(encoding="utf-8")
        except OSError as exc:
            raise MarkdownPromptParseError(f"Failed to read file '{content_or_path}': {exc}") from exc
    else:
        # Check if content_or_path is a path string pointing to an existing file
        try:
            p = Path(content_or_path)
            if p.is_file():
                raw_content = p.read_text(encoding="utf-8")
            else:
                raw_content = str(content_or_path)
        except (ValueError, OSError):
            raw_content = str(content_or_path)

    if not raw_content.strip():
        raise MarkdownPromptParseError("Markdown content is empty.")

    matches = list(PROMPT_MARKER_RE.finditer(raw_content))
    if not matches:
        raise MarkdownPromptParseError("No valid '<!-- PROMPT: prompt_id -->' markers found in Markdown content.")

    seen_ids: set[str] = set()
    forms: list[PromptForm] = []

    for i, match in enumerate(matches):
        prompt_id = match.group(1).strip()
        if not prompt_id:
            raise MarkdownPromptParseError("Found empty or invalid prompt ID in comment marker.")

        if prompt_id in seen_ids:
            raise MarkdownPromptParseError(f"Duplicate prompt ID '{prompt_id}' found in Markdown content.")
        seen_ids.add(prompt_id)

        # Extract text content between current marker end and next marker start (or end of string)
        start_idx = match.end()
        end_idx = matches[i + 1].start() if i + 1 < len(matches) else len(raw_content)
        section_text = raw_content[start_idx:end_idx].strip()

        # Check for optional SYSTEM comment inside section
        system_prompt = "You are a helpful AI assistant."
        system_match = SYSTEM_MARKER_RE.search(section_text)
        if system_match:
            system_prompt = system_match.group(1).strip()
            # Remove system tag from user prompt text
            user_text = SYSTEM_MARKER_RE.sub("", section_text).strip()
        else:
            user_text = section_text

        if not user_text:
            raise MarkdownPromptParseError(f"Prompt '{prompt_id}' has empty prompt text.")

        # Extract placeholders
        found_vars = PLACEHOLDER_RE.findall(user_text) + PLACEHOLDER_RE.findall(system_prompt)
        unique_vars: list[str] = []
        for var in found_vars:
            if var not in unique_vars:
                unique_vars.append(var)

        fields: list[FieldSchema] = []
        for var in unique_vars:
            label = var.replace("_", " ").title()
            field_type = "textarea" if any(kw in var.lower() for kw in ("copy", "summary", "desc", "text", "details", "content", "prompt")) else "text"
            fields.append(
                FieldSchema(
                    id=var,
                    label=label,
                    type=field_type,
                    required=True,
                )
            )

        form_name = prompt_id.replace("_", " ").replace("-", " ").title()

        form = PromptForm(
            id=prompt_id,
            name=form_name,
            version="1.0",
            fields=fields,
            prompt=Prompt(system=system_prompt, user=user_text),
            model=ModelSettings(provider="openai", name="gpt-4o-mini", temperature=0.7),
        )
        forms.append(form)

    return forms
