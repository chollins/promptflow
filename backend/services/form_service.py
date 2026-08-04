from __future__ import annotations

import logging
from pathlib import Path

from pydantic import ValidationError

from .schemas.prompt_form import PromptForm

logger = logging.getLogger(__name__)
FORMS_DIR = Path(__file__).resolve().parent.parent / "forms"


class FormNotFoundError(Exception):
    pass


class InvalidFormError(Exception):
    pass


def get_form(form_id: str) -> PromptForm:
    file_path = FORMS_DIR / f"{form_id}.form.json"
    if not file_path.is_file():
        raise FormNotFoundError(f"Form '{form_id}' not found")
    try:
        return PromptForm.model_validate_json(file_path.read_text(encoding="utf-8"))
    except ValidationError as exc:
        raise InvalidFormError(f"Invalid PromptForm JSON for '{form_id}': {exc}") from exc


def get_all_forms() -> list[PromptForm]:
    forms: list[PromptForm] = []
    for file_path in sorted(FORMS_DIR.glob("*.form.json")):
        try:
            forms.append(PromptForm.model_validate_json(file_path.read_text(encoding="utf-8")))
        except Exception as exc:
            logger.warning("Skipping invalid form '%s': %s", file_path.name, exc)
    return forms
