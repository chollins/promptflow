from __future__ import annotations

import logging
from pathlib import Path

from pydantic import ValidationError

from models import Form
from .schemas.prompt_form import PromptForm

logger = logging.getLogger(__name__)
FORMS_DIR = Path(__file__).resolve().parent.parent / "forms"


class FormNotFoundError(Exception):
    pass


class InvalidFormError(Exception):
    pass


def get_form(form_id: str) -> PromptForm:
    record = Form.query.filter((Form.id == form_id) | (Form.slug == form_id)).first()
    if record:
        if record.content_json:
            try:
                return PromptForm.model_validate_json(record.content_json)
            except ValidationError as exc:
                raise InvalidFormError(f"Invalid PromptForm JSON for '{form_id}': {exc}") from exc
        file_path = FORMS_DIR / f"{form_id}.form.json"
        if file_path.is_file():
            try:
                return PromptForm.model_validate_json(file_path.read_text(encoding="utf-8"))
            except ValidationError as exc:
                raise InvalidFormError(f"Invalid PromptForm JSON for '{form_id}': {exc}") from exc

    file_path = FORMS_DIR / f"{form_id}.form.json"
    if not file_path.is_file():
        raise FormNotFoundError(f"Form '{form_id}' not found")
    try:
        return PromptForm.model_validate_json(file_path.read_text(encoding="utf-8"))
    except ValidationError as exc:
        raise InvalidFormError(f"Invalid PromptForm JSON for '{form_id}': {exc}") from exc


def get_all_forms() -> list[PromptForm]:
    forms: list[PromptForm] = []
    seen_slugs: set[str] = set()
    for record in Form.query.order_by(Form.name.asc()).all():
        if not record.content_json:
            continue
        try:
            forms.append(PromptForm.model_validate_json(record.content_json))
            seen_slugs.add(record.slug)
        except Exception as exc:
            logger.warning("Skipping invalid db form '%s': %s", record.slug, exc)
    for file_path in sorted(FORMS_DIR.glob("*.form.json")):
        slug = file_path.stem.replace(".form", "")
        if slug in seen_slugs:
            continue
        try:
            forms.append(PromptForm.model_validate_json(file_path.read_text(encoding="utf-8")))
        except Exception as exc:
            logger.warning("Skipping invalid form '%s': %s", file_path.name, exc)
    return forms

