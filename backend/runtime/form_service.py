from __future__ import annotations

import logging
from pathlib import Path

from pydantic import ValidationError

from models import Form
from .schemas.prompt_form import PromptForm

logger = logging.getLogger(__name__)
FORMS_DIR = Path(__file__).resolve().parent.parent / "forms"
SAMPLE_FORMS_DIR = Path(__file__).resolve().parent.parent / "sample_forms"


class FormNotFoundError(Exception):
    pass


class InvalidFormError(Exception):
    pass


def _resolve_form_file_path(file_path: str | None) -> Path | None:
    if not file_path:
        return None

    path = Path(file_path)
    if not path.is_absolute():
        path = (FORMS_DIR.parent / file_path).resolve()
    return path


_RUNTIME_FORMS_CACHE: dict[str, PromptForm] = {}


def register_runtime_form(form: PromptForm) -> None:
    _RUNTIME_FORMS_CACHE[form.id] = form


def get_form(form_id: str) -> PromptForm:
    # 0. Check runtime cache (for markdown forms or dynamically registered forms)
    if form_id in _RUNTIME_FORMS_CACHE:
        return _RUNTIME_FORMS_CACHE[form_id]

    # 1. Search filesystem form assets first
    for file_path in (
        FORMS_DIR / f"{form_id}.form.json",
        SAMPLE_FORMS_DIR / f"{form_id}.form.json",
        SAMPLE_FORMS_DIR / f"{form_id}.json",
    ):
        if file_path.is_file():
            try:
                return PromptForm.model_validate_json(file_path.read_text(encoding="utf-8"))
            except ValidationError as exc:
                raise InvalidFormError(f"Invalid PromptForm JSON for '{form_id}': {exc}") from exc

    # 2. Search DB if inside Flask application context
    from flask import has_app_context
    if has_app_context():
        from models import Form
        from services.form_service import _slugify
        record = Form.query.filter((Form.id == form_id) | (Form.slug == form_id) | (Form.slug == _slugify(form_id))).first()
        if record:
            if record.content_json:
                try:
                    return PromptForm.model_validate_json(record.content_json)
                except ValidationError as exc:
                    raise InvalidFormError(f"Invalid PromptForm JSON for '{form_id}': {exc}") from exc

            resolved_path = _resolve_form_file_path(record.file_path)
            if resolved_path and resolved_path.is_file():
                try:
                    return PromptForm.model_validate_json(resolved_path.read_text(encoding="utf-8"))
                except ValidationError as exc:
                    raise InvalidFormError(f"Invalid PromptForm JSON for '{form_id}': {exc}") from exc

    raise FormNotFoundError(f"Form '{form_id}' not found")


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
    for file_path in sorted(list(FORMS_DIR.glob("*.form.json")) + list(SAMPLE_FORMS_DIR.glob("*.form.json")) + list(SAMPLE_FORMS_DIR.glob("*.json"))):
        slug = file_path.stem.replace(".form", "")
        if slug in seen_slugs:
            continue
        try:
            forms.append(PromptForm.model_validate_json(file_path.read_text(encoding="utf-8")))
        except Exception as exc:
            logger.warning("Skipping invalid form '%s': %s", file_path.name, exc)
    return forms

