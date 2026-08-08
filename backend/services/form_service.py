from __future__ import annotations

import json
import logging
from pathlib import Path

from pydantic import ValidationError

from extensions import db
from models import Form
from .schemas.prompt_form import PromptForm

logger = logging.getLogger(__name__)
FORMS_DIR = Path(__file__).resolve().parent.parent / "forms"


class FormNotFoundError(Exception):
    pass


class InvalidFormError(Exception):
    pass


def _slugify(name: str) -> str:
    slug = "".join(ch.lower() if ch.isalnum() else "-" for ch in name).strip("-")
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug or "form"


def _unique_slug(base: str, form_id: str | None = None) -> str:
    slug = base
    counter = 2
    while True:
        query = Form.query.filter_by(slug=slug)
        if form_id:
            query = query.filter(Form.id != form_id)
        if query.first() is None:
            return slug
        slug = f"{base}-{counter}"
        counter += 1


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


def create_form(*, name: str, description: str | None, content_json: str, is_active: bool = True) -> Form:
    form_def = PromptForm.model_validate_json(content_json)
    slug = _unique_slug(_slugify(name))
    file_path = f"forms/{slug}.form.json"
    form = Form(
        name=name,
        slug=slug,
        description=description,
        content_json=json.dumps(form_def.model_dump(), indent=2),
        file_path=file_path,
        is_active=is_active,
    )
    db.session.add(form)
    db.session.commit()
    return form


def update_form(
    form_id: str,
    *,
    name: str,
    description: str | None,
    content_json: str,
    is_active: bool,
) -> Form:
    form = Form.query.filter((Form.id == form_id) | (Form.slug == form_id)).first()
    if not form:
        raise FormNotFoundError(f"Form '{form_id}' not found")

    form_def = PromptForm.model_validate_json(content_json)
    form.name = name
    form.description = description
    form.content_json = json.dumps(form_def.model_dump(), indent=2)
    form.is_active = is_active
    form.slug = _unique_slug(_slugify(name), form.id)
    db.session.commit()
    return form


def delete_form(form_id: str) -> None:
    form = Form.query.filter((Form.id == form_id) | (Form.slug == form_id)).first()
    if not form:
        raise FormNotFoundError(f"Form '{form_id}' not found")
    db.session.delete(form)
    db.session.commit()
