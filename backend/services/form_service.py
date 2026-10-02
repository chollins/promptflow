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
SAMPLE_FORMS_DIR = Path(__file__).resolve().parent.parent / "sample_forms"


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


def _resolve_form_file_path(file_path: str | None) -> Path | None:
    if not file_path:
        return None

    path = Path(file_path)
    if not path.is_absolute():
        path = (FORMS_DIR.parent / file_path).resolve()
    return path


def _candidate_form_paths(form_id: str) -> list[Path]:
    slugified = _slugify(form_id)
    paths = [
        FORMS_DIR / f"{form_id}.form.json",
        SAMPLE_FORMS_DIR / f"{form_id}.form.json",
        SAMPLE_FORMS_DIR / f"{form_id}.json",
    ]
    if slugified != form_id:
        paths.extend([
            FORMS_DIR / f"{slugified}.form.json",
            SAMPLE_FORMS_DIR / f"{slugified}.form.json",
            SAMPLE_FORMS_DIR / f"{slugified}.json",
        ])
    return paths


def find_form_by_id_or_slug(form_id: str) -> Form | None:
    return Form.query.filter(
        (Form.id == form_id) | (Form.slug == form_id) | (Form.slug == _slugify(form_id))
    ).first()


def get_form(form_id: str) -> PromptForm:
    record = find_form_by_id_or_slug(form_id)
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

    for file_path in _candidate_form_paths(form_id):
        if not file_path.is_file():
            continue
        try:
            return PromptForm.model_validate_json(file_path.read_text(encoding="utf-8"))
        except ValidationError as exc:
            raise InvalidFormError(f"Invalid PromptForm JSON for '{form_id}': {exc}") from exc

    raise FormNotFoundError(
        f"Form '{form_id}' not found"
        + (f" (stored file_path: {record.file_path})" if record and record.file_path else "")
    )


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


def create_form(*, name: str, description: str | None, content_json: str, is_active: bool = True, slug: str | None = None) -> Form:
    form_def = PromptForm.model_validate_json(content_json)
    if not slug:
        slug = _unique_slug(_slugify(name))
    else:
        slug = _unique_slug(_slugify(slug))
    file_path = f"forms/{slug}.form.json"
    form = Form(
        name=name,
        slug=slug,
        description=description,
        content_json=json.dumps(form_def.model_dump(exclude_none=True, by_alias=True), indent=2),
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
    form = find_form_by_id_or_slug(form_id)
    if not form:
        raise FormNotFoundError(f"Form '{form_id}' not found")

    form_def = PromptForm.model_validate_json(content_json)
    form.name = name
    form.description = description
    form.content_json = json.dumps(form_def.model_dump(exclude_none=True, by_alias=True), indent=2)
    form.is_active = is_active
    form.slug = _unique_slug(_slugify(name), form.id)
    db.session.commit()
    return form


def delete_form(form_id: str) -> None:
    form = find_form_by_id_or_slug(form_id)
    if not form:
        raise FormNotFoundError(f"Form '{form_id}' not found")
    db.session.delete(form)
    db.session.commit()


def import_forms_from_markdown(content_or_path: str | Path) -> list[Form]:
    from .markdown_prompt_parser import parse_markdown_prompts
    prompt_forms = parse_markdown_prompts(content_or_path)
    imported_db_forms: list[Form] = []
    for p_form in prompt_forms:
        json_str = json.dumps(p_form.model_dump(exclude_none=True, by_alias=True), indent=2)
        existing = find_form_by_id_or_slug(p_form.id)
        if existing:
            updated = update_form(
                existing.id,
                name=p_form.name,
                description=p_form.description,
                content_json=json_str,
                is_active=True,
            )
            imported_db_forms.append(updated)
        else:
            created = create_form(
                name=p_form.name,
                description=p_form.description,
                content_json=json_str,
                is_active=True,
                slug=p_form.id,
            )
            imported_db_forms.append(created)
    return imported_db_forms

