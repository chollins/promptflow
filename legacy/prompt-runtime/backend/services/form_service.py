"""Load and validate PromptForm JSON definitions."""

from __future__ import annotations
import logging
from pathlib import Path
from pydantic import ValidationError
from schemas.prompt_form import PromptForm

logger = logging.getLogger(__name__)

FORMS_DIR = Path(__file__).resolve().parent.parent / "forms"


class FormNotFoundError(Exception):
    """Raised when a form JSON file does not exist."""


class InvalidFormError(Exception):
    """Raised when a form JSON file fails validation."""


def get_form(form_id: str) -> PromptForm:
    """Load a PromptForm by id from forms/{form_id}.form.json."""
    file_path = FORMS_DIR / f"{form_id}.form.json"

    if not file_path.is_file():
        raise FormNotFoundError(f"Form '{form_id}' not found")

    try:
        form = PromptForm.model_validate_json(
            file_path.read_text(encoding="utf-8")
        )
    except ValidationError as exc:
        raise InvalidFormError(
            f"Invalid PromptForm JSON for '{form_id}': {exc}"
        ) from exc
    except OSError as exc:
        raise InvalidFormError(
            f"Unable to read form '{form_id}': {exc}"
        ) from exc

    logger.info("Loaded form id=%s name=%s", form.id, form.name)
    return form



def get_all_forms() -> list[PromptForm]:
    forms = []

    for file in sorted(FORMS_DIR.glob("*.form.json")):
        try:
            form = PromptForm.model_validate_json(
                file.read_text(encoding="utf-8")
            )
            forms.append(form)
        except Exception as exc:
            logger.warning(f"Skipping invalid form '{file.name}': {exc}")

    return forms