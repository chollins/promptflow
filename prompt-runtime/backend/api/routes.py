"""REST API routes for Prompt Runtime Phase 1."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from schemas.prompt_form import PromptForm
from services.form_service import (
    FormNotFoundError,
    InvalidFormError,
    get_form,
    get_all_forms,
)
from services.prompt_executor import (
    LLMConfigurationError,
    LLMExecutionError,
    execute_prompt,
)
from services.prompt_service import render_prompt

logger = logging.getLogger(__name__)
router = APIRouter()


class ExecuteRequest(BaseModel):
    values: dict = Field(default_factory=dict)


class ExecuteResponse(BaseModel):
    prompt: str
    result: str

class FormSummary(BaseModel):
    id: str
    name: str
    description: str | None = None
    version: str

@router.get("/forms", response_model=list[FormSummary])
def list_forms() -> list[FormSummary]:
    """Return a list of all available forms."""
    try:
        return get_all_forms()
    except FormNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except InvalidFormError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected error listing forms")
        raise HTTPException(status_code=500, detail="Unexpected error listing forms") from exc


@router.get("/forms/{form_id}", response_model=PromptForm)
def read_form(form_id: str) -> PromptForm:
    """Return a validated PromptForm definition."""
    try:
        return get_form(form_id)
    except FormNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except InvalidFormError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected error loading form %s", form_id)
        raise HTTPException(
            status_code=500,
            detail="Unexpected error loading form",
        ) from exc


@router.post("/execute/{form_id}", response_model=ExecuteResponse)
def execute_form(form_id: str, request: ExecuteRequest) -> ExecuteResponse:
    """Load a form, render its prompt, and execute the LLM."""
    try:
        form = get_form(form_id)
    except FormNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except InvalidFormError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        rendered_prompt = render_prompt(form.prompt.user, request.values)
        result = execute_prompt(
            system_prompt=form.prompt.system,
            user_prompt=rendered_prompt,
            model=form.model.name,
            temperature=form.model.temperature,
        )
        return ExecuteResponse(prompt=rendered_prompt, result=result)
    except LLMConfigurationError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except LLMExecutionError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Unexpected error executing form %s", form_id)
        raise HTTPException(
            status_code=500,
            detail="Unexpected error executing prompt",
        ) from exc


@router.get("/health")
def health():
    return {
        "status":"ok",
        "version":"1.0"
    }