"""REST API routes for Prompt Runtime."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from schemas.prompt_flow import PromptFlow
from schemas.prompt_form import PromptForm
from services.flow_executor import (
    FlowExecuteResponse,
    FlowStepNotFoundError,
    execute_flow as execute_flow_runtime,
)
from services.flow_service import (
    FlowNotFoundError,
    InvalidFlowError,
    get_all_flows,
    get_flow,
)
from services.form_service import (
    FormNotFoundError,
    InvalidFormError,
    get_all_forms,
    get_form,
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


class FlowSummary(BaseModel):
    id: str
    name: str
    description: str
    version: str


class FlowExecuteRequest(BaseModel):
    step_id: str | None = None
    values: dict = Field(default_factory=dict)
    context: dict = Field(default_factory=dict)


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
        raise HTTPException(
            status_code=500,
            detail="Unexpected error listing forms",
        ) from exc


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


@router.get("/flows", response_model=list[FlowSummary])
def list_flows() -> list[FlowSummary]:
    """Return a list of all available flows."""
    try:
        return get_all_flows()
    except FlowNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except InvalidFlowError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected error listing flows")
        raise HTTPException(
            status_code=500,
            detail="Unexpected error listing flows",
        ) from exc


@router.get("/flows/{flow_id}", response_model=PromptFlow)
def read_flow(flow_id: str) -> PromptFlow:
    """Return a validated PromptFlow definition."""
    try:
        return get_flow(flow_id)
    except FlowNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except InvalidFlowError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected error loading flow %s", flow_id)
        raise HTTPException(
            status_code=500,
            detail="Unexpected error loading flow",
        ) from exc


@router.post("/flows/{flow_id}/execute", response_model=FlowExecuteResponse)
def execute_flow(
    flow_id: str,
    request: FlowExecuteRequest,
) -> FlowExecuteResponse:
    """Execute a single PromptFlow step in guided mode."""
    try:
        return execute_flow_runtime(
            flow_id=flow_id,
            values=request.values,
            context=request.context,
            step_id=request.step_id,
        )
    except FlowNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except FlowStepNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (InvalidFlowError, InvalidFormError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FormNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except LLMConfigurationError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except LLMExecutionError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Unexpected error executing flow %s", flow_id)
        raise HTTPException(
            status_code=500,
            detail="Unexpected error executing flow",
        ) from exc


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "version": "1.0"}
