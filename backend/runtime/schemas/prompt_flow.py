from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class RuntimeSettings(BaseModel):
    mode: Literal["guided", "automatic"] = "guided"
    default_review_required: bool = True


class ReviewSettings(BaseModel):
    required: bool = True
    editable: bool = True


class OutputSettings(BaseModel):
    save_as: str
    formats: list[str] = Field(default_factory=lambda: ["json"])


class FlowStep(BaseModel):
    id: str
    sequence: int
    name: str
    prompt_form_id: str
    input_bindings: dict[str, str] = Field(default_factory=dict)
    dynamic_fields: list[str] = Field(default_factory=list)
    review: ReviewSettings = Field(default_factory=ReviewSettings)
    output: OutputSettings | None = None
    next: str | None = None


class PromptFlow(BaseModel):
    id: str
    version: str
    name: str
    description: str = ""
    runtime: RuntimeSettings = Field(default_factory=RuntimeSettings)
    steps: list[FlowStep]

