from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ModelSettings(BaseModel):
    provider: str = "openai"
    name: str = "gpt-4o-mini"
    temperature: float = 0.7


class Prompt(BaseModel):
    system: str
    user: str


class FieldSchema(BaseModel):
    id: str
    label: str
    description: str | None = None
    type: Literal["text", "textarea", "checkbox", "radio", "dropdown", "hidden"]
    required: bool = False
    default: str | None = None
    options: list[str] = Field(default_factory=list)


class PromptForm(BaseModel):
    id: str
    name: str
    description: str | None = None
    version: str = "1.0"
    fields: list[FieldSchema]
    prompt: Prompt
    model: ModelSettings = Field(default_factory=ModelSettings)
