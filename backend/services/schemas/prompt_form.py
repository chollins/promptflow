from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ModelSettings(BaseModel):
    provider: str = "openai"
    name: str = "gpt-4o-mini"
    temperature: float = 0.7


class Prompt(BaseModel):
    system: str
    user: str


class DataSource(BaseModel):
    type: str
    step_id: str | None = None
    path: str | None = None


class FieldSchema(BaseModel):
    id: str
    label: str
    description: str | None = None
    type: Literal["text", "textarea", "date", "checkbox", "radio", "dropdown", "hidden"]
    required: bool = False
    default: str | None = None
    options: list[str] = Field(default_factory=list)
    data_source: DataSource | None = None


class OutputSchema(BaseModel):
    type: str
    schema_: dict[str, Any] | None = Field(default=None, alias="schema")

    model_config = {"populate_by_name": True}


class ExecutionSettings(BaseModel):
    mode: str = "interactive"


class PromptForm(BaseModel):
    id: str
    name: str
    description: str | None = None
    version: str = "1.0"
    fields: list[FieldSchema]
    prompt: Prompt
    model: ModelSettings = Field(default_factory=ModelSettings)
    output: OutputSchema | None = None
    execution: ExecutionSettings | None = None
