from typing import List, Literal, Optional

from pydantic import BaseModel, Field as PydanticField


class ModelSettings(BaseModel):
    provider: str = "openai"
    model: str = "gpt-4o-mini"
    temperature: float = 0.7


class Prompt(BaseModel):
    system: str
    user: str


class Field(BaseModel):
    id: str
    label: str
    description: Optional[str] = None
    type: Literal["text", "textarea", "checkbox", "radio", "dropdown", "hidden"]
    required: bool = False
    default: Optional[str] = None
    options: List[str] = PydanticField(default_factory=list)


class PromptForm(BaseModel):
    id: str
    name: str
    description: Optional[str] = None
    version: str = "1.0"
    fields: List[Field]
    prompt: Prompt
    model: ModelSettings = PydanticField(default_factory=ModelSettings)
