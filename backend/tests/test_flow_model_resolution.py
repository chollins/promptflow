from __future__ import annotations

import pytest
from models import Flow
from runtime.schemas.prompt_flow import PromptFlow
from runtime.schemas.prompt_form import ModelSettings, PromptForm, FieldSchema, Prompt
from services.flow_executor import resolve_effective_model


def test_model_resolution_step_override():
    """Form with explicit model overrides Flow-level model configuration."""
    flow = PromptFlow(
        id="test_flow",
        version="1.0",
        name="Test Flow",
        model=ModelSettings(provider="openai", name="gpt-4o", temperature=0.5),
        steps=[],
    )
    form = PromptForm(
        id="test_form",
        name="Test Form",
        fields=[FieldSchema(id="q", label="Question", type="text")],
        prompt=Prompt(system="sys", user="{{q}}"),
        model=ModelSettings(provider="openai", name="gpt-4-turbo", temperature=0.2),
    )

    resolved = resolve_effective_model(flow, form)
    assert resolved.name == "gpt-4-turbo"
    assert resolved.temperature == 0.2


def test_model_resolution_flow_default():
    """Form without model inherits Flow-level model configuration."""
    flow = PromptFlow(
        id="test_flow",
        version="1.0",
        name="Test Flow",
        model=ModelSettings(provider="openai", name="gpt-4o", temperature=0.3),
        steps=[],
    )
    # Form created from dict without "model" specified
    form_data = {
        "id": "test_form",
        "name": "Test Form",
        "fields": [{"id": "q", "label": "Question", "type": "text"}],
        "prompt": {"system": "sys", "user": "{{q}}"},
    }
    form = PromptForm.model_validate(form_data)

    resolved = resolve_effective_model(flow, form)
    assert resolved.name == "gpt-4o"
    assert resolved.temperature == 0.3


def test_model_resolution_system_default():
    """When neither Flow nor Form defines a model, fallback to system default."""
    flow = PromptFlow(
        id="test_flow",
        version="1.0",
        name="Test Flow",
        model=None,
        steps=[],
    )
    form_data = {
        "id": "test_form",
        "name": "Test Form",
        "fields": [{"id": "q", "label": "Question", "type": "text"}],
        "prompt": {"system": "sys", "user": "{{q}}"},
    }
    form = PromptForm.model_validate(form_data)

    resolved = resolve_effective_model(flow, form)
    assert resolved.name == "gpt-4o-mini"
    assert resolved.temperature == 0.7
