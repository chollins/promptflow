from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from dotenv import load_dotenv

from .schemas.prompt_form import PromptForm

logger = logging.getLogger(__name__)
_ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(_ENV_PATH)


def generate_form_from_prompt(user_description: str, model: str = "gpt-4o-mini") -> dict:
    """
    Uses OpenAI to generate a complete, schema-compliant PromptForm JSON definition
    based on a natural language description provided by the user.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set. Please add it to backend/.env")

    from openai import OpenAI

    client = OpenAI(api_key=api_key)

    system_instruction = (
        "You are an expert AI form and prompt engineer for PromptFlow. PromptFlow forms define input fields "
        "and prompt templates that an LLM executes.\n\n"
        "Given the user's description of what form they want to create, generate a complete, valid JSON object "
        "matching this exact schema:\n"
        "{\n"
        '  "id": "lowercase-hyphenated-slug",\n'
        '  "name": "Human Readable Name",\n'
        '  "description": "Short explanation of what this form generates or achieves",\n'
        '  "version": "1.0",\n'
        '  "fields": [\n'
        "    {\n"
        '      "id": "lowercase_snake_case_variable",\n'
        '      "label": "User-facing Field Label",\n'
        '      "type": "text" | "textarea" | "dropdown" | "radio" | "checkbox" | "date",\n'
        '      "description": "Helpful guidance or placeholder text for the input",\n'
        '      "placeholder": "Helpful guidance or placeholder text for the input",\n'
        '      "required": true | false,\n'
        '      "default": "optional default value",\n'
        '      "options": ["Option 1", "Option 2"] // only for dropdown, radio, or multi-choice checkbox\n'
        "    }\n"
        "  ],\n"
        '  "prompt": {\n'
        '    "system": "Clear instructions defining the AI persona, rules, and output style",\n'
        '    "user": "Prompt template containing {{field_id}} placeholders matching the field IDs defined in fields"\n'
        "  },\n"
        '  "model": {\n'
        '    "provider": "openai",\n'
        '    "name": "gpt-4o-mini",\n'
        '    "temperature": 0.7\n'
        "  }\n"
        "}\n\n"
        "Design Rules:\n"
        "1. Choose appropriate field types: 'textarea' for multi-line text/notes, 'dropdown' or 'radio' for multiple choice, 'text' for short input, 'checkbox' for boolean toggles or multi-choice lists, 'date' for dates.\n"
        "2. For choice fields (dropdown, radio, checkbox), provide realistic and helpful 'options' strings.\n"
        "3. Every placeholder in prompt.user like {{variable_name}} MUST correspond to a field id in fields.\n"
        "4. Keep field IDs concise, lowercase snake_case (e.g. topic, tone, target_audience).\n"
        "5. Output ONLY valid raw JSON with no Markdown backticks, explanation, or conversational text."
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": f"Create a prompt form for: {user_description}"},
        ],
        response_format={"type": "json_object"},
        temperature=0.7,
    )

    raw_json = response.choices[0].message.content or "{}"
    validated_form = PromptForm.model_validate_json(raw_json)
    return validated_form.model_dump(exclude_none=True, by_alias=True)

