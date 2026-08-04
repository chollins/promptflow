from __future__ import annotations

import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)
_ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(_ENV_PATH)


class LLMConfigurationError(Exception):
    pass


class LLMExecutionError(Exception):
    pass


def execute_prompt(system_prompt: str, user_prompt: str, model: str = "gpt-4o-mini", temperature: float = 0.7) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise LLMConfigurationError("OPENAI_API_KEY is not set. Add it to backend/.env")

    try:
        llm = ChatOpenAI(model=model, temperature=temperature, api_key=api_key)
        response = llm.invoke([("system", system_prompt), ("human", user_prompt)])
    except Exception as exc:
        raise LLMExecutionError(f"LLM execution failed: {exc}") from exc

    return response.content

