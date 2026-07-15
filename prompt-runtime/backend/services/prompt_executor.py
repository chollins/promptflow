"""Execute prompts via LangChain ChatOpenAI."""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)

_ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(_ENV_PATH)


class LLMConfigurationError(Exception):
    """Raised when required LLM configuration is missing."""


class LLMExecutionError(Exception):
    """Raised when the LLM call fails."""


def execute_prompt(
    system_prompt: str,
    user_prompt: str,
    model: str = "gpt-4o-mini",
    temperature: float = 0.7,
) -> str:
    """Invoke the configured LLM and return the text response."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise LLMConfigurationError(
            "OPENAI_API_KEY is not set. Add it to backend/.env"
        )

    logger.info(
        "LLM execution starting model=%s temperature=%s",
        model,
        temperature,
    )
    started = time.perf_counter()

    try:
        llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            api_key=api_key,
        )
        response = llm.invoke([
            ("system", system_prompt),
            ("human", user_prompt),
        ])
    except LLMConfigurationError:
        raise
    except Exception as exc:
        raise LLMExecutionError(f"LLM execution failed: {exc}") from exc

    elapsed_ms = (time.perf_counter() - started) * 1000
    logger.info("LLM execution finished in %.1f ms", elapsed_ms)
    return response.content
