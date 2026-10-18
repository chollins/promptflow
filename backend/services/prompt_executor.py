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


import json
import re


def extract_clean_image_prompt(system_prompt: str, user_prompt: str) -> tuple[str, str, str]:
    """
    Extracts a clean image prompt, negative prompt, and aspect ratio from user_prompt
    and system_prompt, ensuring no JSON schemas, curly braces, or template instructions
    leak to Leonardo or DALL-E image generation models.
    """
    trimmed = user_prompt.strip()

    # 1. Check if user_prompt is directly valid JSON or contains a JSON code block
    json_candidate = None
    if trimmed.startswith("{") and trimmed.endswith("}"):
        try:
            json_candidate = json.loads(trimmed)
        except Exception:
            pass
    elif "```" in trimmed:
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", trimmed, re.DOTALL)
        if match:
            try:
                json_candidate = json.loads(match.group(1))
            except Exception:
                pass

    if isinstance(json_candidate, dict):
        p = json_candidate.get("prompt", "")
        if p and not str(p).strip().startswith("<"):
            neg = json_candidate.get("negative_prompt", "")
            ar = json_candidate.get("aspect_ratio") or json_candidate.get("ar", "16:9")
            return str(p).strip(), str(neg).strip(), str(ar).strip()

    # 2. Extract key-value pairs if present (e.g. Subject:, Style:, Lighting:, etc.)
    subject_m = re.search(r"(?i)^Subject:\s*(.+)$", user_prompt, re.MULTILINE)
    style_m = re.search(r"(?i)^Style:\s*(.+)$", user_prompt, re.MULTILINE)
    lighting_m = re.search(r"(?i)^Lighting:\s*(.+)$", user_prompt, re.MULTILINE)
    camera_m = re.search(r"(?i)^Camera Details:\s*(.+)$", user_prompt, re.MULTILINE)
    aspect_m = re.search(r"(?i)^Aspect Ratio:\s*(.+)$", user_prompt, re.MULTILINE)
    negative_m = re.search(r"(?i)^Negative Prompt:\s*(.+)$", user_prompt, re.MULTILINE)

    if subject_m and subject_m.group(1).strip() and not subject_m.group(1).strip().startswith("{{"):
        parts = []
        subject_val = subject_m.group(1).strip()
        parts.append(subject_val)

        if style_m:
            s_val = style_m.group(1).strip()
            if s_val and not s_val.startswith("{{"):
                parts.append(s_val)

        if lighting_m:
            l_val = lighting_m.group(1).strip()
            if l_val and not l_val.startswith("{{"):
                parts.append(l_val)

        if camera_m:
            c_val = camera_m.group(1).strip()
            if c_val and not c_val.startswith("{{"):
                parts.append(c_val)

        parts.append("highly detailed, 8k resolution, photorealistic")
        clean_prompt = ", ".join(parts)

        negative_prompt = ""
        if negative_m:
            neg_val = negative_m.group(1).strip()
            if neg_val and not neg_val.startswith("{{"):
                negative_prompt = neg_val

        ar_val = "16:9"
        if aspect_m:
            raw_ar = aspect_m.group(1).strip()
            if raw_ar and not raw_ar.startswith("{{"):
                ar_val = raw_ar

        return clean_prompt, negative_prompt, ar_val

    # 3. Clean boilerplate instructions, schemas, and markdown headers
    cleaned = user_prompt
    cleaned = re.sub(r"```[\s\S]*?```", "", cleaned)
    cleaned = re.sub(r"\{[\s\S]*?\}", "", cleaned)
    cleaned = re.sub(r"(?i)create a detailed image generation.*?:?", "", cleaned)
    cleaned = re.sub(r"(?i)output a valid json.*?:?", "", cleaned)
    cleaned = re.sub(r"(?i)return only a valid.*?:?", "", cleaned)
    cleaned = re.sub(r"<[^>]+>", "", cleaned)
    cleaned = re.sub(r"^#+\s+.*$", "", cleaned, flags=re.MULTILINE)

    cleaned = " ".join(cleaned.split())
    if not cleaned or cleaned.startswith("{{"):
        cleaned = "A high-quality detailed photograph"

    return cleaned, "", "16:9"


def execute_prompt(system_prompt: str, user_prompt: str, model: str = "gpt-4o-mini", temperature: float = 0.7) -> str:
    # Handle Leonardo.AI image generation models
    if "leonardo" in model.lower():
        try:
            try:
                from services.adapters.leonardo_adapter import LeonardoAdapter
            except ImportError:
                from .adapters.leonardo_adapter import LeonardoAdapter

            adapter = LeonardoAdapter()
            clean_prompt, neg_prompt, ar = extract_clean_image_prompt(system_prompt, user_prompt)

            res = adapter.generate_image({
                "prompt": clean_prompt,
                "negative_prompt": neg_prompt,
                "aspect_ratio": ar,
                "model": model,
            })
            url = res.get("url") or ""
            return json.dumps({
                "type": "image",
                "url": url,
                "prompt": clean_prompt,
                "negative_prompt": neg_prompt,
                "aspect_ratio": ar,
                "model": model,
                "generation_id": res.get("generation_id"),
                "status": res.get("status", "completed"),
            }, indent=2)
        except Exception as exc:
            raise LLMExecutionError(f"Leonardo image generation failed: {exc}") from exc

    # OpenAI-based models (DALL-E & ChatOpenAI)
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise LLMConfigurationError("OPENAI_API_KEY is not set. Add it to backend/.env")

    # Handle DALL-E image generation models
    if "dall-e" in model.lower() or "dalle" in model.lower():
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            dalle_model = "dall-e-3" if "3" in model else ("dall-e-2" if "2" in model else "dall-e-3")
            
            clean_prompt, _, _ = extract_clean_image_prompt(system_prompt, user_prompt)
            clean_prompt = clean_prompt[:1000]

            response = client.images.generate(
                model=dalle_model,
                prompt=clean_prompt,
                n=1,
                size="1024x1024",
            )
            image_url = response.data[0].url if response.data else ""
            return json.dumps({
                "type": "image",
                "url": image_url,
                "prompt": clean_prompt,
                "model": dalle_model
            }, indent=2)
        except Exception as exc:
            raise LLMExecutionError(f"DALL-E image generation failed: {exc}") from exc

    try:
        llm = ChatOpenAI(model=model, temperature=temperature, api_key=api_key)
        response = llm.invoke([("system", system_prompt), ("human", user_prompt)])
    except Exception as exc:
        raise LLMExecutionError(f"LLM execution failed: {exc}") from exc

    return response.content
