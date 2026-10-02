from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"
MODELS_FILE = CONFIG_DIR / "models.json"

DEFAULT_MODEL_CONFIG: Dict[str, Any] = {
    "default_provider": "openai",
    "default_model": "gpt-4o-mini",
    "categories": [
        {"id": "text", "label": "Text & Generative LLMs"},
        {"id": "image", "label": "Picture & Image Generation"},
        {"id": "audio", "label": "Audio & Speech Synthesis"}
    ],
    "models": [
        {"id": "gpt-4o-mini", "name": "GPT-4o Mini", "provider": "openai", "category": "text", "recommended": True},
        {"id": "gpt-4o", "name": "GPT-4o", "provider": "openai", "category": "text", "recommended": False},
        {"id": "gpt-4o-2024-08-06", "name": "GPT-4o (2024-08-06)", "provider": "openai", "category": "text", "recommended": False},
        {"id": "gpt-4-turbo", "name": "GPT-4 Turbo", "provider": "openai", "category": "text", "recommended": False},
        {"id": "o3-mini", "name": "o3-Mini", "provider": "openai", "category": "text", "recommended": False},
        {"id": "o1-mini", "name": "o1-Mini", "provider": "openai", "category": "text", "recommended": False},
        {"id": "claude-3-5-sonnet", "name": "Claude 3.5 Sonnet", "provider": "anthropic", "category": "text", "recommended": False},
        {"id": "claude-3-5-haiku", "name": "Claude 3.5 Haiku", "provider": "anthropic", "category": "text", "recommended": False},
        {"id": "gemini-1.5-pro", "name": "Gemini 1.5 Pro", "provider": "google", "category": "text", "recommended": False},
        {"id": "gemini-1.5-flash", "name": "Gemini 1.5 Flash", "provider": "google", "category": "text", "recommended": False},
        {"id": "dall-e-3", "name": "DALL-E 3", "provider": "openai", "category": "image", "recommended": True},
        {"id": "dall-e-2", "name": "DALL-E 2", "provider": "openai", "category": "image", "recommended": False},
        {"id": "leonardo-phoenix", "name": "Leonardo Phoenix", "provider": "leonardo", "category": "image", "recommended": True},
        {"id": "leonardo-kino-xl", "name": "Leonardo Kino XL", "provider": "leonardo", "category": "image", "recommended": False},
        {"id": "leonardo-diffusion-xl", "name": "Leonardo Diffusion XL", "provider": "leonardo", "category": "image", "recommended": False},
        {"id": "stable-diffusion-xl-1.0", "name": "Stable Diffusion XL 1.0", "provider": "stability", "category": "image", "recommended": False},
        {"id": "tts-1", "name": "OpenAI TTS 1", "provider": "openai", "category": "audio", "recommended": True},
        {"id": "tts-1-hd", "name": "OpenAI TTS 1 HD", "provider": "openai", "category": "audio", "recommended": False},
        {"id": "whisper-1", "name": "Whisper 1", "provider": "openai", "category": "audio", "recommended": True},
        {"id": "eleven_multilingual_v2", "name": "ElevenLabs Multilingual v2", "provider": "elevenlabs", "category": "audio", "recommended": True},
        {"id": "eleven_turbo_v2_5", "name": "ElevenLabs Turbo v2.5", "provider": "elevenlabs", "category": "audio", "recommended": False}
    ]
}


def seed_model_registry() -> Path:
    """Ensures models.json exists and seeds AIModel DB table if app context is active."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    if not MODELS_FILE.is_file():
        MODELS_FILE.write_text(json.dumps(DEFAULT_MODEL_CONFIG, indent=2), encoding="utf-8")
        logger.info("Created default model registry file at %s", MODELS_FILE)

    try:
        from flask import has_app_context
        if has_app_context():
            from extensions import db
            from models import AIModel
            for item in DEFAULT_MODEL_CONFIG["models"]:
                is_active = item.get("is_active", item.get("provider") == "openai")
                existing = AIModel.query.filter_by(model_id=item["id"]).first()
                if not existing:
                    m = AIModel(
                        model_id=item["id"],
                        name=item["name"],
                        provider=item["provider"],
                        category=item["category"],
                        description=item.get("description"),
                        is_active=is_active,
                        is_recommended=item.get("recommended", False),
                    )
                    db.session.add(m)
                else:
                    # Sync active state by provider if newly initialized
                    if item.get("provider") != "openai":
                        existing.is_active = False
            db.session.commit()
    except Exception as exc:
        logger.warning("Could not seed AIModel DB table: %s", exc)

    return MODELS_FILE


def load_model_registry() -> Dict[str, Any]:
    """Loads the model registry configuration file."""
    if not MODELS_FILE.is_file():
        seed_model_registry()
    try:
        content = MODELS_FILE.read_text(encoding="utf-8")
        return json.loads(content)
    except Exception as exc:
        logger.warning("Failed to parse models.json (%s). Falling back to defaults.", exc)
        return DEFAULT_MODEL_CONFIG


def get_available_models(category: Optional[str] = None, include_inactive: bool = False) -> List[Dict[str, Any]]:
    """Returns list of models from DB (or fallback models.json) optionally filtered by category and active status."""
    try:
        from flask import has_app_context
        if has_app_context():
            from models import AIModel
            query = AIModel.query
            if not include_inactive:
                query = query.filter_by(is_active=True)
            if category:
                query = query.filter_by(category=category)
            db_models = query.order_by(AIModel.name.asc()).all()
            if db_models:
                return [
                    {
                        "id": m.model_id,
                        "name": m.name,
                        "provider": m.provider,
                        "category": m.category,
                        "description": m.description,
                        "is_active": m.is_active,
                        "recommended": m.is_recommended,
                    }
                    for m in db_models
                ]
    except Exception as exc:
        logger.debug("Falling back to models.json: %s", exc)

    data = load_model_registry()
    models = data.get("models", [])
    if category:
        models = [m for m in models if m.get("category") == category]
    if not include_inactive:
        models = [m for m in models if m.get("is_active", True) is not False]
    return models


def get_model_categories() -> List[Dict[str, str]]:
    """Returns list of categories."""
    data = load_model_registry()
    return data.get("categories", [])


def get_default_model() -> Dict[str, str]:
    """Returns the default model configuration dict."""
    data = load_model_registry()
    return {
        "provider": data.get("default_provider", "openai"),
        "name": data.get("default_model", "gpt-4o-mini"),
        "temperature": 0.7
    }


def is_valid_model(model_id: str) -> bool:
    """Checks whether a given model_id exists in the registry."""
    models = get_available_models(include_inactive=True)
    return any(m.get("id") == model_id for m in models)


def create_db_model(data: Dict[str, Any]):
    """Creates a new model in the DB registry."""
    from extensions import db
    from models import AIModel
    model_id = data.get("id") or data.get("model_id")
    if not model_id:
        raise ValueError("Model ID is required.")
    existing = AIModel.query.filter_by(model_id=model_id).first()
    if existing:
        raise ValueError(f"Model with ID '{model_id}' already exists.")

    m = AIModel(
        model_id=model_id,
        name=data.get("name", model_id),
        provider=data.get("provider", "openai"),
        category=data.get("category", "text"),
        description=data.get("description"),
        is_active=bool(data.get("is_active", True)),
        is_recommended=bool(data.get("is_recommended", False)),
    )
    db.session.add(m)
    db.session.commit()
    return m


def update_db_model(model_id: str, data: Dict[str, Any]):
    """Updates an existing model in the DB registry."""
    from extensions import db
    from models import AIModel
    m = AIModel.query.filter((AIModel.id == model_id) | (AIModel.model_id == model_id)).first()
    if not m:
        raise ValueError(f"Model '{model_id}' not found.")

    if "name" in data:
        m.name = data["name"]
    if "provider" in data:
        m.provider = data["provider"]
    if "category" in data:
        m.category = data["category"]
    if "description" in data:
        m.description = data["description"]
    if "is_active" in data:
        m.is_active = bool(data["is_active"])
    if "is_recommended" in data:
        m.is_recommended = bool(data["is_recommended"])

    db.session.commit()
    return m


def delete_db_model(model_id: str):
    """Deletes a model from the DB registry."""
    from extensions import db
    from models import AIModel
    m = AIModel.query.filter((AIModel.id == model_id) | (AIModel.model_id == model_id)).first()
    if not m:
        raise ValueError(f"Model '{model_id}' not found.")
    db.session.delete(m)
    db.session.commit()

