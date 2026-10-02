import pytest
from services.model_registry import (
    get_available_models,
    get_model_categories,
    get_default_model,
    is_valid_model,
    seed_model_registry,
)


def test_model_registry_seeder_and_fetch():
    seed_model_registry()

    all_models = get_available_models()
    assert len(all_models) > 0

    text_models = get_available_models("text")
    assert any(m["id"] == "gpt-4o-mini" for m in text_models)
    assert all(m["category"] == "text" for m in text_models)

    image_models = get_available_models("image")
    assert any(m["id"] == "leonardo-phoenix" for m in image_models)
    assert any(m["id"] == "dall-e-3" for m in image_models)
    assert all(m["category"] == "image" for m in image_models)

    audio_models = get_available_models("audio")
    assert any(m["id"] == "tts-1" for m in audio_models)
    assert any(m["id"] == "eleven_multilingual_v2" for m in audio_models)
    assert all(m["category"] == "audio" for m in audio_models)


def test_model_categories():
    categories = get_model_categories()
    cat_ids = [c["id"] for c in categories]
    assert "text" in cat_ids
    assert "image" in cat_ids
    assert "audio" in cat_ids


def test_default_model():
    default_m = get_default_model()
    assert default_m["provider"] == "openai"
    assert default_m["name"] == "gpt-4o-mini"


def test_is_valid_model():
    assert is_valid_model("gpt-4o-mini") is True
    assert is_valid_model("leonardo-phoenix") is True
    assert is_valid_model("tts-1") is True
    assert is_valid_model("non_existent_model_123") is False
