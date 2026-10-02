from __future__ import annotations

import json
import pytest
from pathlib import Path

from services.adapters.leonardo_adapter import LeonardoAdapter, LeonardoAdapterError
from services.output_handlers import LeonardoOutputHandler, default_registry


def test_leonardo_adapter_aspect_ratio_resolution():
    adapter = LeonardoAdapter(api_key="mock")

    w1, h1 = adapter.resolve_dimensions({"aspect_ratio": "16:9"})
    assert (w1, h1) == (1280, 720)

    w2, h2 = adapter.resolve_dimensions({"ar": "1:1"})
    assert (w2, h2) == (1024, 1024)

    w3, h3 = adapter.resolve_dimensions({"aspect_ratio": "4:3"})
    assert (w3, h3) == (1024, 768)


def test_leonardo_adapter_missing_prompt_error():
    adapter = LeonardoAdapter(api_key="mock")
    with pytest.raises(LeonardoAdapterError, match="Missing required 'prompt'"):
        adapter.generate_image({"type": "image", "prompt": ""})


def test_leonardo_output_handler_mock_generation(tmp_path: Path):
    handler = LeonardoOutputHandler(adapter=LeonardoAdapter(api_key="mock"))

    image_spec = {
        "type": "image",
        "prompt": "A futuristic city in 2099",
        "negative_prompt": "blurry",
        "aspect_ratio": "16:9",
    }

    res = handler.process(
        raw_output=image_spec,
        save_as="city_art",
        output_dir=tmp_path,
    )

    assert res.success
    assert res.output_type == "image"
    assert len(res.artifacts) == 1
    artifact_path = Path(res.artifacts[0])
    assert artifact_path.is_file()

    meta = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert meta["aspect_ratio"] == "16:9"
    assert meta["dimensions"] == "1280x720"
    assert "city_art" in meta["generation_id"]


def test_registry_integration_leonardo(tmp_path: Path):
    res = default_registry.process_output(
        handler_name="leonardo",
        raw_output={"type": "image", "prompt": "A peaceful forest lake", "aspect_ratio": "1:1"},
        save_as="lake_art",
        output_dir=tmp_path,
    )
    assert res.success
    assert res.output_type == "image"
