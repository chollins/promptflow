from __future__ import annotations

import json
import pytest
from pathlib import Path

from services.output_handlers import (
    JSONOutputHandler,
    MarkdownOutputHandler,
    OutputHandlerRegistry,
    default_registry,
    transform_json_to_markdown,
)


def test_json_output_handler(tmp_path: Path):
    handler = JSONOutputHandler()
    res = handler.process(
        raw_output={"title": "Test Title", "status": "ok"},
        save_as="test_json",
        output_dir=tmp_path,
    )
    assert res.success
    assert res.output_type == "json"
    assert len(res.artifacts) == 1
    artifact_file = Path(res.artifacts[0])
    assert artifact_file.is_file()

    content = json.loads(artifact_file.read_text(encoding="utf-8"))
    assert content["title"] == "Test Title"


def test_markdown_output_handler(tmp_path: Path):
    handler = MarkdownOutputHandler()
    raw_md = "# Title\n\nThis is a test markdown string."
    res = handler.process(
        raw_output=raw_md,
        save_as="test_doc",
        output_dir=tmp_path,
    )
    assert res.success
    assert res.output_type == "markdown"
    assert len(res.artifacts) == 1
    artifact_file = Path(res.artifacts[0])
    assert artifact_file.is_file()
    assert "# Title" in artifact_file.read_text(encoding="utf-8")


def test_json_to_markdown_transformation(tmp_path: Path):
    handler = MarkdownOutputHandler()
    raw_json = {"product": "AI Assistant", "audience": "Developers"}
    template = "# {{product}}\n\nDesigned specifically for {{audience}}."

    res = handler.process(
        raw_output=raw_json,
        options={"template": template},
        save_as="transformed",
        output_dir=tmp_path,
    )

    assert res.success
    artifact_file = Path(res.artifacts[0])
    text = artifact_file.read_text(encoding="utf-8")
    assert text == "# AI Assistant\n\nDesigned specifically for Developers."


def test_output_handler_registry(tmp_path: Path):
    registry = OutputHandlerRegistry()
    assert "json" in registry.list_handlers()
    assert "markdown" in registry.list_handlers()
    assert "leonardo" in registry.list_handlers()
    assert "image" in registry.list_handlers()

    res = registry.process_output(
        handler_name="json",
        raw_output={"test": "data"},
        save_as="reg_test",
        output_dir=tmp_path,
    )
    assert res.success
    assert Path(res.artifacts[0]).is_file()


def test_unregistered_handler():
    res = default_registry.process_output("nonexistent_handler", "data")
    assert not res.success
    assert "is not registered" in res.error
