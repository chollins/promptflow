from __future__ import annotations

import pytest
from services.markdown_prompt_parser import (
    MarkdownPromptParseError,
    parse_markdown_prompts,
)
from services.form_service import import_forms_from_markdown
from models import Form


def test_parse_single_markdown_prompt():
    md = """
<!-- PROMPT: company_summary -->
Summarize {{company}} in the {{industry}} industry.
"""
    forms = parse_markdown_prompts(md)
    assert len(forms) == 1
    form = forms[0]
    assert form.id == "company_summary"
    assert form.name == "Company Summary"
    assert len(form.fields) == 2
    assert [f.id for f in form.fields] == ["company", "industry"]
    assert form.prompt.user == "Summarize {{company}} in the {{industry}} industry."


def test_parse_multiple_markdown_prompts_with_system():
    md = """
<!-- PROMPT: company_summary -->
Summarize {{company}} in the {{industry}} industry.

<!-- PROMPT: marketing_copy -->
<!-- SYSTEM: You are an expert marketing copywriter. -->
Write marketing copy for {{product}} targeting {{audience}}.
"""
    forms = parse_markdown_prompts(md)
    assert len(forms) == 2

    f1 = forms[0]
    assert f1.id == "company_summary"
    assert f1.prompt.system == "You are a helpful AI assistant."

    f2 = forms[1]
    assert f2.id == "marketing_copy"
    assert f2.prompt.system == "You are an expert marketing copywriter."
    assert [f.id for f in f2.fields] == ["product", "audience"]


def test_duplicate_prompt_id_error():
    md = """
<!-- PROMPT: summary -->
First summary {{var1}}.

<!-- PROMPT: summary -->
Second summary {{var2}}.
"""
    with pytest.raises(MarkdownPromptParseError, match="Duplicate prompt ID 'summary'"):
        parse_markdown_prompts(md)


def test_no_prompt_markers_error():
    md = "Just normal text without any prompt markers."
    with pytest.raises(MarkdownPromptParseError, match="No valid '<!-- PROMPT: prompt_id -->' markers"):
        parse_markdown_prompts(md)


def test_empty_content_error():
    with pytest.raises(MarkdownPromptParseError, match="Markdown content is empty"):
        parse_markdown_prompts("   ")


def test_import_forms_from_markdown_db():
    from app import create_app
    from extensions import db
    from config import Config
    from services.form_service import get_form

    class TestConfig(Config):
        TESTING = True
        SQLALCHEMY_DATABASE_URI = "sqlite:///:memory:"

    app = create_app(TestConfig)
    md = """
<!-- PROMPT: test_summary -->
Generate a report for {{project_name}} managed by {{lead_person}}.
"""
    with app.app_context():
        db.create_all()
        db_forms = import_forms_from_markdown(md)
        assert len(db_forms) == 1
        record = Form.query.filter_by(slug="test_summary").first() or Form.query.filter_by(slug="test-summary").first()
        assert record is not None
        assert record.name == "Test Summary"

        # Verify get_form works both with underscore and hyphen
        form_by_underscore = get_form("test_summary")
        assert form_by_underscore.id == "test_summary"
        form_by_hyphen = get_form("test-summary")
        assert form_by_hyphen.id == "test_summary"


