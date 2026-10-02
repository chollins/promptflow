from __future__ import annotations

import json
import pytest
from pathlib import Path

from file_runner import run_flow_from_file, FileRunnerValidationError
from cli import main as cli_main
import promptflow


def test_cli_execution_valid_input(tmp_path: Path, monkeypatch):
    # Setup test input JSON file
    input_file = tmp_path / "input.json"
    input_file.write_text(
        json.dumps(
            {
                "company": "Acme Corp",
                "industry": "Technology",
                "tone": "Professional",
                "prospect_name": "Jane Doe",
                "job_title": "CTO",
                "goal": "Improve productivity",
                "length": "Short",
            }
        ),
        encoding="utf-8",
    )

    output_file = tmp_path / "output.json"

    # Mock execute_prompt to return structured mock response without calling external LLM API
    def mock_execute_prompt(*args, **kwargs):
        return json.dumps({"summary": "Acme Corp summary", "email": "Hello Jane"})

    monkeypatch.setattr("runtime.flow_executor.execute_prompt", mock_execute_prompt)

    # Run flow from file
    result = run_flow_from_file(
        flow="client_assessment",
        input_path=input_file,
        output_path=output_file,
    )

    assert result["status"] == "completed"
    assert result["flow_id"] == "client_assessment"
    assert output_file.is_file()

    saved_output = json.loads(output_file.read_text(encoding="utf-8"))
    assert saved_output["flow_id"] == "client_assessment"


def test_promptflow_python_api(tmp_path: Path, monkeypatch):
    input_file = tmp_path / "input_api.json"
    input_file.write_text(
        json.dumps(
            {
                "company": "Beta Inc",
                "industry": "Healthcare",
                "tone": "Friendly",
                "prospect_name": "John Smith",
                "job_title": "CEO",
                "goal": "Reduce costs",
                "length": "Medium",
            }
        ),
        encoding="utf-8",
    )
    output_file = tmp_path / "output_api.json"

    def mock_execute_prompt(*args, **kwargs):
        return json.dumps({"result": "Beta Inc result"})

    monkeypatch.setattr("runtime.flow_executor.execute_prompt", mock_execute_prompt)

    res = promptflow.run(
        flow="client_assessment",
        input=input_file,
        output=output_file,
        model="gpt-4o",
    )

    assert res["status"] == "completed"
    assert output_file.is_file()


def test_missing_input_file_error():
    with pytest.raises(FileRunnerValidationError, match="Input file not found"):
        run_flow_from_file(
            flow="client_assessment",
            input_path="./non_existent_file.json",
        )


def test_invalid_json_input_error(tmp_path: Path):
    invalid_file = tmp_path / "bad.json"
    invalid_file.write_text("{ unclosed json", encoding="utf-8")

    with pytest.raises(FileRunnerValidationError, match="Invalid JSON"):
        run_flow_from_file(
            flow="client_assessment",
            input_path=invalid_file,
        )


def test_missing_required_fields_error(tmp_path: Path):
    incomplete_file = tmp_path / "incomplete.json"
    incomplete_file.write_text(json.dumps({"company": "Only Company"}), encoding="utf-8")

    with pytest.raises(FileRunnerValidationError, match="Missing required fields"):
        run_flow_from_file(
            flow="client_assessment",
            input_path=incomplete_file,
        )


def test_cli_main_entrypoint(tmp_path: Path, monkeypatch):
    input_file = tmp_path / "cli_input.json"
    input_file.write_text(
        json.dumps(
            {
                "company": "CLI Corp",
                "industry": "Finance",
                "tone": "Executive",
                "prospect_name": "Alice",
                "job_title": "CFO",
                "goal": "Security",
                "length": "Short",
            }
        ),
        encoding="utf-8",
    )
    output_file = tmp_path / "cli_output.json"

    def mock_execute_prompt(*args, **kwargs):
        return json.dumps({"result": "CLI output"})

    monkeypatch.setattr("runtime.flow_executor.execute_prompt", mock_execute_prompt)

    code = cli_main(["run", "client_assessment", "--input", str(input_file), "--output", str(output_file)])
    assert code == 0
    assert output_file.is_file()


def test_cli_interactive_input(tmp_path: Path, monkeypatch):
    output_file = tmp_path / "interactive_output.json"

    # Mock python input() answers
    user_inputs = iter([
        "Acme Interactive",  # company
        "Retail",            # industry
        "Friendly",          # tone
        "Bob",               # prospect_name
        "VP Sales",          # job_title
        "Grow revenue",      # goal
        "Short",             # length
    ])
    monkeypatch.setattr("builtins.input", lambda prompt: next(user_inputs))

    def mock_execute_prompt(*args, **kwargs):
        return json.dumps({"result": "Interactive CLI result"})

    monkeypatch.setattr("runtime.flow_executor.execute_prompt", mock_execute_prompt)

    code = cli_main(["run", "client_assessment", "--output", str(output_file)])
    assert code == 0
    assert output_file.is_file()
    saved = json.loads(output_file.read_text(encoding="utf-8"))
    assert saved["context"]["company"] == "Acme Interactive"


def test_cli_markdown_file_execution(tmp_path: Path, monkeypatch):
    output_file = tmp_path / "md_output.json"

    user_inputs = iter([
        "Acme Corp",
        "We make AI software.",
    ])
    monkeypatch.setattr("builtins.input", lambda prompt: next(user_inputs))

    def mock_execute_prompt(*args, **kwargs):
        return "Generated summary result"

    monkeypatch.setattr("runtime.flow_executor.execute_prompt", mock_execute_prompt)

    code = cli_main(["run", "prompts/test_prompt.md", "--output", str(output_file)])
    assert code == 0
    assert output_file.is_file()
    saved = json.loads(output_file.read_text(encoding="utf-8"))
    assert saved["flow_id"] == "test_prompt"
    assert saved["context"]["business_name"] == "Acme Corp"


