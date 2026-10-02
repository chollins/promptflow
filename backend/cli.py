from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

try:
    from file_runner import FileRunnerError, FileRunnerValidationError, run_flow_from_file
    from runtime.flow_service import get_flow
    from runtime.form_service import get_form
except ImportError:
    from .file_runner import FileRunnerError, FileRunnerValidationError, run_flow_from_file
    from .runtime.flow_service import get_flow
    from .runtime.form_service import get_form


def collect_interactive_inputs(flow_id_or_path: str, existing_inputs: dict[str, Any] | None = None) -> dict[str, Any]:
    inputs = dict(existing_inputs or {})
    try:
        prompt_flow = get_flow(flow_id_or_path)
    except Exception:
        return inputs

    needed_fields = []
    for step in sorted(prompt_flow.steps, key=lambda s: s.sequence):
        try:
            form = get_form(step.prompt_form_id)
        except Exception:
            continue
        for field in form.fields:
            if field.type == "hidden":
                continue
            is_bound = field.id in step.input_bindings or any(
                k == field.id or k.endswith(f".{field.id}") for k in step.input_bindings.values()
            )
            if not is_bound and (field.id not in inputs or inputs[field.id] is None or str(inputs[field.id]).strip() == ""):
                if field.id not in [f.id for f in needed_fields]:
                    needed_fields.append(field)

    if not needed_fields:
        return inputs

    print("\n" + "=" * 55)
    print(f" PromptFlow Interactive Input Mode")
    print(f" Flow: {prompt_flow.name or prompt_flow.id}")
    print("=" * 55)

    for field in needed_fields:
        label = field.label or field.id
        desc = f" ({field.description})" if field.description else ""
        prompt_str = f"Enter value for '{label}'{desc}:\n> "

        val = ""
        while not val:
            try:
                val = input(prompt_str).strip()
            except (EOFError, KeyboardInterrupt):
                print("\nExecution cancelled.")
                sys.exit(1)
            if not val and not field.required:
                break
            if not val:
                print(f"Field '{field.id}' is required. Please enter a value.")

        inputs[field.id] = val

    print("=" * 55 + "\n")
    return inputs


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="promptflow",
        description="PromptFlow CLI — Run flows interactively or with local JSON input files.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Execute a flow interactively or using a local input JSON file.")
    run_parser.add_argument("flow", help="Flow ID, slug, or file path to .flow.json / .md")
    run_parser.add_argument("--input", "-i", required=False, default=None, help="Optional path to input JSON file containing form values.")
    run_parser.add_argument("--output", "-o", help="Optional output JSON file path to write results.")
    run_parser.add_argument("--model", "-m", help="Optional model override name (e.g. gpt-4o).")
    run_parser.add_argument("--provider", "-p", default="openai", help="Optional model provider (default: openai).")
    run_parser.add_argument("--interactive", action="store_true", help="Interactively prompt for input variables via CLI.")

    return parser


def main(args: list[str] | None = None) -> int:
    parser = create_parser()
    parsed_args = parser.parse_args(args)

    if parsed_args.command == "run":
        model_override = None
        if parsed_args.model:
            model_override = {
                "provider": parsed_args.provider,
                "name": parsed_args.model,
                "temperature": 0.7,
            }

        input_values: dict[str, Any] = {}
        if parsed_args.input:
            input_path = Path(parsed_args.input)
            if not input_path.is_file():
                print(f"Validation Error: Input file not found: '{parsed_args.input}'", file=sys.stderr)
                return 1
            try:
                content = input_path.read_text(encoding="utf-8")
                input_values = json.loads(content)
            except Exception as exc:
                print(f"Validation Error: Failed to read input JSON '{parsed_args.input}': {exc}", file=sys.stderr)
                return 1

        # Trigger interactive prompt scanner if --interactive is set OR no --input file was provided
        if parsed_args.interactive or not parsed_args.input:
            input_values = collect_interactive_inputs(parsed_args.flow, input_values)

        try:
            result = run_flow_from_file(
                flow=parsed_args.flow,
                input_path=None,
                output_path=parsed_args.output,
                model_override=model_override,
                input_values=input_values,
            )
            print(f"Flow '{result['flow_id']}' executed successfully!")
            if parsed_args.output:
                print(f"Output saved to: {parsed_args.output}")
            else:
                print(json.dumps(result["output"], indent=2))
            return 0
        except FileRunnerValidationError as exc:
            print(f"Validation Error: {exc}", file=sys.stderr)
            return 1
        except FileRunnerError as exc:
            print(f"Execution Error: {exc}", file=sys.stderr)
            return 1
        except Exception as exc:
            print(f"Unexpected Error: {exc}", file=sys.stderr)
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())

