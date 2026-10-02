# PromptFlow Notebooks & CLI Examples

This folder contains Jupyter Notebooks and tutorial scripts for orchestrating PromptFlow workflows programmatically.

## Files

- [`promptflow_cli_orchestration.ipynb`](file:///c:/Projects/promptflow/notebooks/promptflow_cli_orchestration.ipynb): Complete tutorial demonstrating CLI and Python API flow orchestration.

## Features Covered

1. **Python API Execution**: Using `run_flow_from_file(flow, input_values)` to run `.md` and `.flow.json` workflows without a database.
2. **Model Overrides**: Overriding models and providers dynamically (`--model gpt-4o-mini --provider openai`).
3. **Multi-Step Flow Execution**: Running multi-step markdown prompt flows.
4. **CLI Subprocess Execution**: Executing `python cli.py run` with `--input` JSON files and `--output` JSON targets.
