# Repository Guidelines

## Project Structure & Module Organization
Primary code lives in `qwen_abstract/`.
- `main.py`: CLI entrypoint (`python main.py ...`), argument parsing, I/O, output files.
- `processor.py`: three-stage text pipeline (correct, clean, summarize).
- `llm_client.py`: HTTP client for model calls.
- `rag.py`: current RAG stub and lookup output.
- `config.py`: API/model defaults.
- `test_input.txt`: sample input for manual verification.
- `docs/plans/`: design and implementation notes.

Keep new runtime modules inside `qwen_abstract/`. Put non-code documentation in `qwen_abstract/docs/`.

## Build, Test, and Development Commands
From repository root:
- `cd qwen_abstract`
- `python -m venv .venv` then `.\.venv\Scripts\Activate.ps1`: create/activate virtualenv (Windows PowerShell).
- `pip install -r requirements.txt`: install dependencies.
- `python main.py --help`: verify CLI wiring.
- `python main.py test_input.txt -v`: run end-to-end flow and print intermediate stages.
- `echo "meeting text" | python main.py -`: test stdin mode.

No dedicated build step exists; this is a direct Python CLI project.

## Coding Style & Naming Conventions
Follow existing Python style in this repo:
- 4-space indentation, UTF-8 source files, type hints for public function signatures.
- `snake_case` for functions/variables, `PascalCase` for classes, `UPPER_CASE` for constants.
- Keep modules focused (I/O in `main.py`, orchestration in `processor.py`, API calls in `llm_client.py`).
- Prefer small, single-purpose functions and explicit error handling with clear `stderr` messages.

No formatter/linter is currently configured. If you introduce one, document the command here.

## Testing Guidelines
Automated tests are not set up yet. Use manual integration checks:
- Run `python main.py test_input.txt -v`.
- Confirm `*_corrected.txt`, `*_cleaned.txt`, and `*_summary.txt` are generated.
- Validate non-zero exit on invalid input paths.

If adding tests, use `pytest` with files named `test_*.py` under `qwen_abstract/tests/`.

## Commit & Pull Request Guidelines
Git history uses Conventional Commits (`feat:`, `fix:`, `refactor:`, `test:`). Continue that pattern.

PRs should include:
- concise problem/solution summary,
- key commands run locally,
- sample input/output paths when behavior changes,
- linked issue/task when applicable.

## Security & Configuration Tips
Treat `config.py` as environment-specific. Do not commit real internal endpoints, secrets, or tokens. Prefer environment variables for sensitive values in future changes.
