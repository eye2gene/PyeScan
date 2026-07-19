# Python coding standards

When generating Python code (modules, classes, functions):

- Include Google-style docstrings.
- Follow Ruff defaults, enforcing rule D212 and line-length limits.
- Use modern type annotations (Python >=3.12).
- Do not use `from typing import ...`.
- Use 4 spaces for indentation.
- Output code only, no explanations.

When generating tests:

- Use pytest.
- Assume pytest is already installed (no setup instructions).
- Include complete and sophisticated unit tests with proper setup/teardown if needed.
- Cover important edge cases.
- Add Python 3.12 type annotations for all functions (no `from typing import`).
- Always specify return types.
- Each test function must include a one-line docstring describing its purpose.
- Output code only, no explanations.

# Git commit message standards

When generating Git commit messages:

- Generate commit messages only from the output of `git diff --cached`.
- Treat the staged diff as the single source of truth.
- If the staged diff is missing, empty, truncated, or ambiguous, ask for the complete output of `git diff --cached` instead of guessing.
- Use the Conventional Commits specification.
- Output only the commit subject (one line, no body, no explanations).
- Keep it concise (preferably <=72 characters).
- Use the imperative mood (e.g. "add", "fix", "remove").
- Choose the most appropriate type (`feat`, `fix`, `refactor`, `perf`, `docs`, `test`, `build`, `ci`, `chore`, `style`, `revert`).
- Prefer the most specific valid Conventional Commit type. Do not use `chore` when another type (`refactor`, `test`, `docs`, `build`, `ci`, etc.) more accurately describes the staged changes.
- Select the commit type based on semantic intent, not file type. For example, updating dependencies without changing behaviour is typically `chore(deps): ...`, not `feat` or `fix`.
- Add a scope only when it meaningfully improves clarity.
- Append `!` only for breaking changes.
- Do not end the subject with a period.

# Command safety

## AWS CLI

Never execute `aws` commands automatically. Always present them to the user and ask them to run manually. This applies to any command that starts with `aws` or invokes the AWS CLI in any form.

## Destructive rm commands

Never automatically run `rm -rf` or `rm -r` on broad or sensitive paths. This includes:

- Home directory (`~/`, `$HOME`)
- Root or top-level system paths (`/`, `/usr`, `/etc`, `/var`, `/opt`, `/mnt`)
- Any path that could contain large amounts of user data (e.g., `~/Documents`, `~/Projects`, `/mnt/data*`)
- Any recursive delete where the target is a variable or glob that could expand unexpectedly

For these cases, present the command and ask the user to run it manually.

Deleting small, specific files or build artifacts (e.g., `rm -rf node_modules`, `rm -rf .venv`, `rm -rf dist/`) within a project directory is fine.
