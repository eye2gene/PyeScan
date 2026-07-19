---
inclusion: auto
---

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
