---
inclusion: auto
---

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
