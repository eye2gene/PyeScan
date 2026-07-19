---
inclusion: auto
---

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
