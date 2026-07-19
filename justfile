# Justfile for pyescan

# Show available commands
list:
    @just --list

# Type check the project with ty
type-check:
    uv run ty check .

# Type check with concise output (one diagnostic per line)
type-check-concise:
    uv run ty check --output-format=concise .

# Type check in watch mode (rechecks on file changes)
type-check-watch:
    uv run ty check --watch .

# Run the formatting, linting and type checking commands
qa:
    uv run ruff format .
    uv run ruff check . --fix
    uv run ty check .
    uv audit

# Check formatting, linting and type checking (no fixes, for CI)
ci:
    uv run ruff format --check .
    uv run ruff check .
    uv run ty check .
    uv audit

# Check dependency licenses
license-check:
    uv run licensecheck --format markdown --show-only-failing --ignore-packages certifi cookiecutter

# Run the formatting, linting, type checking and tests commands
qa-all:
    just qa
    uv run pytest

# Upgrade the project library and build it
up *ARGS:
    @echo "Upgrading with {{ARGS}}"
    uv lock --exclude-newer "7 days" -U {{ARGS}}
    uv audit
    uv sync --exclude-newer "7 days"
    just build

# Run all the tests, but allow for arguments to be passed
test *ARGS:
    @echo "Running with arg: {{ARGS}}"
    uv run pytest {{ARGS}}
    @echo "See htmlcov/index.html for detailed coverage report"

# Run all the tests, but on failure, drop into the debugger
pdb *ARGS:
    @echo "Running with arg: {{ARGS}}"
    uv run pytest --pdb --maxfail=10 {{ARGS}}

# Build the project, useful for checking that packaging is correct
build:
    rm -rf build
    rm -rf dist
    uv build

# remove all build, test, coverage and Python artifacts
clean:
    just clean-build
    just clean-pyc
    just clean-test

# remove build artifacts
clean-build:
    rm -fr build/
    rm -fr dist/
    rm -fr .eggs/
    find . -name '*.egg-info' -exec rm -fr {} +
    find . -name '*.egg' -exec rm -f {} +

# remove Python file artifacts
clean-pyc:
    find . -name '*.pyc' -exec rm -f {} +
    find . -name '*.pyo' -exec rm -f {} +
    find . -name '*~' -exec rm -f {} +
    find . -name '__pycache__' -exec rm -fr {} +

# remove test and coverage artifacts
clean-test:
    rm -f .coverage
    rm -f .coverage.*
    rm -fr htmlcov/
    rm -fr .pytest_cache

VERSION := `python -c 'import importlib.metadata as m; print(next((d.version for d in m.distributions() if d.metadata["Name"].lower()=="pyescan"), "0.0.0.dev0"))'`

# Print the current version of the project
version:
    @echo "Current version is {{VERSION}}"
