import tempfile
from collections.abc import Generator
from pathlib import Path

import pytest
from typer.testing import CliRunner


@pytest.fixture
def runner() -> CliRunner:
    """Create CLI test runner."""
    return CliRunner()


def pytest_configure(config: pytest.Config) -> None:
    """Dynamically add filterwarnings rules for pytest when suppression is desired.

    Using addinivalue_line ensures pytest's own warnings plugin honours the filters
    across the entire test session, without modifying each test file.
    """
    rules = [
        # NOTE: we should know the warnings and address them
        # Apply warning filters here instead of in pyproject.toml
        # so IDEs (e.g., VS Code, PyCharm) pick them up correctly.
        "error",  # it will make pytest FAIL if an unknown warn is raised
        # Runtime warnings in linear algebra
        # "ignore:.*invalid value.*:RuntimeWarning",
        # "ignore:.*divide by zero.*:RuntimeWarning",
    ]
    for rule in rules:
        config.addinivalue_line("filterwarnings", rule)


@pytest.fixture
def tmp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)
