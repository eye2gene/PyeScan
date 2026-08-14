"""Smoke-test an installed PyeScan distribution, not the source checkout."""

from importlib.metadata import version
from subprocess import run


def main() -> None:
    """Verify imports, metadata, and the installed command-line entry point."""
    from pyescan.CELoader import load_record_from_CE
    from pyescan.core import BaseScan, OCTScan
    from pyescan.tools.cli import app

    assert version("pyescan") != "0.0.0.dev0"
    assert callable(load_record_from_CE)
    assert BaseScan is not None
    assert OCTScan is not None
    assert app.info.name == "pyescan"

    cli = run(["pyescan", "--help"], check=True, capture_output=True, text=True)
    assert "PyeScan CLI tools" in cli.stdout


if __name__ == "__main__":
    main()
