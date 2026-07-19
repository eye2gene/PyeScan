"""Top-level package for PyeScan."""

import os
from importlib import metadata

try:
    __version__ = metadata.version(__name__)
except metadata.PackageNotFoundError:
    __version__ = "0.0.0.dev0"

SUPPRESS_WARNINGS = os.getenv("SHOW_WARNINGS", "").lower() not in (
    "1",
    "true",
    "yes",
    "on",
)
