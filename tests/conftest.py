"""Shared fixtures for PyeScan tests."""

import os
import tempfile
from collections.abc import Generator
from pathlib import Path

import numpy as np
import pytest
from PIL import Image as PILImage
from typer.testing import CliRunner

EXAMPLE_DATA_ROOT = os.path.join(os.path.dirname(__file__), "..", "example_data")
EXAMPLE_PAT = os.path.join(EXAMPLE_DATA_ROOT, "00046792.pat")
EXAMPLE_OCT_SDB = os.path.join(EXAMPLE_PAT, "00522105.sdb")
EXAMPLE_FAF_SDB = os.path.join(EXAMPLE_PAT, "00522107.sdb")


@pytest.fixture
def runner() -> CliRunner:
    """Create CLI test runner."""
    return CliRunner()


@pytest.fixture
def oct_sdb_path():
    """Path to example OCT sdb folder."""
    return EXAMPLE_OCT_SDB


@pytest.fixture
def faf_sdb_path():
    """Path to example FAF sdb folder."""
    return EXAMPLE_FAF_SDB


@pytest.fixture
def tmp_mask_dir(tmp_path):
    """Create temporary mask images for annotation testing."""
    mask_dir = tmp_path / "masks"
    mask_dir.mkdir()
    # Create 5 small binary mask images
    for i in range(5):
        mask = np.zeros((64, 64), dtype=np.uint8)
        # Put a simple rectangle in each mask (different position per bscan)
        mask[10 + i : 30 + i, 10:50] = 255
        img = PILImage.fromarray(mask)
        img.save(mask_dir / f"scan1_{i}.png")
    return mask_dir


@pytest.fixture
def tmp_mask_dir_with_features(tmp_path):
    """Create temporary mask images with multiple features."""
    mask_dir = tmp_path / "masks"
    for feature in ["GA", "drusen"]:
        feat_dir = mask_dir / feature
        feat_dir.mkdir(parents=True)
        for i in range(5):
            mask = np.zeros((64, 64), dtype=np.uint8)
            if feature == "GA":
                mask[10:30, 10:50] = 255
            else:
                mask[40:55, 20:40] = 255
            img = PILImage.fromarray(mask)
            img.save(feat_dir / f"scan1_{i}.png")
    return mask_dir


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
