"""
Shared fixtures for PyeScan tests.
"""
import os
import pytest
import numpy as np
from PIL import Image as PILImage

EXAMPLE_DATA_ROOT = os.path.join(os.path.dirname(__file__), '..', 'example_data')
EXAMPLE_PAT = os.path.join(EXAMPLE_DATA_ROOT, '00046792.pat')
EXAMPLE_OCT_SDB = os.path.join(EXAMPLE_PAT, '00522105.sdb')
EXAMPLE_FAF_SDB = os.path.join(EXAMPLE_PAT, '00522107.sdb')


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
        mask[10 + i:30 + i, 10:50] = 255
        img = PILImage.fromarray(mask)
        img.save(mask_dir / f"scan1_{i}.png")

    return mask_dir


@pytest.fixture
def tmp_mask_dir_with_features(tmp_path):
    """Create temporary mask images with multiple features."""
    mask_dir = tmp_path / "masks"

    for feature in ['GA', 'drusen']:
        feat_dir = mask_dir / feature
        feat_dir.mkdir(parents=True)
        for i in range(5):
            mask = np.zeros((64, 64), dtype=np.uint8)
            if feature == 'GA':
                mask[10:30, 10:50] = 255
            else:
                mask[40:55, 20:40] = 255
            img = PILImage.fromarray(mask)
            img.save(feat_dir / f"scan1_{i}.png")

    return mask_dir
