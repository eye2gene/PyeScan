"""
Tests for annotation loading and validation.
"""

import numpy as np
import pandas as pd
import pytest

from pyescan.annotation_loader import (
    _build_annotation_from_array,
    load_annotation_from_df,
    load_annotation_from_folder,
)
from pyescan.core.annotation import AnnotationOCT


class TestAnnotationFromDataFrame:
    """Tests for loading annotations from DataFrames."""

    def test_load_basic(self, tmp_mask_dir):
        file_paths = [str(tmp_mask_dir / f"scan1_{i}.png") for i in range(5)]
        df = pd.DataFrame(
            {
                "file_path": file_paths,
                "bscan_index": list(range(5)),
            }
        )
        ann = load_annotation_from_df(df)
        assert isinstance(ann, AnnotationOCT)
        assert ann.data.shape == (5, 64, 64)

    def test_load_with_features(self, tmp_mask_dir):
        file_paths = [str(tmp_mask_dir / f"scan1_{i}.png") for i in range(5)]
        df = pd.DataFrame(
            {
                "file_path": file_paths * 2,
                "bscan_index": list(range(5)) * 2,
                "feature": ["GA"] * 5 + ["drusen"] * 5,
            }
        )
        result = load_annotation_from_df(df, feature_col="feature")
        assert isinstance(result, dict)
        assert "GA" in result
        assert "drusen" in result
        assert isinstance(result["GA"], AnnotationOCT)

    def test_not_a_dataframe_raises(self):
        with pytest.raises(TypeError, match="Expected a pandas DataFrame"):
            load_annotation_from_df("not a df")

    def test_empty_dataframe_raises(self):
        with pytest.raises(ValueError, match="empty"):
            load_annotation_from_df(pd.DataFrame())

    def test_duplicate_indices_raises(self, tmp_mask_dir):
        file_paths = [str(tmp_mask_dir / "scan1_0.png")] * 3
        df = pd.DataFrame(
            {
                "file_path": file_paths,
                "bscan_index": [0, 1, 1],  # duplicate
            }
        )
        with pytest.raises(ValueError, match="Duplicate"):
            load_annotation_from_df(df)

    def test_gaps_raises_by_default(self, tmp_mask_dir):
        file_paths = [str(tmp_mask_dir / f"scan1_{i}.png") for i in [0, 2, 4]]
        df = pd.DataFrame(
            {
                "file_path": file_paths,
                "bscan_index": [0, 2, 4],  # gaps at 1, 3
            }
        )
        with pytest.raises(ValueError, match="gaps"):
            load_annotation_from_df(df)

    def test_allow_gaps_permits_gaps(self, tmp_mask_dir):
        file_paths = [str(tmp_mask_dir / f"scan1_{i}.png") for i in [0, 2, 4]]
        df = pd.DataFrame(
            {
                "file_path": file_paths,
                "bscan_index": [0, 2, 4],
            }
        )
        ann = load_annotation_from_df(df, allow_gaps=True)
        assert isinstance(ann, AnnotationOCT)

    def test_allow_gaps_still_catches_duplicates(self, tmp_mask_dir):
        file_paths = [str(tmp_mask_dir / "scan1_0.png")] * 2
        df = pd.DataFrame(
            {
                "file_path": file_paths,
                "bscan_index": [0, 0],
            }
        )
        with pytest.raises(ValueError, match="Duplicate"):
            load_annotation_from_df(df, allow_gaps=True)

    def test_identity_col_multiple_values_raises(self, tmp_mask_dir):
        file_paths = [str(tmp_mask_dir / f"scan1_{i}.png") for i in range(4)]
        df = pd.DataFrame(
            {
                "file_path": file_paths,
                "bscan_index": [0, 1, 0, 1],
                "source_id": ["s1", "s1", "s2", "s2"],
            }
        )
        with pytest.raises(ValueError, match="distinct values"):
            load_annotation_from_df(df, identity_col="source_id")

    def test_validate_false_bypasses(self, tmp_mask_dir):
        file_paths = [str(tmp_mask_dir / "scan1_0.png")] * 3
        df = pd.DataFrame(
            {
                "file_path": file_paths,
                "bscan_index": [0, 0, 0],  # all duplicates
            }
        )
        # Should not raise
        ann = load_annotation_from_df(df, validate=False)
        assert isinstance(ann, AnnotationOCT)

    def test_per_feature_validation(self, tmp_mask_dir):
        file_paths = [str(tmp_mask_dir / "scan1_0.png")] * 4
        df = pd.DataFrame(
            {
                "file_path": file_paths,
                "bscan_index": [0, 0, 0, 1],  # duplicate in GA
                "feature": ["GA", "GA", "drusen", "drusen"],
            }
        )
        with pytest.raises(ValueError, match="feature=GA"):
            load_annotation_from_df(df, feature_col="feature")


class TestAnnotationFromArray:
    """Tests for building annotations from numpy arrays."""

    def test_build_from_array(self):
        data = np.random.randint(0, 255, (5, 64, 64), dtype=np.uint8)
        ann = _build_annotation_from_array(data)
        assert isinstance(ann, AnnotationOCT)
        assert ann.data.shape == (5, 64, 64)


class TestAnnotationFromFolder:
    """Tests for loading annotations from folder structures."""

    def test_load_from_folder(self, tmp_mask_dir_with_features):
        annotations = load_annotation_from_folder(
            str(tmp_mask_dir_with_features),
            folder_structure="{feature}/{source_id}_{bscan_index:\\d+}.png",
        )
        assert isinstance(annotations, dict)
        assert "GA" in annotations
        assert "drusen" in annotations
        assert isinstance(annotations["GA"], AnnotationOCT)

    def test_nonexistent_folder_raises(self, tmp_path):
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        with pytest.raises(ValueError, match="No annotation files found"):
            load_annotation_from_folder(
                str(empty_dir),
                folder_structure="{feature}/{source_id}_{bscan_index:\\d+}.png",
            )
