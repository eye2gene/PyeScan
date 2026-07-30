"""
Tests for loading scans from CrystalEye exports and DataFrames.
"""
import json
import os
import pytest
import pandas as pd
import numpy as np

from pyescan.CELoader import (
    load_record_from_CE,
    load_record_from_json_CE,
    load_record_from_df,
    load_records_from_df,
)
from pyescan.core.scan_oct import OCTScan
from pyescan.core.scan_enface import FAFScan, IRScan
from pyescan.core.scan_building import ScanBuildError


class TestLoadFromCE:
    """Tests for loading from CrystalEye JSON exports."""

    def test_load_oct_record(self, oct_sdb_path):
        scans = load_record_from_CE(oct_sdb_path)
        assert len(scans) == 1
        scan = scans[0]
        assert isinstance(scan, OCTScan)

    def test_oct_has_correct_bscan_count(self, oct_sdb_path):
        scans = load_record_from_CE(oct_sdb_path)
        scan = scans[0]
        assert len(scan.bscans) == 25

    def test_oct_has_enface(self, oct_sdb_path):
        scans = load_record_from_CE(oct_sdb_path)
        scan = scans[0]
        assert scan.enface is not None
        assert isinstance(scan.enface, IRScan)

    def test_oct_bscan_data_shape(self, oct_sdb_path):
        scans = load_record_from_CE(oct_sdb_path)
        scan = scans[0]
        # Preload first bscan and check shape
        bscan = scan.bscans[0]
        data = bscan.data
        assert data.ndim == 2
        assert data.shape == (496, 512)  # height x width from metadata

    def test_oct_enface_data_shape(self, oct_sdb_path):
        scans = load_record_from_CE(oct_sdb_path)
        scan = scans[0]
        data = scan.enface.data
        assert data.ndim == 2
        assert data.shape == (768, 768)

    def test_load_faf_record(self, faf_sdb_path):
        scans = load_record_from_CE(faf_sdb_path)
        assert len(scans) == 1
        scan = scans[0]
        assert isinstance(scan, FAFScan)

    def test_faf_data_loads(self, faf_sdb_path):
        scans = load_record_from_CE(faf_sdb_path)
        scan = scans[0]
        data = scan.data
        assert data is not None
        assert data.ndim == 2

    def test_nonexistent_path_raises(self):
        with pytest.raises(FileNotFoundError, match="not found"):
            load_record_from_CE("/nonexistent/path")

    def test_invalid_json_raises(self, tmp_path):
        bad_json = tmp_path / "metadata.json"
        bad_json.write_text("not valid json {{{")
        with pytest.raises(ValueError, match="Failed to parse"):
            load_record_from_json_CE(str(bad_json))

    def test_missing_images_key_raises(self, tmp_path):
        bad_json = tmp_path / "metadata.json"
        bad_json.write_text(json.dumps({"patient": "test"}))
        with pytest.raises(ValueError, match="missing the required 'images' key"):
            load_record_from_json_CE(str(bad_json))


class TestLoadFromDataFrame:
    """Tests for loading from pandas DataFrames."""

    def test_not_a_dataframe_raises(self):
        with pytest.raises(TypeError, match="Expected a pandas DataFrame"):
            load_record_from_df("not a df")

    def test_empty_dataframe_raises(self):
        with pytest.raises(ValueError, match="empty"):
            load_record_from_df(pd.DataFrame())

    def test_missing_structure_columns_raises(self):
        df = pd.DataFrame({"col_a": [1], "col_b": [2]})
        with pytest.raises(ValueError, match="missing columns required for scan structure"):
            load_record_from_df(df)

    def test_missing_image_location_raises(self):
        df = pd.DataFrame({
            "source_id": ["s1"], "modality": ["OCT"], "group": ["g1"],
            "number_of_images": [1], "bscan_index": [0],
        })
        with pytest.raises(ValueError, match="image location column"):
            load_record_from_df(df)

    def test_oct_missing_bscan_index_raises(self):
        df = pd.DataFrame({
            "source_id": ["s1"], "modality": ["OCT"], "group": ["g1"],
            "number_of_images": [1], "file_path": ["a.png"],
        })
        with pytest.raises(ValueError, match="bscan_index"):
            load_record_from_df(df)

    def test_oct_missing_number_of_images_raises(self):
        df = pd.DataFrame({
            "source_id": ["s1"], "modality": ["OCT"], "group": ["g1"],
            "bscan_index": [0], "file_path": ["a.png"],
        })
        with pytest.raises(ValueError, match="number_of_images"):
            load_record_from_df(df)

    def test_identity_col_multiple_values_raises(self):
        df = pd.DataFrame({
            "source_id": ["s1", "s2"], "modality": ["OCT", "OCT"],
            "group": ["g1", "g1"], "number_of_images": [1, 1],
            "bscan_index": [0, 0], "file_path": ["a.png", "b.png"],
            "patient": ["p1", "p2"],
        })
        with pytest.raises(ValueError, match="distinct values"):
            load_record_from_df(df, identity_col="patient")

    def test_identity_col_nonexistent_raises(self):
        df = pd.DataFrame({
            "source_id": ["s1"], "modality": ["OCT"], "group": ["g1"],
            "number_of_images": [1], "bscan_index": [0], "file_path": ["a.png"],
        })
        with pytest.raises(ValueError, match="not found"):
            load_record_from_df(df, identity_col="nonexistent")

    def test_duplicate_bscan_indices_raises(self):
        df = pd.DataFrame({
            "source_id": ["s1"] * 3, "modality": ["OCT"] * 3,
            "group": ["g1"] * 3, "number_of_images": [2] * 3,
            "bscan_index": [0, 1, 1], "file_path": ["a.png", "b.png", "c.png"],
        })
        with pytest.raises(ValueError, match="Duplicate"):
            load_record_from_df(df)

    def test_validate_false_bypasses_checks(self):
        df = pd.DataFrame({
            "source_id": ["s1"] * 3, "modality": ["OCT"] * 3,
            "group": ["g1"] * 3, "number_of_images": [2] * 3,
            "bscan_index": [0, 0, 0], "file_path": ["a.png", "b.png", "c.png"],
        })
        # Should not raise ValueError about integrity
        # (will fail later in build, but that's fine)
        with pytest.raises(Exception) as exc_info:
            load_record_from_df(df, validate=False)
        assert "integrity" not in str(exc_info.value).lower()

    def test_column_headings_mapping(self, tmp_path):
        # Create a fake image so loading doesn't fail
        img = PILImage.fromarray(np.zeros((10, 10), dtype=np.uint8))
        img_path = str(tmp_path / "test.png")
        img.save(img_path)

        df = pd.DataFrame({
            "source_id": ["s1"],
            "modality": ["AF - Blue"],
            "group": ["g1"],
            "number_of_images": [1],
            "bscan_index": [0],
            "my_custom_path": [img_path],
        })
        scans = load_record_from_df(
            df,
            column_headings={"image_location": "my_custom_path"},
            validate=False,
        )
        assert len(scans) == 1
        assert isinstance(scans[0], FAFScan)

    def test_load_records_from_df_missing_id_cols(self):
        df = pd.DataFrame({"source_id": ["s1"], "modality": ["OCT"]})
        with pytest.raises(ValueError, match="missing identifier columns"):
            load_records_from_df(df, identifier_columns=["pat", "sdb"])


# Need PIL for the column_headings test
from PIL import Image as PILImage
