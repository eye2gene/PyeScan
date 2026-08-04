"""
Tests for core scan, image, and metadata classes.
"""
import os
import pytest
import numpy as np
from PIL import Image as PILImage

from pyescan.core.image import LazyImage, ImageVolume
from pyescan.core.scan import BaseScan, SingleImageScan
from pyescan.core.scan_oct import OCTScan, BScan, BScanArray
from pyescan.core.scan_enface import EnfaceScan, FAFScan, IRScan
from pyescan.core.annotation import (
    MaskImage, MaskVolume, AnnotationOCT, AnnotationEnface,
)
from pyescan.core.metadata import (
    MetadataRecord, MetadataView, MetadataParserJSON, MetadataParserCSV,
)
from pyescan.core.scan_building import ScanBuildError
from pyescan.core.utils import ArrayView, _stack_arrays_with_empties, _pad_array


class TestLazyImage:
    """Tests for LazyImage lazy-loading behaviour."""

    def test_load_from_file(self, tmp_path):
        img = PILImage.fromarray(np.zeros((32, 32), dtype=np.uint8))
        path = str(tmp_path / "test.png")
        img.save(path)

        lazy = LazyImage(file_path=path)
        assert lazy.loaded is False

        data = lazy.data
        assert lazy.loaded is True
        assert data.shape == (32, 32)

    def test_load_from_raw_image(self):
        raw = PILImage.fromarray(np.ones((16, 16), dtype=np.uint8) * 128)
        lazy = LazyImage(raw_image=raw)
        data = lazy.data
        assert data.shape == (16, 16)
        assert data[0, 0] == 128

    def test_unload(self, tmp_path):
        img = PILImage.fromarray(np.zeros((8, 8), dtype=np.uint8))
        path = str(tmp_path / "test.png")
        img.save(path)

        lazy = LazyImage(file_path=path)
        _ = lazy.data  # force load
        assert lazy.loaded is True

        lazy.unload()
        assert lazy.loaded is False

    def test_colour_mode_conversion(self, tmp_path):
        # Save an RGB image, load as L
        img = PILImage.fromarray(np.zeros((8, 8, 3), dtype=np.uint8))
        path = str(tmp_path / "rgb.png")
        img.save(path)

        lazy = LazyImage(file_path=path, mode='L')
        data = lazy.data
        assert data.ndim == 2


class TestImageVolume:
    """Tests for ImageVolume."""

    def test_from_file_paths(self, tmp_path):
        paths = []
        for i in range(3):
            img = PILImage.fromarray(np.full((8, 8), i * 50, dtype=np.uint8))
            path = str(tmp_path / f"img_{i}.png")
            img.save(path)
            paths.append(path)

        vol = ImageVolume(file_paths=paths)
        assert len(vol) == 3
        assert vol[0].data.shape == (8, 8)

    def test_data_stacks_correctly(self, tmp_path):
        paths = []
        for i in range(3):
            img = PILImage.fromarray(np.full((8, 8), i * 50, dtype=np.uint8))
            path = str(tmp_path / f"img_{i}.png")
            img.save(path)
            paths.append(path)

        vol = ImageVolume(file_paths=paths)
        data = vol.data
        assert data.shape == (3, 8, 8)


class TestMetadataView:
    """Tests for MetadataView attribute resolution."""

    def test_getattr_raises_on_missing(self):
        record = MetadataRecord({"test": 1})
        view = MetadataView(record, view_info={"scan_number": 0})
        # No parser configured
        with pytest.raises(AttributeError, match="no parser"):
            _ = view.nonexistent

    def test_getattr_raises_on_none_from_parser(self):
        """Parser returns None for unknown attributes."""
        import pandas as pd
        from pyescan.CELoader import CrystalEyeParserCSV

        df = pd.DataFrame({"source_id": ["s1"], "modality": ["OCT"], "group": ["g1"]})
        record = MetadataRecord(df)
        parser = CrystalEyeParserCSV()
        view = MetadataView(record, view_info={"scan_number": 0}, parser=parser)

        with pytest.raises(AttributeError, match="parser returned None"):
            _ = view.totally_fake_attribute

    def test_scan_number_from_view_info(self):
        record = MetadataRecord({})
        view = MetadataView(record, view_info={"scan_number": 3})
        assert view.scan_number == 3

    def test_image_number_from_view_info(self):
        record = MetadataRecord({})
        view = MetadataView(record, view_info={"scan_number": 0, "image_number": 7})
        assert view.image_number == 7
        assert view.bscan_index == 7


class TestScanBuilding:
    """Tests for scan_building with real example data."""

    def test_build_oct_from_json(self, oct_sdb_path):
        from pyescan.CELoader import load_record_from_CE
        scans = load_record_from_CE(oct_sdb_path)
        assert len(scans) == 1
        assert isinstance(scans[0], OCTScan)

    def test_build_faf_from_json(self, faf_sdb_path):
        from pyescan.CELoader import load_record_from_CE
        scans = load_record_from_CE(faf_sdb_path)
        assert len(scans) == 1
        assert isinstance(scans[0], FAFScan)

    def test_scan_build_error_on_bad_metadata(self):
        from pyescan.core.scan_building import build_from_metadata

        # Create a metadata view that will fail on get_groups
        record = MetadataRecord({})
        view = MetadataView(record)  # no parser
        with pytest.raises(ScanBuildError, match="failed to retrieve scan groups"):
            build_from_metadata(view)


class TestOCTScanMethods:
    """Tests for OCTScan preload/unload and indexing."""

    def test_preload_unload(self, oct_sdb_path):
        from pyescan.CELoader import load_record_from_CE
        scans = load_record_from_CE(oct_sdb_path)
        scan = scans[0]

        # Preload should not crash
        scan.preload()
        assert scan.bscans[0].image.loaded is True

        # Unload should release
        scan.unload()
        assert scan.bscans[0].image.loaded is False

    def test_indexing(self, oct_sdb_path):
        from pyescan.CELoader import load_record_from_CE
        scans = load_record_from_CE(oct_sdb_path)
        scan = scans[0]

        bscan = scan[0]
        assert isinstance(bscan, BScan)

    def test_len(self, oct_sdb_path):
        from pyescan.CELoader import load_record_from_CE
        scans = load_record_from_CE(oct_sdb_path)
        scan = scans[0]
        assert len(scan) == 25


class TestArrayView:
    """Tests for the ArrayView utility class."""

    def test_len(self):
        items = [LazyImage(raw_image=PILImage.fromarray(np.zeros((4, 4), dtype=np.uint8))) for _ in range(3)]
        vol = ImageVolume(images=items)
        assert len(vol) == 3

    def test_slice(self):
        items = [LazyImage(raw_image=PILImage.fromarray(np.full((4, 4), i, dtype=np.uint8))) for i in range(5)]
        vol = ImageVolume(images=items)
        sliced = vol[1:3]
        assert len(sliced) == 2

    def test_data_property(self):
        items = [LazyImage(raw_image=PILImage.fromarray(np.full((4, 4), i, dtype=np.uint8))) for i in range(3)]
        vol = ImageVolume(images=items)
        data = vol.data
        assert data.shape == (3, 4, 4)


class TestUtilFunctions:
    """Tests for utility functions."""

    def test_stack_with_empties(self):
        arrays = [np.ones((4, 4)), None, np.ones((4, 4)) * 2]
        result = _stack_arrays_with_empties(arrays)
        assert result.shape == (3, 4, 4)
        assert result[1].sum() == 0  # None replaced with zeros

    def test_pad_array_expand(self):
        arr = np.ones((3, 4, 4))
        result = _pad_array(arr, 5)
        assert result.shape == (5, 4, 4)
        assert result[3:].sum() == 0

    def test_pad_array_truncate(self):
        arr = np.ones((5, 4, 4))
        result = _pad_array(arr, 3)
        assert result.shape == (3, 4, 4)
