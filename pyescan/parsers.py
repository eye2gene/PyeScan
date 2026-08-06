"""
CrystalEye/PrivateEye metadata parser implementations.

Provides MetadataParser subclasses that know how to navigate
CrystalEye JSON exports and CSV-based DataFrame representations.

HEYEX example format (output of PrivateEye/CrystalEye)
-------------------------------------------------------
OCT:
- patient
- exam
- series
- images
  - Image 0
    - Source ID, modality, group, size, ...
    - contents (per-bscan entries)
    - extras
    - contours
  - Image 1
    - ...

Proposed Taxonomy:
0. (Patient)
1. Record - i.e. a single sdb/sda/e2e file
2. Scan group (usually OCT + enface; sometimes multiple per record)
3. Scan (each 'image' entry in CrystalEye)
4. Image (usually 1 per scan, except OCT which has multiple B-scans)
"""
import os

from .core.metadata import MetadataParserCSV, MetadataParserJSON


class CrystalEyeParser(MetadataParserJSON):
    """Parser for CrystalEye JSON metadata exports."""

    _scan_level = ["images", "images", "{scan_number}"]
    _image_level = _scan_level + ["contents", "{image_number}"]

    _path_map = {
        "group": _scan_level + ["group"],
        "source_id": _scan_level + ["source_id"],
        "modality": _scan_level + ["modality"],
        "manufacturer": _scan_level + ["manufacturer"],

        "bscan_start_x": _image_level + ["photo_locations", 0, "start", "x"],
        "bscan_start_y": _image_level + ["photo_locations", 0, "start", "y"],
        "bscan_end_x": _image_level + ["photo_locations", 0, "end", "x"],
        "bscan_end_y": _image_level + ["photo_locations", 0, "end", "y"],
    }

    def __init__(self):
        self._overrides = {
            "n_scans": self.n_scans,
            "n_images": self.n_images,
            "image_location": self.image_location
        }

    def n_scans(self, metadata_record, view_info):
        path = self._map_path(self._scan_level[:-1], view_info)
        return len(self._get_by_path(metadata_record, path))

    def n_images(self, metadata_record, view_info):
        path = self._map_path(self._image_level[:-1], view_info)
        return len(self._get_by_path(metadata_record, path))

    def image_location(self, metadata_record, view_info):
        modality = self.get_value('modality', metadata_record, view_info)
        bscan_index = view_info['image_number'] if 'OCT' in modality else 0
        source_id = self.get_value('source_id', metadata_record, view_info)

        file_name = f"{source_id}_{bscan_index}.png"
        return os.path.join(metadata_record.location, file_name)


class CrystalEyeParserCSV(MetadataParserCSV):
    """Parser for CrystalEye data loaded into a pandas DataFrame."""

    _base_col_map = {
        "n_images": "number_of_images",
        "group": "group",
        "source_id": "source_id",
        "modality": "modality",
        "image_location": "file_path",

        "bscan_start_x": "bscan_location_start_x",
        "bscan_start_y": "bscan_location_start_y",
        "bscan_end_x": "bscan_location_end_x",
        "bscan_end_y": "bscan_location_end_y",
    }

    def __init__(self, column_headings=None):
        self._col_map = self._base_col_map.copy()
        if column_headings:
            self._col_map.update(column_headings)
        self._overrides = {"n_scans": self.n_scans}

    def _get_records_subset(self, metadata_record, view_info):
        df = metadata_record.raw
        if "scan_number" in view_info:
            scan_number = metadata_record.raw.source_id.unique()[view_info["scan_number"]]  # noqa: F841
            df = df.query("source_id == @scan_number")
        if "image_number" in view_info:
            image_number = view_info["image_number"]
            # Coerce to match column dtype (DataFrame may have strings from CSV)
            if "bscan_index" in df.columns and len(df) > 0:
                col_dtype = df["bscan_index"].dtype
                if col_dtype == object:  # string column
                    image_number = str(image_number)  # noqa: F841
                else:
                    image_number = col_dtype.type(image_number)  # noqa: F841
            df = df.query("bscan_index == @image_number")
        return df

    def n_scans(self, metadata_record, view_info):
        return metadata_record.raw.source_id.nunique()
