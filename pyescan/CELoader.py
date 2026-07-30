import logging
import os

from pyescan.core.metadata import MetadataParserCSV, MetadataParserJSON, MetadataRecord
from pyescan.core.scan_building import build_from_metadata

logger = logging.getLogger(__name__)

"""

HEYEX example format (output of privateEye)

OCT:
- patient
- exam
- series
- images
  - Image 0
    - Source ID
    - modality
    - group
    - size
    - ...
    - contents
    - extras
    - contours
  - Image 1
    - Source ID
    - modality
    - group
    - size
    - ...
    - contents
      - 1
      - 2
      - ...
    - extras
    - contours
- debug
- parser version

Proposed Taxonomy:
0. (Patient)
1. Record - i.e. a single sdb/sda/e2e file
2. Scan group (usually OCT + enface, there may be just one of these per record, but sometimes multiple scan-groups are captured in a single record in which case they are given different group ids)
3. Scan (in the privateeye/crystaleye this is represented by each 'image' entry)
4. Image (usually 1 per scan, except for OCT scans whre there are multiple)


Scan / scan grp (volume) / scan image
Scan metadata
Mask (single image)
Mask volume
Fovea (prediction)
Registration projection
Classification output
Dataset
Model

"""


def _validate_dataframe_integrity(df, identity_col=None, bscan_index_col='bscan_index',
                                  source_id_col='source_id', modality_col='modality',
                                  n_images_col='number_of_images'):
    """
    Validate a scan DataFrame for completeness and uniqueness.

    Checks:
    - No duplicate B-scan indices within each source_id
    - B-scan indices form a complete sequence (0..n-1) for OCT modalities
    - Expected number of images matches actual count (if n_images column present)

    Parameters
    ----------
    df : pd.DataFrame
        The scan DataFrame to validate.
    identity_col : str, optional
        Column identifying each volume/record for clearer error messages.
        If None, source_id_col is used as the identity.
    bscan_index_col : str
        Column containing B-scan indices.
    source_id_col : str
        Column containing the source/scan identifier.
    modality_col : str
        Column containing the modality string.
    n_images_col : str
        Column containing the expected number of images.

    Raises
    ------
    ValueError
        If duplicates or missing B-scans are detected, with details of which
        identity/source_id the problem occurred at.
    """
    errors = []

    id_col = identity_col or source_id_col

    # Check columns exist before validating
    if source_id_col not in df.columns:
        return  # Can't validate without source_id
    if bscan_index_col not in df.columns:
        return  # No bscan index to validate

    for group_id, df_group in df.groupby(source_id_col):
        identity_label = group_id
        if identity_col and identity_col in df.columns and identity_col != source_id_col:
            identity_values = df_group[identity_col].unique()
            identity_label = f"{id_col}={identity_values[0]}, {source_id_col}={group_id}"

        # Skip non-OCT modalities for completeness checks (they have 1 image)
        is_oct = False
        if modality_col in df.columns:
            modalities = df_group[modality_col].unique()
            is_oct = any('OCT' in str(m) for m in modalities)

        # --- Uniqueness check: duplicate bscan indices ---
        if bscan_index_col in df_group.columns:
            bscan_indices = df_group[bscan_index_col].dropna()
            if len(bscan_indices) > 0:
                duplicates = bscan_indices[bscan_indices.duplicated()]
                if len(duplicates) > 0:
                    dup_values = sorted(duplicates.unique().tolist())
                    errors.append(
                        f"[{identity_label}] Duplicate B-scan indices found: {dup_values}. "
                        f"Each B-scan index must be unique within a scan."
                    )

        # --- Completeness check (OCT only): contiguous 0..n-1 ---
        if is_oct and bscan_index_col in df_group.columns:
            bscan_indices = df_group[bscan_index_col].dropna()
            if len(bscan_indices) > 0:
                try:
                    indices = sorted(bscan_indices.astype(int).tolist())
                except (ValueError, TypeError):
                    continue

                expected = list(range(len(indices)))
                # Check if expected count matches n_images metadata
                if n_images_col in df_group.columns:
                    n_expected_values = df_group[n_images_col].dropna().unique()
                    if len(n_expected_values) > 0:
                        try:
                            n_expected = int(n_expected_values[0])
                            if len(indices) != n_expected:
                                errors.append(
                                    f"[{identity_label}] Expected {n_expected} B-scans "
                                    f"(from '{n_images_col}' column) but found {len(indices)}."
                                )
                        except (ValueError, TypeError):
                            pass

                # Check for gaps in the sequence
                if indices != expected:
                    missing = sorted(set(expected) - set(indices))
                    extra = sorted(set(indices) - set(range(max(indices) + 1)))
                    msg_parts = []
                    if missing:
                        # Only show first 10 to keep messages readable
                        shown = missing[:10]
                        msg_parts.append(f"missing indices {shown}{'...' if len(missing) > 10 else ''}")
                    if extra:
                        shown = extra[:10]
                        msg_parts.append(f"unexpected indices {shown}{'...' if len(extra) > 10 else ''}")
                    if msg_parts:
                        errors.append(
                            f"[{identity_label}] B-scan indices are not a complete 0..{len(indices)-1} sequence: "
                            f"{', '.join(msg_parts)}."
                        )

    if errors:
        raise ValueError(
            f"Data integrity check failed with {len(errors)} issue(s):\n" +
            "\n".join(f"  • {e}" for e in errors)
        )


class CrystalEyeParser(MetadataParserJSON):
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
        self._overrides = { "n_scans": self.n_scans }
        
    def _get_records_subset(self, metadata_record, view_info):
        df = metadata_record.raw
        if "scan_number" in view_info:
            scan_number = metadata_record.raw.source_id.unique()[view_info["scan_number"]]  # noqa: F841
            df = df.query("source_id == @scan_number")
        if "image_number" in view_info:
            image_number = view_info["image_number"]  # noqa: F841
            df = df.query("bscan_index == @image_number")
        return df
    
    def n_scans(self, metadata_record, view_info):
        #records_subset = self._get_records_subset(metadata_record, view_info)
        return metadata_record.raw.source_id.nunique()
    
def load_record_from_json_CE(metadata_file_path, format=None):
    
    import json
    file_path = metadata_file_path
    if not file_path.endswith(".json"):
        file_path = os.path.join(file_path, "metadata.json")

    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Metadata file not found: '{file_path}'. "
            f"Expected a JSON metadata file at this location."
        )

    try:
        with open(file_path) as f:
            json_data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Failed to parse metadata JSON at '{file_path}': {e}. "
            f"The file may be corrupted or not valid JSON."
        ) from e

    # Basic schema validation
    if not isinstance(json_data, dict):
        raise ValueError(
            f"Metadata file '{file_path}' does not contain a JSON object at the top level."
        )
    if "images" not in json_data:
        raise ValueError(
            f"Metadata file '{file_path}' is missing the required 'images' key. "
            f"Top-level keys found: {list(json_data.keys())}. "
            f"Is this a valid CrystalEye/PrivateEye export?"
        )

    record = MetadataRecord(json_data, file_path)
    metadata = record.get_view(parser=CrystalEyeParser())
    return build_from_metadata(metadata)

def load_record_from_CE(path_to_record_folder, format=None):

    file_path = path_to_record_folder
    if not file_path.endswith(".json"):
        file_path = os.path.join(file_path, "metadata.json")

    return load_record_from_json_CE(file_path, format=format)

def load_records_from_CE(path_to_records_folder, folder_structure="{pat}/{sdb}/metadata.json"):
    raise NotImplementedError()
    

def load_record_from_df(df_scan, column_headings=None, identity_col=None, validate=True):
    """
    Load scan records from a single-record DataFrame.

    Parameters
    ----------
    df_scan : pd.DataFrame
        DataFrame containing scan metadata for a single record (e.g. one sdb).
    column_headings : dict, optional
        Mapping of internal column names to actual DataFrame column names.
    identity_col : str, optional
        Column that uniquely identifies each volume/scan group. Used in error
        messages and to prevent loading DataFrames that contain multiple records
        without explicit grouping. If provided and the column contains more than
        one unique value, a ValueError is raised.
    validate : bool, default True
        Whether to run completeness and uniqueness checks. Set to False to bypass.

    Returns
    -------
    list[BaseScan]
        List of scan objects built from the metadata.

    Raises
    ------
    TypeError
        If df_scan is not a DataFrame.
    ValueError
        If the DataFrame is empty, missing required columns, contains multiple
        identities without grouping, or fails integrity checks.
    """
    if column_headings is None:
        column_headings = {}

    import pandas as pd
    if not isinstance(df_scan, pd.DataFrame):
        raise TypeError(
            f"Expected a pandas DataFrame, got {type(df_scan).__name__}."
        )
    if df_scan.empty:
        raise ValueError(
            "The provided DataFrame is empty. Cannot load scan records from an empty DataFrame."
        )

    # Validate required columns are present
    # These are the minimum columns needed for scan building:
    #   source_id - identifies each scan, used for grouping
    #   group     - groups scans (e.g. OCT + enface together)
    #   modality  - determines which builder to use
    # For OCT scans, also needed (checked below):
    #   n_images (number_of_images) - how many bscans
    #   image_location (file_path)  - where images are
    #   bscan_index                 - identifies each bscan row (used in subsetting)
    parser = CrystalEyeParserCSV(column_headings=column_headings)

    required_for_structure = ('source_id', 'modality', 'group')
    missing_structure = []
    for internal_name in required_for_structure:
        df_col_name = parser._col_map.get(internal_name)
        if df_col_name and df_col_name not in df_scan.columns:
            missing_structure.append(f"'{df_col_name}' (for {internal_name})")
    if missing_structure:
        raise ValueError(
            f"DataFrame is missing columns required for scan structure: "
            f"{', '.join(missing_structure)}. "
            f"Available columns: {list(df_scan.columns)}. "
            f"Use the column_headings parameter to map your column names."
        )

    # Check columns needed for image loading
    image_loc_col = parser._col_map.get('image_location')
    if image_loc_col and image_loc_col not in df_scan.columns:
        raise ValueError(
            f"DataFrame is missing the image location column '{image_loc_col}' "
            f"(mapped from 'image_location'). "
            f"Available columns: {list(df_scan.columns)}. "
            f"Use column_headings={{'image_location': 'your_column'}} to specify it."
        )

    # Check OCT-specific columns if any OCT modalities are present
    modality_col = parser._col_map.get('modality', 'modality')
    if modality_col in df_scan.columns:
        has_oct = df_scan[modality_col].str.contains('OCT', na=False).any()
        if has_oct:
            oct_required = {
                'n_images': parser._col_map.get('n_images', 'number_of_images'),
            }
            missing_oct = []
            for internal_name, df_col_name in oct_required.items():
                if df_col_name not in df_scan.columns:
                    missing_oct.append(f"'{df_col_name}' (for {internal_name})")

            # bscan_index is used directly by _get_records_subset, not via _col_map
            bscan_idx_col = column_headings.get('bscan_index', 'bscan_index')
            if bscan_idx_col not in df_scan.columns:
                missing_oct.append(f"'{bscan_idx_col}' (for bscan_index)")

            if missing_oct:
                raise ValueError(
                    f"DataFrame contains OCT scans but is missing required columns: "
                    f"{', '.join(missing_oct)}. "
                    f"OCT scans require B-scan indices and image counts. "
                    f"Available columns: {list(df_scan.columns)}."
                )

    # If identity_col is provided, check the DataFrame only contains one identity
    if identity_col:
        if identity_col not in df_scan.columns:
            raise ValueError(
                f"identity_col='{identity_col}' not found in DataFrame. "
                f"Available columns: {list(df_scan.columns)}."
            )
        unique_ids = df_scan[identity_col].unique()
        if len(unique_ids) > 1:
            raise ValueError(
                f"DataFrame contains {len(unique_ids)} distinct values in "
                f"identity column '{identity_col}': {list(unique_ids[:5])}{'...' if len(unique_ids) > 5 else ''}. "
                f"load_record_from_df expects a single record. "
                f"Use load_records_from_df with identity_col to load multiple records, "
                f"or filter the DataFrame to a single identity first."
            )

    # Run integrity checks
    if validate:
        source_id_col = parser._col_map.get('source_id', 'source_id')
        bscan_index_col = column_headings.get('bscan_index', 'bscan_index')
        modality_col = parser._col_map.get('modality', 'modality')
        n_images_col = parser._col_map.get('n_images', 'number_of_images')

        _validate_dataframe_integrity(
            df_scan,
            identity_col=identity_col,
            bscan_index_col=bscan_index_col,
            source_id_col=source_id_col,
            modality_col=modality_col,
            n_images_col=n_images_col,
        )

    record = MetadataRecord(df_scan)
    metadata = record.get_view(parser=parser)
            
    return build_from_metadata(metadata)

def load_records_from_df(df, column_headings=None, identifier_columns=None,
                         identity_col=None, validate=True):
    """
    Load scan records from a DataFrame containing multiple records.

    Groups the DataFrame by identifier_columns and loads each group as a
    separate record.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing scan metadata for multiple records.
    column_headings : dict, optional
        Mapping of internal column names to actual DataFrame column names.
    identifier_columns : list[str], optional
        Columns to group by to split records. Defaults to ['pat', 'sdb'].
    identity_col : str, optional
        Column that uniquely identifies each volume within a record group.
        Passed through to load_record_from_df for per-record validation.
        This prevents collisions when a single record group contains multiple
        volumes (e.g. multiple source_ids).
    validate : bool, default True
        Whether to run completeness and uniqueness checks. Set to False to bypass.

    Returns
    -------
    dict
        Dictionary mapping (identifier..., index) tuples to scan objects.
    """
    if column_headings is None:
        column_headings = {}
    if identifier_columns is None:
        identifier_columns = ['pat', 'sdb']

    import pandas as pd
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected a pandas DataFrame, got {type(df).__name__}.")
    if df.empty:
        raise ValueError("The provided DataFrame is empty.")

    missing_id_cols = [col for col in identifier_columns if col not in df.columns]
    if missing_id_cols:
        raise ValueError(
            f"DataFrame is missing identifier columns: {missing_id_cols}. "
            f"Available columns: {list(df.columns)}."
        )

    from tqdm import tqdm
    scans = {}
    for identifier, df_scan in tqdm(df.groupby(identifier_columns)):
        scan_set = load_record_from_df(
            df_scan,
            column_headings=column_headings,
            identity_col=identity_col,
            validate=validate,
        )
        for i, scan in enumerate(scan_set):
            scans[(*identifier,i)] = scan
    return scans