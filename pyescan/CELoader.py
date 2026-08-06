"""
CrystalEye/PrivateEye scan loading functions.

This module provides the public API for loading retinal scans from
CrystalEye JSON exports and pandas DataFrames.
"""
import json
import logging
import os

from .core.metadata import MetadataRecord
from .core.scan_building import build_from_metadata
from .parsers import CrystalEyeParser, CrystalEyeParserCSV
from .validation import validate_scan_dataframe

logger = logging.getLogger(__name__)


def load_record_from_json_CE(metadata_file_path, format=None):
    """
    Load scan records from a CrystalEye JSON metadata file.

    Parameters
    ----------
    metadata_file_path : str
        Path to the metadata.json file or the directory containing it.

    Returns
    -------
    list[BaseScan]
        List of scan objects built from the metadata.
    """
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
    """
    Load scan records from a CrystalEye export folder (containing metadata.json).

    Parameters
    ----------
    path_to_record_folder : str
        Path to the sdb folder or directly to a metadata.json file.

    Returns
    -------
    list[BaseScan]
        List of scan objects built from the metadata.
    """
    file_path = path_to_record_folder
    if not file_path.endswith(".json"):
        file_path = os.path.join(file_path, "metadata.json")

    return load_record_from_json_CE(file_path, format=format)


def load_records_from_CE(path_to_records_folder, folder_structure="{pat}/{sdb}/metadata.json"):
    """Load multiple records from a CrystalEye export folder structure."""
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
        modality_col_name = parser._col_map.get('modality', 'modality')
        n_images_col = parser._col_map.get('n_images', 'number_of_images')

        validate_scan_dataframe(
            df_scan,
            identity_col=identity_col,
            bscan_index_col=bscan_index_col,
            source_id_col=source_id_col,
            modality_col=modality_col_name,
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
            scans[(*identifier, i)] = scan
    return scans
