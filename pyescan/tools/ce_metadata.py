"""
CrystalEye metadata parsing utilities.

Decouples metadata JSON parsing from file system scraping.
Provides:
- parse_ce_metadata_json(): parse a single metadata.json into records
- scrape_ce_export(): find + parse + validate a full export
"""
import json
import logging
import os
from typing import Dict, List, Optional, Tuple

import pandas as pd

from .dataset_utils import summarise_dataset

logger = logging.getLogger(__name__)


def parse_ce_metadata_json(metadata_path: str,
                           identifier_dict: Optional[Dict[str, str]] = None,
                           skip_image_level: bool = False) -> List[Dict]:
    """
    Parse a single CrystalEye metadata.json into a list of record dicts.

    This is the core parsing function exposed as a public helper.
    Can be used standalone on any metadata.json file.

    Parameters
    ----------
    metadata_path : str
        Path to the metadata.json file.
    identifier_dict : dict, optional
        Additional identifier fields to include in each record
        (e.g. {"pat": "00046792.pat", "sdb": "00522105.sdb"}).
    skip_image_level : bool, default False
        If True, only produce scan-level records (one per source_id).
        If False (default), produce image-level records (one per B-scan).

    Returns
    -------
    list[dict]
        List of record dictionaries, one per image (or per scan if
        skip_image_level=True).

    Raises
    ------
    FileNotFoundError
        If the metadata file doesn't exist.
    ValueError
        If the JSON is invalid or missing required keys.
    """
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    with open(metadata_path, 'r') as f:
        try:
            metadata = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Invalid JSON in {metadata_path}: {e}"
            ) from e

    if "images" not in metadata:
        raise ValueError(
            f"Metadata file {metadata_path} is missing 'images' key. "
            f"Keys found: {list(metadata.keys())}"
        )

    if identifier_dict is None:
        # Infer from file path
        parts = metadata_path.replace("\\", "/").split("/")
        # Try to extract pat/sdb from path
        identifier_dict = {}
        for part in parts:
            if part.endswith(".pat"):
                identifier_dict["pat"] = part
            elif part.endswith(".sdb") or part.endswith(".sda"):
                identifier_dict["sdb"] = part

    return _process_metadata(metadata, identifier_dict, skip_image_level,
                             metadata_dir=os.path.dirname(os.path.abspath(metadata_path)))


def _flatten_dict(dict_in, name=""):
    """Flatten a nested dict by concatenating key names."""
    dict_out = {}
    if isinstance(dict_in, dict):
        for k, v in dict_in.items():
            dict_out.update(_flatten_dict(v, name=f"{name}_{k}" if name else k))
    else:
        dict_out[name] = dict_in
    return dict_out


def _process_metadata(metadata: dict, identifier_dict: dict,
                      skip_image_level: bool, metadata_dir: str = "") -> List[Dict]:
    """Core metadata processing logic."""
    scan_records = []

    scan_date = metadata.get('exam', {}).get('scan_datetime')
    series_info = _flatten_dict(metadata.get('series', {}))

    scans = metadata['images']['images']

    for i, scan in enumerate(scans):
        scan_data = {}
        scan_data.update(identifier_dict)

        uid_path = "/".join(identifier_dict.values())
        scan_data['scan_uid'] = uid_path + "/" + scan['source_id']

        for attr in ['source_id', 'group', 'modality', 'field_of_view']:
            scan_data[attr] = scan.get(attr)
        scan_data['scan_number'] = i
        scan_data['date'] = scan_date
        scan_data.update(series_info)

        if scan.get("crystal_eye_laterality"):
            scan_data["group_laterality"] = scan["crystal_eye_laterality"]

        scan_data['series_source_id'] = metadata.get('series', {}).get('source_id')
        scan_data['source_id'] = scan['source_id']
        scan_data['number_of_images'] = len(scan.get('contents', []))

        scan_data['scan_width_px'] = scan.get('size', {}).get('width')
        scan_data['scan_height_px'] = scan.get('size', {}).get('height')
        for attr in ['dimensions_mm', 'resolutions_mm']:
            if attr in scan:
                scan_data.update(_flatten_dict(scan[attr], attr))

        # Store the metadata directory for file path resolution
        scan_data['_metadata_dir'] = metadata_dir

        if skip_image_level:
            scan_records.append(scan_data)
        else:
            images = scan.get('contents', [])
            for j, image in enumerate(images):
                image_data = {}
                image_data.update(scan_data)
                image_data['bscan_index'] = str(j)
                image_data['image_capture_datetime'] = image.get('capture_datetime')
                image_data['image_quality_heidelberg'] = image.get('quality')

                if image.get('photo_locations'):
                    locations = image['photo_locations'][0]
                    image_data.update(_flatten_dict(locations, 'bscan_location'))

                # Construct expected file path
                source_id = scan['source_id']
                filename = f"{source_id}_{j}.png"
                image_data['file_path'] = os.path.join(metadata_dir, filename)

                scan_records.append(image_data)

    return scan_records


def scrape_ce_export(export_location: str,
                     file_structure: str = "{pat}/{sdb}/metadata.json",
                     image_structure: str = "{pat}/{sdb}/{source_id}_{bscan_index:\\d+}.png",
                     validate: bool = True,
                     skip_image_level: bool = False,
                     on_missing: str = "warn",
                     on_duplicate: str = "warn",
                     progress: bool = True) -> pd.DataFrame:
    """
    Scrape a CrystalEye export: find metadata files, parse them, validate.

    This is the decoupled replacement for get_ce_export_summary.
    Separates file discovery (glob-based) from metadata parsing.

    Parameters
    ----------
    export_location : str
        Root directory of the CE export.
    file_structure : str
        Glob/regex pattern for finding metadata.json files.
    image_structure : str
        Pattern for image files (used for validation).
    validate : bool, default True
        Whether to validate that expected files exist on disk
        and check for duplicates.
    skip_image_level : bool, default False
        If True, only produce scan-level records.
    on_missing : str, default "warn"
        Action when expected image files are missing:
        "warn" = log warning, "raise" = raise ValueError, "ignore" = skip silently.
    on_duplicate : str, default "warn"
        Action when duplicate scan_uid + bscan_index combinations found:
        "warn" = log warning, "raise" = raise ValueError, "ignore" = skip silently.
    progress : bool, default True
        Whether to show progress bar.

    Returns
    -------
    pd.DataFrame
        DataFrame with one row per image (or per scan if skip_image_level),
        including metadata fields and file paths.
    """
    export_location = os.path.abspath(export_location)

    # Step 1: Find metadata files using summarise_dataset (glob-based)
    df_meta_files = summarise_dataset(export_location, structure=file_structure, progress=progress)

    if df_meta_files.empty:
        raise FileNotFoundError(
            f"No metadata files found in '{export_location}' "
            f"matching pattern '{file_structure}'."
        )

    # Step 2: Parse each metadata file
    all_records = []
    meta_paths = df_meta_files['file_path'].tolist()

    if progress:
        try:
            import tqdm
            meta_paths = tqdm.tqdm(meta_paths, desc="Parsing metadata")
        except ImportError:
            pass

    for meta_path in meta_paths:
        try:
            # Infer identifiers from the matched row
            row = df_meta_files[df_meta_files['file_path'] == meta_path].iloc[0]
            identifier_dict = {
                k: v for k, v in row.items()
                if k not in ('file_path', 'file_path_relative')
            }
            records = parse_ce_metadata_json(
                meta_path,
                identifier_dict=identifier_dict,
                skip_image_level=skip_image_level,
            )
            all_records.extend(records)
        except (ValueError, FileNotFoundError) as e:
            logger.warning(f"Failed to parse {meta_path}: {e}")

    if not all_records:
        raise ValueError("No scan records could be parsed from the metadata files.")

    df = pd.DataFrame(all_records)

    # Remove internal column
    if '_metadata_dir' in df.columns:
        df = df.drop(columns=['_metadata_dir'])

    # Step 3: Validate
    if validate and not skip_image_level:
        _validate_export(df, on_missing=on_missing, on_duplicate=on_duplicate)

    return df


def _validate_export(df: pd.DataFrame, on_missing: str = "warn", on_duplicate: str = "warn"):
    """Validate parsed export for missing files and duplicates."""
    issues = []

    # Check for duplicate scan_uid + bscan_index
    if 'scan_uid' in df.columns and 'bscan_index' in df.columns:
        dup_mask = df.duplicated(subset=['scan_uid', 'bscan_index'], keep=False)
        if dup_mask.any():
            n_dups = dup_mask.sum()
            dup_examples = df[dup_mask][['scan_uid', 'bscan_index']].head(5).to_string()
            msg = (
                f"Found {n_dups} duplicate (scan_uid, bscan_index) entries:\n{dup_examples}"
            )
            if on_duplicate == "raise":
                raise ValueError(msg)
            elif on_duplicate == "warn":
                logger.warning(msg)
                issues.append(msg)

    # Check for missing image files
    if 'file_path' in df.columns:
        missing_mask = ~df['file_path'].apply(os.path.exists)
        n_missing = missing_mask.sum()
        if n_missing > 0:
            missing_examples = df[missing_mask]['file_path'].head(5).tolist()
            msg = (
                f"Found {n_missing} expected image files that don't exist on disk. "
                f"Examples: {missing_examples}"
            )
            if on_missing == "raise":
                raise ValueError(msg)
            elif on_missing == "warn":
                logger.warning(msg)
                issues.append(msg)

    return issues
