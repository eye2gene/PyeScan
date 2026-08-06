import logging

import numpy as np
from PIL import Image as PILImage

from .core.annotation import Annotation, AnnotationSlice, AnnotationVolume
from .tools.dataset_utils import summarise_dataset

logger = logging.getLogger(__name__)


def _validate_annotation_dataframe(df, file_path_col='file_path', index_col='bscan_index',
                                   identity_col=None, feature_col=None,
                                   allow_gaps=False):
    """
    Validate an annotation DataFrame for completeness and uniqueness.

    Checks:
    - No duplicate bscan indices within each feature/identity group
    - B-scan indices form a complete 0..max sequence (unless allow_gaps=True)

    Parameters
    ----------
    df : pd.DataFrame
        Annotation DataFrame to validate.
    file_path_col : str
        Column containing file paths.
    index_col : str
        Column containing B-scan indices.
    identity_col : str, optional
        Column identifying each volume for clearer error messages.
    feature_col : str, optional
        Column identifying annotation features (checked per-feature).
    allow_gaps : bool, default False
        If True, only check for duplicates and skip the completeness check.
        Useful when not all masks are saved for every B-scan.

    Raises
    ------
    ValueError
        If duplicates are detected, with details of which identity/feature
        the problem occurred at.
    """
    errors = []

    if index_col not in df.columns:
        return

    # Determine grouping columns
    group_cols = []
    if feature_col and feature_col in df.columns:
        group_cols.append(feature_col)
    if identity_col and identity_col in df.columns:
        group_cols.append(identity_col)

    if group_cols:
        groups = df.groupby(group_cols)
    else:
        groups = [("all", df)]

    for group_id, df_group in groups:
        # Build a label for error messages
        if isinstance(group_id, tuple):
            label = ", ".join(f"{col}={val}" for col, val in zip(group_cols, group_id))
        elif group_cols:
            label = f"{group_cols[0]}={group_id}"
        else:
            label = "all annotations"

        bscan_indices = df_group[index_col].dropna()
        if len(bscan_indices) == 0:
            continue

        try:
            bscan_indices_int = bscan_indices.astype(int)
        except (ValueError, TypeError):
            errors.append(
                f"[{label}] B-scan index column '{index_col}' contains non-integer values."
            )
            continue

        # --- Uniqueness: duplicate indices ---
        duplicates = bscan_indices_int[bscan_indices_int.duplicated()]
        if len(duplicates) > 0:
            dup_values = sorted(duplicates.unique().tolist())
            errors.append(
                f"[{label}] Duplicate B-scan indices: {dup_values[:10]}"
                f"{'...' if len(dup_values) > 10 else ''}. "
                f"Each B-scan index must appear only once per annotation."
            )

        # --- Completeness: check for gaps ---
        if not allow_gaps:
            indices = sorted(bscan_indices_int.tolist())
            max_idx = max(indices)
            expected = list(range(max_idx + 1))
            missing = sorted(set(expected) - set(indices))
            if missing:
                shown = missing[:10]
                errors.append(
                    f"[{label}] Annotation has gaps in B-scan indices. "
                    f"Expected 0..{max_idx} but missing: {shown}"
                    f"{'...' if len(missing) > 10 else ''} "
                    f"({len(missing)} missing out of {max_idx + 1} expected)."
                )

    if errors:
        raise ValueError(
            f"Annotation integrity check failed with {len(errors)} issue(s):\n" +
            "\n".join(f"  • {e}" for e in errors)
        )


def _build_annotation_from_file_paths(file_paths):
    """Build an Annotation from a list of file paths (one per B-scan slice)."""
    slices = [AnnotationSlice(file_path=fp) for fp in file_paths]
    return Annotation(slices=AnnotationVolume(slices))


def _build_annotation_from_array(data):
    """Build an Annotation from a 3D numpy array (n_bscans x H x W)."""
    slices = [AnnotationSlice(raster=bscan_data.astype(np.uint8)) for bscan_data in data]
    return Annotation(slices=AnnotationVolume(slices))

def _build_annotation_from_dataframe_base(df, file_path_col='file_path', index_col='bscan_index'):
    df = df.copy()
    
    # Convert bscan_index to int (it's float after to_numeric)
    df[index_col] = df[index_col].astype(int)
    
    # Ensure the DataFrame is sorted by bscan_index
    df_sorted = df.sort_values(index_col)
    max_index = df_sorted[index_col].max()
    
    # Create an array of None values with length max_index + 1
    file_paths = [None] * (int(max_index) + 1)
    
    # Fill in the array with file paths where bscan_index matches
    for _, row in df_sorted.iterrows():
        file_paths[row[index_col]] = row[file_path_col]
    return _build_annotation_from_file_paths(file_paths)


def load_annotation_from_df(df, file_path_col='file_path', index_col='bscan_index',
                            feature_col=None, identity_col=None, validate=True,
                            allow_gaps=False):
    """
    Load annotations from a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing annotation file paths and B-scan indices.
    file_path_col : str, default 'file_path'
        Column containing mask file paths.
    index_col : str, default 'bscan_index'
        Column containing B-scan indices.
    feature_col : str, optional
        Column identifying different annotation features (e.g. 'GA', 'drusen').
        If provided, returns a dict of {feature_name: annotation}.
    identity_col : str, optional
        Column uniquely identifying each volume/scan. If provided and contains
        multiple unique values, raises an error to prevent accidental collisions
        from loading multiple volumes' annotations together.
    validate : bool, default True
        Whether to check for duplicate/missing B-scan indices. Set to False to bypass.
    allow_gaps : bool, default False
        If True, only validate against duplicates and skip the completeness
        check. Useful when not all B-scan masks are saved (common scenario).

    Returns
    -------
    Annotation or dict[str, Annotation]
        Single annotation or dict of annotations keyed by feature name.

    Raises
    ------
    ValueError
        If identity_col has multiple values, or if validation detects issues.
    """
    import pandas as pd
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected a pandas DataFrame, got {type(df).__name__}.")
    if df.empty:
        raise ValueError("The provided DataFrame is empty. Cannot load annotations.")

    # Check identity column for collisions
    if identity_col:
        if identity_col not in df.columns:
            raise ValueError(
                f"identity_col='{identity_col}' not found in DataFrame. "
                f"Available columns: {list(df.columns)}."
            )
        unique_ids = df[identity_col].unique()
        if len(unique_ids) > 1:
            raise ValueError(
                f"DataFrame contains {len(unique_ids)} distinct values in "
                f"identity column '{identity_col}': {list(unique_ids[:5])}"
                f"{'...' if len(unique_ids) > 5 else ''}. "
                f"load_annotation_from_df expects annotations for a single volume. "
                f"Filter the DataFrame to one identity, or use identity_col with "
                f"load_annotation_from_folder's group_by parameter."
            )

    # Run integrity checks
    if validate:
        _validate_annotation_dataframe(
            df,
            file_path_col=file_path_col,
            index_col=index_col,
            identity_col=identity_col,
            feature_col=feature_col,
            allow_gaps=allow_gaps,
        )

    if feature_col:
        annotations_dict = {}
        for feature, df_feat in df.groupby(feature_col):
            ann = _build_annotation_from_dataframe_base(df_feat, file_path_col, index_col)
            ann.feature_name = feature
            # Set source_id from identity_col or infer from data
            if identity_col and identity_col in df_feat.columns:
                ann.source_id = str(df_feat[identity_col].iloc[0])
            annotations_dict[feature] = ann
        return annotations_dict
    else:
        ann = _build_annotation_from_dataframe_base(df, file_path_col, index_col)
        if identity_col and identity_col in df.columns:
            ann.source_id = str(df[identity_col].iloc[0])
        return ann


def load_annotation_from_folder(annotations_folder, folder_structure=r"{feature}/{source_id}_{bscan_index:\d+}.png",
                                group_by=None, identity_col=None, validate=True,
                                allow_gaps=False):
    """
    Load annotations from a folder structure.

    Parameters
    ----------
    annotations_folder : str
        Path to the root annotations folder.
    folder_structure : str
        Pattern describing the folder/file naming convention.
    group_by : str, optional
        Column to group annotations by (e.g. 'source_id') to produce
        a dict of {group: annotations}.
    identity_col : str, optional
        Column uniquely identifying each volume. Passed to load_annotation_from_df.
    validate : bool, default True
        Whether to check for duplicate/missing B-scan indices. Set to False to bypass.
    allow_gaps : bool, default False
        If True, only validate against duplicates and skip the completeness
        check. Useful when not all B-scan masks are saved.

    Returns
    -------
    dict
        Dictionary of annotations, keyed by feature or group.
    """
    df = summarise_dataset(annotations_folder, structure=folder_structure, progress=False)

    if df.empty:
        raise ValueError(
            f"No annotation files found in '{annotations_folder}' "
            f"matching structure '{folder_structure}'."
        )

    annotations = {}
    if group_by:
        if group_by not in df.columns:
            raise ValueError(
                f"group_by='{group_by}' not found in summarised DataFrame. "
                f"Available columns: {list(df.columns)}. "
                f"Check your folder_structure pattern produces this column."
            )
        for grp_id, df_grp in df.groupby(group_by):
            annotations[grp_id] = load_annotation_from_df(
                df_grp, feature_col='feature',
                identity_col=identity_col, validate=validate,
                allow_gaps=allow_gaps,
            )
    else:
        annotations = load_annotation_from_df(
            df, feature_col='feature',
            identity_col=identity_col, validate=validate,
            allow_gaps=allow_gaps,
        )

    return annotations
