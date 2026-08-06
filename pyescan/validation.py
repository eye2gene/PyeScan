"""
DataFrame validation utilities for PyeScan.

Provides integrity checks for scan and annotation DataFrames,
including duplicate detection and completeness verification.
"""
import logging

logger = logging.getLogger(__name__)


def validate_scan_dataframe(df, identity_col=None, bscan_index_col='bscan_index',
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
