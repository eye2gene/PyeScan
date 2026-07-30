"""
Annotation save/export utilities.

Handles serialisation of annotations to various formats.
"""
import os
from typing import Dict, Optional

import numpy as np
from PIL import Image as PILImage

from .core.annotation import Annotation


def save_annotation(annotation: Annotation,
                    output_dir: str,
                    feature_name: Optional[str] = None,
                    source_id: Optional[str] = None,
                    format: str = "crystaleye",
                    exist_ok: bool = True) -> str:
    """
    Save an annotation to disk.

    Parameters
    ----------
    annotation : Annotation
        The annotation to save.
    output_dir : str
        Root output directory.
    feature_name : str, optional
        Feature name for the directory structure. Falls back to
        annotation.feature_name.
    source_id : str, optional
        Source ID for file naming. Falls back to annotation.source_id.
    format : str, default "crystaleye"
        Output format. Currently supported:
        - "crystaleye": saves as [output_dir]/[feature]/[source_id]_[index].png
    exist_ok : bool, default True
        Whether to overwrite existing files.

    Returns
    -------
    str
        Path to the output directory where files were saved.

    Raises
    ------
    ValueError
        If feature_name or source_id cannot be determined.
    """
    feat = feature_name or annotation.feature_name
    src = source_id or annotation.source_id

    if feat is None:
        raise ValueError(
            "feature_name must be provided either as argument or set on the annotation."
        )
    if src is None:
        raise ValueError(
            "source_id must be provided either as argument or set on the annotation."
        )

    if format == "crystaleye":
        return _save_crystaleye(annotation, output_dir, feat, src, exist_ok)
    else:
        raise ValueError(f"Unknown save format: {format!r}. Supported: 'crystaleye'")


def _save_crystaleye(annotation: Annotation, output_dir: str,
                     feature_name: str, source_id: str, exist_ok: bool) -> str:
    """
    Save in CrystalEye convention: [output_dir]/[feature]/[source_id]_[index].png
    """
    feat_dir = os.path.join(output_dir, feature_name)
    os.makedirs(feat_dir, exist_ok=True)

    for i, ann_slice in enumerate(annotation.slices):
        data = ann_slice.data
        if data is None:
            continue

        filename = f"{source_id}_{i}.png"
        filepath = os.path.join(feat_dir, filename)

        if not exist_ok and os.path.exists(filepath):
            raise FileExistsError(f"File already exists: {filepath}")

        img = PILImage.fromarray(data.astype(np.uint8))
        img.save(filepath)

    return feat_dir


def save_annotations(annotations: Dict[str, Annotation],
                     output_dir: str,
                     source_id: Optional[str] = None,
                     format: str = "crystaleye",
                     exist_ok: bool = True) -> str:
    """
    Save a dict of annotations (as returned by scan.annotations).

    Parameters
    ----------
    annotations : dict[str, Annotation]
        Mapping of feature_name -> Annotation.
    output_dir : str
        Root output directory.
    source_id : str, optional
        Source ID for file naming. Falls back to each annotation's source_id.
    format : str
        Output format.
    exist_ok : bool
        Whether to overwrite existing files.

    Returns
    -------
    str
        The output directory path.
    """
    for feat_name, annotation in annotations.items():
        save_annotation(
            annotation,
            output_dir=output_dir,
            feature_name=feat_name,
            source_id=source_id,
            format=format,
            exist_ok=exist_ok,
        )
    return output_dir
