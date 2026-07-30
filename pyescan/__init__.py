"""
PyeScan - A library for working with retinal scans in Python.

Provides a common interface for loading, viewing, and analysing
retinal imaging data (OCT, FAF, IR, Color Fundus).
"""

from .CELoader import (
    load_record_from_CE,
    load_record_from_df,
    load_records_from_df,
)
from .annotation_loader import (
    load_annotation_from_df,
    load_annotation_from_folder,
)
from .core.scan import BaseScan, SingleImageScan
from .core.scan_oct import OCTScan, BScan, BScanArray
from .core.scan_enface import EnfaceScan, FAFScan, IRScan, ColorFundusScan
from .core.annotation import Annotation, AnnotationEnface, AnnotationOCT
from .core.image import LazyImage, ImageVolume
from .core.metadata import MetadataRecord, MetadataView

__all__ = [
    # Loading
    "load_record_from_CE",
    "load_record_from_df",
    "load_records_from_df",
    "load_annotation_from_df",
    "load_annotation_from_folder",
    # Scan classes
    "BaseScan",
    "SingleImageScan",
    "OCTScan",
    "BScan",
    "BScanArray",
    "EnfaceScan",
    "FAFScan",
    "IRScan",
    "ColorFundusScan",
    # Annotations
    "Annotation",
    "AnnotationEnface",
    "AnnotationOCT",
    # Images
    "LazyImage",
    "ImageVolume",
    # Metadata
    "MetadataRecord",
    "MetadataView",
]
