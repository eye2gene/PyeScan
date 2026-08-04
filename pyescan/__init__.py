"""
PyeScan - A library for working with retinal scans in Python.

Provides a common interface for loading, viewing, and analysing
retinal imaging data (OCT, FAF, IR, Color Fundus).
"""

from .annotation_io import save_annotation, save_annotations
from .annotation_loader import (
    load_annotation_from_df,
    load_annotation_from_folder,
)
from .CELoader import (
    load_record_from_CE,
    load_record_from_df,
    load_records_from_df,
)
from .core.annotation import Annotation, AnnotationEnface, AnnotationOCT, AnnotationSlice, AnnotationVolume
from .core.image import ImageVolume, LazyImage
from .core.metadata import MetadataRecord, MetadataView
from .core.scan import BaseScan, SingleImageScan
from .core.scan_enface import ColorFundusScan, EnfaceScan, FAFScan, IRScan
from .core.scan_oct import BScan, BScanArray, OCTScan

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
    "AnnotationSlice",
    "AnnotationVolume",
    "AnnotationEnface",
    "AnnotationOCT",
    # Annotation IO
    "save_annotation",
    "save_annotations",
    # Images
    "LazyImage",
    "ImageVolume",
    # Metadata
    "MetadataRecord",
    "MetadataView",
]
