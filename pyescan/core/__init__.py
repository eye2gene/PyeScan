"""
PyeScan core module - scan classes, images, metadata, and annotations.
"""

from .annotation import (
    Annotation, AnnotationSlice, AnnotationVolume,
    AnnotationEnface, AnnotationOCT, MaskImage, MaskVolume, ModelInfo,
)
from .image import BaseImage, ImageVolume, LazyImage
from .metadata import MetadataParser, MetadataRecord, MetadataView
from .scan import BaseScan, SingleImageScan
from .scan_building import ScanBuildError, build_from_metadata
from .scan_enface import ColorFundusScan, EnfaceScan, FAFScan, IRScan
from .scan_oct import BScan, BScanArray, OCTScan

__all__ = [
    "Annotation",
    "AnnotationEnface",
    "AnnotationOCT",
    "AnnotationSlice",
    "AnnotationVolume",
    "BScan",
    "BScanArray",
    "BaseImage",
    "BaseScan",
    "ColorFundusScan",
    "EnfaceScan",
    "FAFScan",
    "IRScan",
    "ImageVolume",
    "LazyImage",
    "MaskImage",
    "MaskVolume",
    "MetadataParser",
    "MetadataRecord",
    "MetadataView",
    "ModelInfo",
    "OCTScan",
    "ScanBuildError",
    "SingleImageScan",
    "build_from_metadata",
]
