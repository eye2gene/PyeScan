"""
PyeScan core module - scan classes, images, metadata, and annotations.
"""

from .scan import BaseScan, SingleImageScan
from .scan_oct import OCTScan, BScan, BScanArray
from .scan_enface import EnfaceScan, FAFScan, IRScan, ColorFundusScan
from .annotation import Annotation, AnnotationEnface, AnnotationOCT, MaskImage, MaskVolume
from .image import BaseImage, LazyImage, ImageVolume
from .metadata import MetadataRecord, MetadataView, MetadataParser
from .scan_building import build_from_metadata

__all__ = [
    "BaseScan",
    "SingleImageScan",
    "OCTScan",
    "BScan",
    "BScanArray",
    "EnfaceScan",
    "FAFScan",
    "IRScan",
    "ColorFundusScan",
    "Annotation",
    "AnnotationEnface",
    "AnnotationOCT",
    "MaskImage",
    "MaskVolume",
    "BaseImage",
    "LazyImage",
    "ImageVolume",
    "MetadataRecord",
    "MetadataView",
    "MetadataParser",
    "build_from_metadata",
]
