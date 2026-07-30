"""
Annotation classes for PyeScan.

Design:
- AnnotationSlice: one frame of annotation data (raster, contours, points)
- AnnotationVolume: ArrayView of AnnotationSlices (indexable, sliceable)
- Annotation: feature-level metadata + slices
"""
from typing import List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from PIL import Image as PILImage

from .image import LazyImage
from .utils import ArrayView


class ModelInfo:
    """Information about the model that produced an annotation."""
    def __init__(self, name: Optional[str] = None, version: Optional[str] = None):
        self.name = name
        self.version = version

    def __repr__(self):
        return f"ModelInfo(name={self.name!r}, version={self.version!r})"


class AnnotationSlice:
    """
    A single frame of annotation data.

    Can hold any combination of:
    - raster: HxW numpy array (binary mask or probability map)
    - contours: list of Nx2 polyline arrays (closed or open)
    - points: Nx2 array of landmark coordinates

    Lazy-loads raster from file if constructed with a file path.
    """
    def __init__(self,
                 raster: Optional[NDArray] = None,
                 contours: Optional[List[NDArray]] = None,
                 points: Optional[NDArray] = None,
                 file_path: Optional[str] = None,
                 threshold: float = 0.5):
        self._raster = raster
        self._contours = contours
        self._points = points
        self._file_path = file_path
        self._threshold = threshold
        self._loaded = raster is not None

    @property
    def file_path(self) -> Optional[str]:
        return self._file_path

    @property
    def has_raster(self) -> bool:
        return self._raster is not None or self._file_path is not None

    @property
    def has_contours(self) -> bool:
        return self._contours is not None and len(self._contours) > 0

    @property
    def has_points(self) -> bool:
        return self._points is not None and len(self._points) > 0

    @property
    def raster(self) -> Optional[NDArray]:
        """The raster mask array. Lazy-loads from file if needed."""
        if self._raster is None and self._file_path is not None:
            self._load_raster()
        return self._raster

    @property
    def contours(self) -> Optional[List[NDArray]]:
        return self._contours

    @property
    def points(self) -> Optional[NDArray]:
        return self._points

    @property
    def data(self) -> Optional[NDArray]:
        """
        Primary raster representation.

        Returns the raster if available. In future, could render
        contours/points to raster if no raster is set.
        """
        if self.has_raster:
            return self.raster
        # TODO: render contours/points to raster
        return None

    @property
    def shape(self) -> Optional[Tuple[int, ...]]:
        if self.data is not None:
            return self.data.shape
        return None

    def _load_raster(self) -> None:
        """Load raster from file path."""
        if self._file_path is None:
            return
        try:
            img = PILImage.open(self._file_path).convert("L")
            self._raster = np.array(img)
        except (FileNotFoundError, OSError):
            self._raster = None
        self._loaded = True

    def load(self) -> None:
        """Force load raster from disk."""
        if not self._loaded:
            self._load_raster()

    def unload(self) -> None:
        """Release raster data from memory (can reload from file)."""
        if self._file_path is not None:
            self._raster = None
            self._loaded = False

    def __array__(self) -> NDArray:
        return self.data

    def _repr_png_(self):
        if self.data is not None:
            img = PILImage.fromarray(self.data.astype(np.uint8))
            return img._repr_png_()
        return None

    def __repr__(self):
        parts = []
        if self.has_raster:
            parts.append(f"raster={self.shape}")
        if self.has_contours:
            parts.append(f"contours={len(self._contours)}")
        if self.has_points:
            parts.append(f"points={len(self._points)}")
        return f"AnnotationSlice({', '.join(parts) or 'empty'})"


class AnnotationVolume(ArrayView):
    """
    A volume of AnnotationSlices — same indexing/slicing as BScanArray.

    Supports:
    - vol[i] -> AnnotationSlice
    - vol[2:5] -> AnnotationVolume (sliced)
    - vol.data -> 3D NDArray (stacked rasters)
    - len(vol) -> number of slices
    - iteration
    """
    def __init__(self, slices: List[AnnotationSlice]):
        self._slices = slices

    def _items(self) -> List[AnnotationSlice]:
        return self._slices

    @property
    def data(self) -> NDArray:
        """Stack all slice rasters into a 3D array."""
        arrays = []
        for s in self._slices:
            d = s.data
            arrays.append(d)

        # Get shape from first non-None
        shape = None
        for a in arrays:
            if a is not None:
                shape = a.shape
                break

        if shape is None:
            return np.array([])

        # Replace None with zeros
        stacked = [a if a is not None else np.zeros(shape, dtype=np.uint8) for a in arrays]
        return np.stack(stacked, axis=0)

    @property
    def images(self):
        """Return PIL images for visualisation compatibility."""
        from .image import ImageVolume
        pil_images = []
        for s in self._slices:
            if s.data is not None:
                pil_images.append(PILImage.fromarray(s.data.astype(np.uint8)))
            else:
                pil_images.append(None)
        return ImageVolume(images=[LazyImage(raw_image=img) if img else LazyImage() for img in pil_images])

    def preload(self) -> None:
        for s in self._slices:
            s.load()

    def unload(self) -> None:
        for s in self._slices:
            s.unload()

    def _repr_png_(self):
        mid = len(self._slices) // 2
        return self._slices[mid]._repr_png_()

    def __repr__(self):
        return f"AnnotationVolume(n_slices={len(self._slices)})"


class Annotation:
    """
    A feature-level annotation with metadata and data.

    An enface annotation has a single slice.
    An OCT annotation has N slices (one per B-scan).

    Parameters
    ----------
    slices : AnnotationVolume or AnnotationSlice or NDArray
        The annotation data. Accepts:
        - AnnotationVolume: used directly
        - AnnotationSlice: wrapped in a single-element volume
        - NDArray: 2D -> single slice, 3D -> volume of slices
    feature_name : str, optional
        Name of the annotated feature (e.g. 'GA', 'drusen').
    source_id : str, optional
        Source scan identifier.
    model_info : ModelInfo, optional
        Information about the model that produced this annotation.
    color : tuple, optional
        RGB color for visualisation.
    threshold : float
        Threshold for binarising probability masks.
    """
    def __init__(self,
                 slices=None,
                 feature_name: Optional[str] = None,
                 source_id: Optional[str] = None,
                 model_info: Optional[ModelInfo] = None,
                 color: Optional[tuple] = None,
                 threshold: float = 0.5):
        self.feature_name = feature_name
        self.source_id = source_id
        self.model_info = model_info
        self.color = color
        self.threshold = threshold

        # Normalise input to AnnotationVolume
        if slices is None:
            self._slices = AnnotationVolume([])
        elif isinstance(slices, AnnotationVolume):
            self._slices = slices
        elif isinstance(slices, AnnotationSlice):
            self._slices = AnnotationVolume([slices])
        elif isinstance(slices, np.ndarray):
            if slices.ndim == 2:
                self._slices = AnnotationVolume([AnnotationSlice(raster=slices)])
            elif slices.ndim == 3:
                self._slices = AnnotationVolume([
                    AnnotationSlice(raster=slices[i]) for i in range(slices.shape[0])
                ])
            else:
                raise ValueError(f"NDArray must be 2D or 3D, got {slices.ndim}D")
        elif isinstance(slices, list):
            # List of AnnotationSlice
            self._slices = AnnotationVolume(slices)
        else:
            raise TypeError(f"Cannot construct Annotation from {type(slices).__name__}")

    @property
    def slices(self) -> AnnotationVolume:
        return self._slices

    @property
    def data(self) -> NDArray:
        """
        The annotation data as an NDArray.

        For single-slice (enface): returns 2D array.
        For multi-slice (OCT): returns 3D array.
        """
        vol_data = self._slices.data
        if len(self._slices) == 1 and vol_data.ndim == 3:
            return vol_data[0]
        return vol_data

    @property
    def images(self):
        """Image representations for visualisation."""
        return self._slices.images

    @property
    def is_volume(self) -> bool:
        return len(self._slices) > 1

    @property
    def is_enface(self) -> bool:
        return len(self._slices) == 1

    def __len__(self):
        return len(self._slices)

    def __getitem__(self, index):
        return self._slices[index]

    def preload(self) -> None:
        self._slices.preload()

    def unload(self) -> None:
        self._slices.unload()

    def _repr_png_(self):
        return self._slices._repr_png_()

    def _ipython_display_(self) -> None:
        from IPython.display import display
        display(self._build_display_widget())

    def _build_display_widget(self):
        from .visualisation import image_array_display_widget, enface_display_widget
        if self.is_enface:
            img = PILImage.fromarray(self.data.astype(np.uint8))
            return enface_display_widget(img, width=320, height=320)
        else:
            return image_array_display_widget(self.images, width=320, height=320)

    def __repr__(self):
        return (
            f"Annotation(feature={self.feature_name!r}, "
            f"source_id={self.source_id!r}, "
            f"slices={len(self._slices)})"
        )


# ---------------------------------------------------------------------------
# Backward compatibility aliases
# ---------------------------------------------------------------------------
# These map old class names to new ones so existing code doesn't break.

# MaskImage is now just a LazyImage used for masks — keep as-is for loading
class MaskImage(LazyImage):
    """Lazy-loading mask image (backward compat)."""
    @property
    def image(self) -> Optional[PILImage.Image]:
        if not isinstance(self._file_location, str):
            if not self._raw_image:
                return None
        return super().image


class MaskVolume(ArrayView):
    """Volume of MaskImages (backward compat). Prefer AnnotationVolume for new code."""
    def __init__(self, masks: List[MaskImage]):
        self._masks = masks

    def _items(self) -> List[MaskImage]:
        return self._masks

    def _repr_png_(self):
        return self._masks[len(self._masks) // 2]._repr_png_()

    def preload(self) -> None:
        for mask in self._masks:
            mask.load()

    def unload(self) -> None:
        for mask in self._masks:
            mask.unload()

    @property
    def images(self):
        from .image import ImageVolume
        return ImageVolume([mask.image for mask in self._masks])

    @property
    def data(self) -> NDArray:
        """Stack mask data into 3D array."""
        arrays = [np.array(m.image) if m.image is not None else None for m in self._masks]
        shape = next((a.shape for a in arrays if a is not None), None)
        if shape is None:
            return np.array([])
        stacked = [a if a is not None else np.zeros(shape, dtype=np.uint8) for a in arrays]
        return np.stack(stacked, axis=0)


# Old annotation classes as thin wrappers around new Annotation
class AnnotationEnface(Annotation):
    """Backward-compatible enface annotation."""
    def __init__(self, mask=None, *args, **kwargs):
        if mask is not None:
            if isinstance(mask, MaskImage):
                # Load from MaskImage
                img = mask.image
                raster = np.array(img) if img is not None else None
                slices = AnnotationSlice(raster=raster, file_path=mask._file_location)
            elif isinstance(mask, np.ndarray):
                slices = AnnotationSlice(raster=mask)
            else:
                slices = mask
            super().__init__(slices=slices, *args, **kwargs)
        else:
            super().__init__(*args, **kwargs)


class AnnotationOCT(Annotation):
    """Backward-compatible OCT volume annotation."""
    def __init__(self, masks=None, slices=None, *args, **kwargs):
        if masks is not None and slices is None:
            if isinstance(masks, MaskVolume):
                # Convert MaskVolume to AnnotationVolume
                ann_slices = []
                for mask_img in masks._masks:
                    ann_slices.append(AnnotationSlice(
                        file_path=mask_img._file_location if isinstance(mask_img._file_location, str) else None,
                        raster=np.array(mask_img._raw_image) if mask_img._raw_image is not None else None,
                    ))
                slices = AnnotationVolume(ann_slices)
            elif isinstance(masks, np.ndarray):
                slices = masks  # Will be handled by Annotation.__init__
            super().__init__(slices=slices, *args, **kwargs)
        else:
            super().__init__(slices=slices, *args, **kwargs)
