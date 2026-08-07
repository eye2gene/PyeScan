"""Image classes for PyeScan.

Design:
- BaseImage: ABC defining the interface all image types must satisfy (.data, .image)
- LazyImage: Lazy-loading image backed by a pluggable loader callable.
- ImageVolume: ArrayView of BaseImage instances (indexable, sliceable, stackable).

Loader factories (file_loader, url_loader, generator_loader) decouple the
loading strategy from the caching shell so new sources can be added without
modifying LazyImage.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image as PILImage

from .utils import ArrayView

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# Loader factories
# ---------------------------------------------------------------------------


def file_loader(
    path: str | Path, mode: str | None = None
) -> Callable[[], PILImage.Image]:
    """Create a loader that reads from a local file path."""
    path = Path(path)

    def _load() -> PILImage.Image:
        img = PILImage.open(path)
        img.load()  # force full read so file handle is released
        return img.convert(mode) if mode else img

    _load.__repr__ = lambda: f"file_loader({path})"
    return _load


def url_loader(url: str, mode: str | None = None) -> Callable[[], PILImage.Image]:
    """Create a loader that fetches an image from a URL."""
    import urllib.request
    from io import BytesIO

    def _load() -> PILImage.Image:
        with urllib.request.urlopen(url) as resp:
            img = PILImage.open(BytesIO(resp.read()))
            img.load()
        return img.convert(mode) if mode else img

    _load.__repr__ = lambda: f"url_loader({url})"
    return _load


def generator_loader(
    func: Callable[[], NDArray], mode: str | None = None
) -> Callable[[], PILImage.Image]:
    """Create a loader from a callable that returns a numpy array."""

    def _load() -> PILImage.Image:
        arr = func()
        img = PILImage.fromarray(arr)
        return img.convert(mode) if mode else img

    _load.__repr__ = lambda: f"generator_loader({func})"
    return _load


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class BaseImage(ABC):
    """Abstract base for all image types in PyeScan."""

    @property
    @abstractmethod
    def data(self) -> NDArray:
        """Pixel data as a numpy array."""
        ...

    def __array__(self) -> NDArray:
        return self.data


# ---------------------------------------------------------------------------
# LazyImage
# ---------------------------------------------------------------------------


class LazyImage(BaseImage):
    """Lazy-loading image backed by a pluggable loader callable.

    The loader is invoked on first access to `.image` or `.data`. Call
    `.unload()` to release memory; the next access will re-invoke the loader.

    Construction
    ------------
    Preferred (explicit loader):
        LazyImage(loader=file_loader("scan.png"))

    Convenience shortcuts (construct the loader internally):
        LazyImage(file_path="scan.png")
        LazyImage(file_path="scan.png", mode="L")
        LazyImage(url="https://example.com/retina.png")
        LazyImage(raw_image=pil_img)

    Custom loader:
        LazyImage(loader=my_dicom_loader)
    """

    def __init__(
        self,
        file_path: str | Path | None = None,
        mode: str | None = None,
        raw_image: PILImage.Image | None = None,
        *,
        loader: Callable[[], PILImage.Image] | None = None,
        url: str | None = None,
    ):
        if loader is not None:
            self._loader = loader
        elif file_path is not None:
            self._loader = file_loader(file_path, mode)
        elif url is not None:
            self._loader = url_loader(url, mode)
        elif raw_image is not None:
            # Already in memory — wrap in a trivial loader that returns a copy
            converted = raw_image.convert(mode) if mode else raw_image
            self._loader: Callable[[], PILImage.Image] | None = lambda: converted.copy()
        else:
            # Empty sentinel (used e.g. by annotation code for placeholder slots)
            self._loader = None

        self._image: PILImage.Image | None = None

        # Keep for backward-compat introspection (used by MaskImage, annotation code)
        self._file_location = str(file_path) if file_path else None
        self._raw_image = raw_image

    # ------------------------------------------------------------------
    # Loading interface
    # ------------------------------------------------------------------

    @property
    def loaded(self) -> bool:
        """True if the image is currently held in memory."""
        return self._image is not None

    def load(self) -> None:
        """Load (or reload) the image into memory."""
        if self._loader is None:
            raise ValueError("No loader configured for this LazyImage.")
        self._image = self._loader()

    def unload(self) -> None:
        """Release the in-memory image. Can be reloaded later if a loader exists."""
        if self._image is not None:
            self._image.close()
            self._image = None

    # ------------------------------------------------------------------
    # Data access (triggers load on first access)
    # ------------------------------------------------------------------

    @property
    def image(self) -> PILImage.Image | None:
        """The underlying PIL image. Loads on first access."""
        if self._loader is None and self._image is None:
            return None
        if self._image is None:
            self.load()
        return self._image

    @property
    def data(self) -> NDArray | None:
        """Numpy array of pixel data. Loads on first access."""
        img = self.image
        if img is None:
            return None
        return np.asarray(img)

    # ------------------------------------------------------------------
    # PIL attribute pass-through (width, height, size, etc.)
    # ------------------------------------------------------------------

    @property
    def width(self) -> int | None:
        img = self.image
        return img.width if img else None

    @property
    def height(self) -> int | None:
        img = self.image
        return img.height if img else None

    @property
    def size(self) -> tuple | None:
        img = self.image
        return img.size if img else None

    def convert(self, mode: str) -> PILImage.Image:
        """Convert to a PIL Image with the given mode. Triggers load."""
        return self.image.convert(mode)

    def copy(self) -> PILImage.Image:
        """Return a copy of the underlying PIL Image. Triggers load."""
        return self.image.copy()

    def resize(self, size, resample=None) -> PILImage.Image:
        """Resize and return a PIL Image. Triggers load."""
        if resample is not None:
            return self.image.resize(size, resample)
        return self.image.resize(size)

    def save(self, fp, format=None, **kwargs) -> None:
        """Save the image to file. Triggers load."""
        self.image.save(fp, format=format, **kwargs)

    # ------------------------------------------------------------------
    # Representations
    # ------------------------------------------------------------------

    def _repr_png_(self):
        """Jupyter notebook display support."""
        img = self.image
        if img is None:
            return None
        return img._repr_png_()

    def __repr__(self) -> str:
        status = "loaded" if self.loaded else "not loaded"
        source = self._file_location or ("in-memory" if self._loader else "empty")
        return f"LazyImage({source}, {status})"

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.unload()


# ---------------------------------------------------------------------------
# ImageVolume
# ---------------------------------------------------------------------------


class ImageVolume(ArrayView):
    """Holder for a set of images, supporting lazy-loading and numpy-style indexing."""

    def __init__(
        self,
        images: list[BaseImage] | None = None,
        file_paths: list[str] | None = None,
        mode: str | None = None,
    ):
        if images is not None:
            self._images = images
        else:
            self._images = [LazyImage(file_path, mode) for file_path in file_paths]

    def _items(self) -> list[BaseImage]:
        return self._images
