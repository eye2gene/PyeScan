from cached_property import cached_property
import numpy as np
from numpy.typing import NDArray
from PIL import Image as PILImage
from typing import Any, List, Optional

from .utils import ArrayView

class BaseImage():
    pass #TODO make this an ABC

class LazyImage(BaseImage):
    """
    Holder for single image to enable lazy loading
    """
    def __init__(self, file_path: str = None, mode: str = None, raw_image: PILImage.Image = None):
        self._source_type = None # For future use to manage different types of loading
        self._file_location = file_path
        
        self._color_mode = mode
        
        # Set initial params
        self._raw_image: Optional[PILImage] = raw_image
        self._image: Optional[PILImage] = None
        
        self.loaded = False # Currently checking for _image is None

    def __getattr__(self, attr: str) -> Any:
        # Pass through other calls to underlying image
        return getattr(self.image, attr)
    
    def __array__(self):
        return self.data
    
    def _repr_png_(self):
        return self.image._repr_png_()
    
    def load(self, force: bool = False) -> None:
        #TODO: add proper debug logging
        #print("Loaded image", self._file_location)
        if self._file_location:
            self._raw_image = PILImage.open(self._file_location)
        self._image = self._raw_image
        if self._color_mode:
            self._image = self._image.convert(mode=self._color_mode)
        self.loaded = True
        
    def reload(self) -> None:
        # Forces reload
        self.load(force=True)
        
    def unload(self) -> None:
        if self._image is not None:
            self._image.close()
            self._image = None
            if self._file_location:
                self._raw_image = None
        self.loaded = False
        
    @property
    def image(self) -> Optional[PILImage.Image]:
        if not self.loaded:
            self.load()
        return self._image
    
    @property
    def data(self) -> NDArray:
        if self.image is None: return None # Note image vs _image
        return np.array(self.image)

class ImageVolume(ArrayView):
    """
    Holder for set of images
    """
    def __init__(self, images: Optional[List[BaseImage]] = None,
                 file_paths: Optional[List[str]] = None,
                 mode:str = None):
        if not images is None:
            self._images = images
        else:
            self._images = [ LazyImage(file_path, mode) for file_path in file_paths ]
        
    def _items(self) -> List[BaseImage]:
        return self._images
