from abc import ABCMeta, abstractmethod, abstractproperty
from cached_property import cached_property
import math
from numpy.typing import NDArray
from PIL import Image as PILImage
from typing import Any, List, Optional

from .image import LazyImage, ImageVolume
from .utils import ArrayView

class ModelInfo():
    """
    Datastructure for holding information about a model
    """
    def __init__(self, name: str = None):
        self.name:str = name
    
class FeatureInfo():
    """
    Datastructure for holding information about a model 
    """
    def __init__(self):
        self.name: str = None
        
class MaskImage(LazyImage):
    @property
    def image(self) -> PILImage.Image:
        # Add exception for Missing filepath as we don't want to throw exception
        #if self._file_location is None or math.isnan(self._file_location):
        if not isinstance(self._file_location, str):
            if not self._raw_image:
                return None
        return super().image

# Could probably just use a transparent arrayview here, rather than parent class
# TODO: Handle missing masks gracefully
class MaskVolume(ArrayView):
    """
    Maybe slightly pointless wrapper for array of bscans
    """
    def __init__(self, masks: List[MaskImage]):
        self._masks = masks
        
    def _items(self) -> List[MaskImage]:
        return self._masks
            
    def _repr_png_(self):
        return self._masks[len(self._masks)//2]._repr_png_()
    
    def preload(self) -> None:
        for mask in self._masks:
            mask.preload()
        
    def unload(self) -> None:
        for mask in self._masks:
            mask.unload()

    @property
    def images(self) -> ImageVolume:
        return ImageVolume([mask.image for mask in self._masks])
    

class Annotation(metaclass=ABCMeta):
    """
    Base class for holding annotation information
    """
    def __init__(self,
                 basescan: "BaseScan" = None,
                 name: Optional[str] = None,
                 modelinfo: Optional[ModelInfo] = None):
        self._scan = basescan
        self._name = name
        self._modelinfo = modelinfo
        
        self._mask = None
        self._color = None
        self._threshold = 0.5
        
    #@abstractmethod
    def get_area_px(self):
        raise NotImplementedError()
        
    #@abstractmethod
    def get_area_mm(self):
        raise NotImplementedError()
        
    #@abstractmethod
    def get_volume_px(self):
        raise NotImplementedError()
        
    #@abstractmethod
    def get_volume_mm(self):
        raise NotImplementedError()
        
    @abstractproperty
    def data(self) -> NDArray:
        raise NotImplementedError()
    
    
class AnnotationEnface(Annotation):
    """
    Base class for holding annotation information
    """
    def __init__(self, mask: MaskImage, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._mask = mask #MaskImage
        
        
class AnnotationOCT(Annotation):
    """
    Base class for holding annotation information
    """
    def __init__(self, masks: MaskVolume, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._masks = masks
        
    def _repr_png_(self):
        return self._masks._repr_png_()
    
    def _ipython_display_(self) -> None:
        from IPython.display import display, Image
        display(self._build_display_widget())
        
    @property
    def images(self) -> ImageVolume:
        return self._masks.images
    
    @property
    def data(self) -> NDArray:
        return self._masks.data
   
    def _build_display_widget(self):
        from .visualisation import image_array_display_widget
        return image_array_display_widget(self.images, width=320, height=320)
        