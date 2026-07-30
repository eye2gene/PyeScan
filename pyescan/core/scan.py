from abc import ABCMeta, abstractmethod
from functools import cached_property
from numpy.typing import NDArray
from typing import Any, Dict, List, Union, Optional

from .annotation import Annotation
from .image import BaseImage
from .metadata import MetadataView

class BaseScan(metaclass=ABCMeta): # We could probably make this into a metaclass itself
    def __init__(self,
                 metadata: MetadataView = None,
                 parent: "BaseScan" = None,
                 *args, **kwargs):
        self._record = None
        #self._group_id = None # This should uniquely identify the high-level scan within the record
        
        self._metadata = metadata # MetadataView onto a record

        self._annotations = dict() # Keep a reference to annotations in the class?
        
        self._parent = parent
    
    def __repr__(self):
        return f"{self.__class__.__name__}(source_id={self.source_id})"

    def __str__(self):
        return str(self.__repr__())
    
    @abstractmethod
    def plot_image(self, include_annotations: bool = False):
        raise NotImplementedError()
    
    @abstractmethod
    def preload(self) -> None:
        """
        Force load of all images/masks/etc, may be useful for dataloading
        """
        raise NotImplementedError()
    
    @abstractmethod
    def unload(self) -> None:
        """
        Force unload of all images/masks/etc, could be useful for saving memory
        """
        raise NotImplementedError()
        
    #@abstractmethod
    def _save(self, path_to_output_folder: str, format: Optional[str] = None) -> None:
        raise NotImplementedError()
        
    def find_annotations(self):
        """ Try search for annotations automatically """
        raise NotImplementedError()
        
    def add_annotation(self, feature_name: str, annotation: Annotation, color=None) -> None:
        annotation._scan = self
        if not (color is None):
            annotation._color = color
        self._annotations[feature_name] = annotation
        
    def add_annotations(self, annotation_dict: Dict[str,Annotation]) -> None:
        for feature_name, annotation in annotation_dict.items():
            self.add_annotation(feature_name, annotation)
            
    def set_parent(self, parent_scan: "BaseScan"):
        self._parent = parent_scan
        
    @property
    def annotations(self) -> Dict[str, Annotation]:
        return self._annotations

    @property
    @abstractmethod
    def image(self) -> BaseImage:
        raise NotImplementedError()

    @property
    @abstractmethod
    def data(self) -> NDArray:
        raise NotImplementedError()
        
    @property
    def metadata(self) -> MetadataView:
        return self._metadata

    @cached_property
    def modality(self) -> str:
        # Can maybe return the class name?
        return self._metadata.modality

    @cached_property
    def patient_id(self) -> str:
        raise NotImplementedError()
        
    @cached_property
    def scan_id(self) -> str:
        raise NotImplementedError()
        
    @cached_property
    def group_id(self) -> str:
        return self.metadata.group
        
    @cached_property
    def source_id(self) -> str:
        return self.metadata.source_id
        
    @cached_property
    def laterality(self) -> str:
        return self.metadata.laterality

    @cached_property
    def scan_angle(self) -> float:
        raise NotImplementedError()
        
    
class SingleImageScan(BaseScan):
    """
    Base class for scan as a single image / bscan
    """
    def __init__(self, image: BaseImage, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._image = image
        
    def _repr_png_(self):
        return self._image._repr_png_()
    
    def __array__(self):
        return self._image.data
      
    @property
    def image(self) -> BaseImage:
        return self._image
    
    @property
    def data(self) -> NDArray:
        return self._image.data
    
    @property
    def shape(self): #TODO: Type annotation
        return self._image.data.shape
    
    def plot_image(self, include_annotations=False):
        raise NotImplementedError()
    
    def preload(self) -> None:
        self._image.load()
    
    def unload(self) -> None:
        self._image.unload()

    
    