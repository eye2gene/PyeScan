from typing import List

import numpy as np
from IPython.display import display
from numpy.typing import NDArray
from PIL import Image as PILImage
from skimage.transform import ProjectiveTransform, warp

from .image import BaseImage, ImageVolume
from .scan import BaseScan, SingleImageScan
from .scan_enface import EnfaceScan
from .utils import ArrayView, _pad_array
from .visualisation import (
    generate_distinct_colors,
    overlay_masks,
    overlay_rgba_images,
    render_volume_data,
)


class BScan(SingleImageScan):
    """
    Class for single OCT b-scan
    """
    def __init__(self, image: BaseImage, bscan_index: int, *args, **kwargs):
        super().__init__(image, *args, **kwargs)
        self._scan_index = bscan_index

class BScanArray(ArrayView):
    """
    Maybe slightly pointless wrapper for array of bscans
    """
    def __init__(self, bscans):
        self._bscans = bscans #TODO: Check type
        
    def _items(self) -> List[BScan]:
        return self._bscans
            
    def _repr_png_(self):
        return self._bscans[len(self._bscans)//2]._repr_png_()
    
    def preload(self) -> None:
        for bscan in self._bscans:
            bscan.preload()
        
    def unload(self) -> None:
        for bscan in self._bscans:
            bscan.unload()

    @property
    def images(self) -> ImageVolume:
        return ImageVolume([bscan.image for bscan in self._bscans])
    
    
class OCTScan(BaseScan):
    """
    Class for OCT scans
    """
    def __init__(self, enface: EnfaceScan, bscans: BScanArray, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        if enface:
            self._enface: EnfaceScan = enface #EnfaceScan
            self._enface.set_parent(self)
        
        self._bscans: BScanArray = bscans #BscanArray
        for bscan in self._bscans:
            bscan.set_parent(self)
        
    def _repr_png_(self):
        return self._bscans._repr_png_()
    
    def _ipython_display_(self):
        display(self._build_display_widget())
    
    def __getitem__(self, index: int):
        return self._bscans[index]
    
    def __len__(self):
        return len(self._bscans)
    
    def __array__(self):
        return self._bscans.data
    
    def preload(self):
        if self._enface:
            self._enface.preload()
        self._bscans.preload()
    
    def unload(self):
        if self._enface:
            self._enface.unload()
        self._bscans.unload()
        
    @property
    def image(self) -> BaseImage:
        return self._enface.image
    
    @property
    def images(self) -> ImageVolume:
        return self._bscans.images
    
    @property
    def data(self) -> NDArray:
        return self._bscans.data
    
    @property
    def shape(self): #TODO
        return self._bscans.data.shape
    
    def plot_image(self, include_annotations=False) -> None:
        raise NotImplementedError()

    @property
    def enface(self) -> EnfaceScan:
        return self._enface

    @property
    def bscans(self) -> BScanArray:
        return self._bscans
    
    def get_bscan_enface_locations(self) -> NDArray:
        """ Returns an Nx2 array of 2D (x,y) points with the start and end position for each bscan line """
        locations = []
        for bscan in self._bscans:
            location_start = (bscan.metadata.bscan_start_x, bscan.metadata.bscan_start_y)
            location_end = (bscan.metadata.bscan_end_x, bscan.metadata.bscan_end_y)
            locations.append((location_start, location_end))
        return np.array(locations)
    
    def _get_enface_transform(self, input_shape=None) -> ProjectiveTransform:
        """ Input shape should be h x w """
        bscan_locations = self.get_bscan_enface_locations()
        destination_pts = np.float32([bscan_locations[0,0],  bscan_locations[0,1],
                                      bscan_locations[-1,0], bscan_locations[-1,1]])

        if input_shape:
            h, w, *_ = input_shape
            source_pts = np.float32([[0, 0], [0, w-1], [h-1, 0], [h-1, w-1]])
        else:
            n = len(bscan_locations) # bit indirect but shoudl work
            w = self.bscans[0].image.width # bit of a hack
            source_pts = np.float32([[0, 0], [0, w-1], [n-1, 0], [n-1, w-1]])

        # Create the transform
        tform = ProjectiveTransform()
        tform.estimate(source_pts, destination_pts)
        return tform

    def project_to_enface(self, points: NDArray) -> NDArray:
        tform = self._get_enface_transform()
        return tform(np.array(points))

    def project_from_enface(self, points: NDArray) -> NDArray:
        tform = self._get_enface_transform()
        return tform.inverse(np.array(points))

    def transform_to_enface(self, image: NDArray) -> NDArray:
        image = np.array(image)
        tform = self._get_enface_transform(image.shape)

        # Apply the transform
        # Seeps to need the inverse trasfm, and also use transpose
        height, width = self.enface.image.height, self.enface.image.width
        warped = warp(image.swapaxes(0, 1), tform.inverse, output_shape=(height, width))

        #return warped[::-1,...] # reverse due to different indexing of y
        return warped
        
    def _annotated_bscan(self, bscan_index: int, features=None) -> NDArray:
        image = self.images[bscan_index]
        masks = [annotation.images.get(bscan_index, None) for annotation in self.annotations.values()]
        default_colors = generate_distinct_colors(len(self.annotations))
        colors = [(annotation._color or default_colors[i]) for i, annotation in enumerate(self.annotations.values())]
        annotated_image = overlay_masks(image, masks, colors=colors, feature_names=self.annotations.keys(), alpha=0.5)
        return annotated_image # Should maybe convert to PIL image
    
    def _annotated_enface(self,
                          heatmap: bool = True,
                          contours: bool = True,
                          alpha: float = 0.5) -> PILImage.Image:

        # Start with enface image
        image = self.enface.image
        img_array = np.array(image.convert('RGBA'))

        # Generate colors if not provided
        default_colors = generate_distinct_colors(len(self.annotations))
        colors = [annotation._color or default_colors[i] for i, annotation in enumerate(self.annotations.values())]

        # Create an empty array for the overlay
        projected_masks = []
        for annotation, color in zip(self.annotations.values(), colors):
            data= _pad_array(annotation.data, len(self))
            rendered_mask = render_volume_data(data, color=color, heatmap=heatmap, contours=contours)
            projected_mask = self.transform_to_enface(rendered_mask) * 255
            projected_mask[...,3] *= alpha
            projected_masks.append(projected_mask)

        # Apply alpha blending
        imgs = [img_array] + projected_masks
        result = overlay_rgba_images(imgs)
        result = result.astype(np.uint8)

        # Convert back to PIL Image for drawing text
        result_image = PILImage.fromarray(result)
        return result_image
        
    def _build_display_widget(self, enface_contours=True, enface_heatmap=True):
        from .visualisation import image_array_display_widget, oct_display_widget

        if self.annotations:
            annotated_images = list()
            for i, _ in enumerate(self.images):
                annotated_images.append(self._annotated_bscan(i))
            enface_image = self._annotated_enface(contours=enface_contours, heatmap=enface_heatmap) if self._enface else None
        else:
            annotated_images = self.images
            enface_image = self.enface.image if self._enface else None

        # Get bscan locations if available
        try:
            bscan_locations = self.get_bscan_enface_locations()
        except (AttributeError, TypeError):
            bscan_locations = None

        # Fall back to simple volume viewer if no enface or positions
        if enface_image is None or bscan_locations is None:
            return image_array_display_widget(annotated_images, width=640, height=320)

        return oct_display_widget(annotated_images, enface_image, bscan_locations, width=640, height=320, enface_size=320)