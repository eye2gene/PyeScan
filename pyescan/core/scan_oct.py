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
        """
        Project an array from OCT space to enface pixel coordinates.

        Handles:
        - 2D (n_bscans x width): single-channel projection (e.g. presence map)
        - 3D (height x width x channels) where channels <= 4: multi-channel
          image warp (e.g. RGBA)
        - 3D (n_bscans x height x width): projects each depth row, returns
          (height x enface_h x enface_w) — useful for full volume projection

        Parameters
        ----------
        image : NDArray
            2D or 3D array in OCT space.

        Returns
        -------
        NDArray
            Warped array in enface pixel coordinates.
        """
        image = np.array(image)
        enface_h, enface_w = self.enface.image.height, self.enface.image.width

        if image.ndim == 2:
            tform = self._get_enface_transform(image.shape)
            return warp(image.swapaxes(0, 1), tform.inverse, output_shape=(enface_h, enface_w))

        elif image.ndim == 3:
            # Distinguish multi-channel image (h, w, c) from volume (n, h, w)
            if image.shape[2] <= 4:
                # Multi-channel image — use 2D warp (skimage handles channels)
                tform = self._get_enface_transform(image.shape[:2])
                return warp(image.swapaxes(0, 1), tform.inverse, output_shape=(enface_h, enface_w))
            else:
                # Volume (n_bscans, depth, width) — warp each depth slice
                n_bscans, depth, width = image.shape
                tform = self._get_enface_transform((n_bscans, width))
                result = np.zeros((depth, enface_h, enface_w), dtype=np.float64)
                for d in range(depth):
                    slice_2d = image[:, d, :]  # (n_bscans, width)
                    result[d] = warp(slice_2d.swapaxes(0, 1), tform.inverse,
                                     output_shape=(enface_h, enface_w))
                return result

        else:
            raise ValueError(f"transform_to_enface expects 2D or 3D input, got {image.ndim}D")

    def annotation_to_enface(self, annotation) -> "Annotation":
        """
        Project an OCT volume annotation to enface space.

        Collapses the depth axis (binary presence per A-scan) then projects
        to enface coordinates using transform_to_enface.

        For custom reductions (thickness maps, etc.), use transform_to_enface
        directly with a pre-reduced array.

        Parameters
        ----------
        annotation : Annotation
            An OCT volume annotation (multiple slices).

        Returns
        -------
        Annotation
            A new single-slice enface annotation.
        """
        from .annotation import Annotation, AnnotationSlice
        from .utils import _pad_array

        vol_data = annotation.data
        if vol_data.ndim == 2:
            return Annotation(
                slices=AnnotationSlice(raster=vol_data),
                feature_name=annotation.feature_name,
                source_id=annotation.source_id,
                model_info=annotation.model_info,
                color=annotation.color,
            )

        # Pad to match scan length, collapse depth, project
        feat_data = _pad_array(vol_data.astype(np.uint8), len(self))
        presence = feat_data.any(axis=1).astype(np.float64)  # (n_bscans x width)
        projected = self.transform_to_enface(presence)
        projected_mask = (projected * 255).astype(np.uint8)

        return Annotation(
            slices=AnnotationSlice(raster=projected_mask),
            feature_name=annotation.feature_name,
            source_id=annotation.source_id,
            model_info=annotation.model_info,
            color=annotation.color,
        )

    def annotation_to_bscans(self, annotation) -> "Annotation":
        """
        Project an enface annotation to B-scan space.

        Takes a single-slice (enface) annotation and projects it onto each
        B-scan, producing a verticalised volume (binary presence per A-scan).

        Parameters
        ----------
        annotation : Annotation
            An enface annotation (single slice with 2D raster).

        Returns
        -------
        Annotation
            A new volume annotation with one slice per B-scan.
        """
        from .annotation import Annotation, AnnotationSlice, AnnotationVolume

        enface_mask = annotation.data
        if enface_mask.ndim != 2:
            raise ValueError(
                f"annotation_to_bscans expects a 2D enface annotation, "
                f"got shape {enface_mask.shape}"
            )

        enface_mask_binary = enface_mask > (annotation.threshold * 255 if enface_mask.max() > 1 else annotation.threshold)
        eH, eW = enface_mask_binary.shape[:2]

        w = self.bscans[0].image.width
        h = self.bscans[0].image.height
        nrows = len(self)

        # Build the enface projection map for all bscan points
        pts = np.indices((nrows, w)).transpose(1, 2, 0).reshape((-1, 2))
        proj_pts = self.project_to_enface(pts).reshape((nrows, w, 2))

        slices = []
        for bscan_index in range(nrows):
            bscan_mask = np.zeros((h, w), dtype=np.uint8)
            for j, (enface_x, enface_y) in enumerate(proj_pts[bscan_index]):
                yi = max(0, min(int(round(enface_y)), eH - 1))
                xi = max(0, min(int(round(enface_x)), eW - 1))
                if enface_mask_binary[yi, xi]:
                    bscan_mask[:, j] = 255
            slices.append(AnnotationSlice(raster=bscan_mask))

        return Annotation(
            slices=AnnotationVolume(slices),
            feature_name=annotation.feature_name,
            source_id=annotation.source_id,
            model_info=annotation.model_info,
            color=annotation.color,
        )
        
    def _annotated_bscan(self, bscan_index: int, features=None) -> NDArray:
        image = self.images[bscan_index]
        masks = [annotation.images.get(bscan_index, None) for annotation in self.annotations.values()]
        default_colors = generate_distinct_colors(len(self.annotations))
        colors = [(annotation.color or default_colors[i]) for i, annotation in enumerate(self.annotations.values())]
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
        colors = [annotation.color or default_colors[i] for i, annotation in enumerate(self.annotations.values())]

        # Create an empty array for the overlay
        projected_masks = []
        for annotation, color in zip(self.annotations.values(), colors):
            data= _pad_array(annotation.data, len(self))
            rendered_mask = render_volume_data(data, color=color, heatmap=heatmap, contours=contours)
            projected_mask = self.transform_to_enface(rendered_mask) * 255
            projected_mask = projected_mask.astype(np.float64)
            projected_mask[..., 3] *= alpha
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