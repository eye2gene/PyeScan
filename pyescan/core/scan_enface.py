from IPython.display import display
from numpy.typing import NDArray

from .scan import SingleImageScan
from .visualisation import enface_display_widget, overlay_masks


class EnfaceScan(SingleImageScan):
    """
    Base class for en-face (2D) retinal scans.
    """

    def _annotated_enface(self, features=None, max_height=320, alpha=0.5) -> NDArray:
        image = self.image
        masks = [annotation.images[0] for annotation in self.annotations.values()]
        annotated_image = overlay_masks(image, masks, feature_names=self.annotations.keys(), max_height=max_height, alpha=alpha)
        return annotated_image  # Should maybe convert to PIL image

    def _build_display_widget(self):
        if self.annotations:
            enface_image = self._annotated_enface()
        else:
            enface_image = self.image
        return enface_display_widget(enface_image, width=320, height=320)

    def _ipython_display_(self):
        display(self._build_display_widget())


class FAFScan(EnfaceScan):
    """Fundus autofluorescence scan."""
    pass


class IRScan(EnfaceScan):
    """Infrared reflectance scan."""
    pass


class ColorFundusScan(EnfaceScan):
    """Color fundus photograph."""
    pass
