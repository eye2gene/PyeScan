import numpy as np
import PIL.Image
from typing import Any, Dict, List, Optional, Tuple

from .registry import pyescan_metric, Meta, MaskStat, ImgStat, Spec, Pred

@pyescan_metric()
def get_mask(file_path_mask: str) -> Tuple[Spec[np.array]]:
    try:
        mask = np.array(PIL.Image.open(file_path_mask).convert("L")) > 0
        return mask,
    except FileNotFoundError as e: 
        return None,


@pyescan_metric()
def get_image(file_path_scan: str) -> Tuple[Spec[np.array]]:
    try:
        image = np.array(PIL.Image.open(file_path_scan).convert("L"))
        return image,
    except FileNotFoundError as e: 
        return None,


@pyescan_metric()
def get_mask_shape(mask: Spec[np.array]) -> Tuple[MaskStat[int], MaskStat[int]]:
    mask_height_px, mask_width_px = mask.shape
    return mask_width_px, mask_height_px


@pyescan_metric()
def get_mask_resoluton(
        scan_width_px: Meta[int],
        scan_height_px: Meta[int],
        resolutions_mm_width: Meta[float],
        resolutions_mm_height: Meta[float],
        mask_width_px: MaskStat[int],
        mask_height_px: MaskStat[int],
    ) -> Tuple[MaskStat[float], MaskStat[float]]:

    mask_resolutions_mm_width = scan_width_px / mask_width_px * resolutions_mm_width
    mask_resolutions_mm_height = scan_height_px / mask_height_px * resolutions_mm_height
    return mask_resolutions_mm_width, mask_resolutions_mm_height


@pyescan_metric()
def get_mask_pixel_counts(mask: Spec[np.array]) -> Tuple[MaskStat[float], MaskStat[float], MaskStat[float]]:
    mask_pixel_count = mask.sum()
    mask_columns_count = mask.any(axis=0).sum()
    mask_rows_count = mask.any(axis=1).sum()
    return mask_pixel_count, mask_columns_count, mask_rows_count


@pyescan_metric()
def get_fovea_enface_position(
        bscan_index: Meta[int],
        scan_width_px: Meta[int],
        resolutions_mm_width: Meta[float],
        resolutions_mm_depth: Meta[float],
        bscan_location_start_x: Meta[int],
        bscan_location_start_y: Meta[int],
        bscan_location_end_x: Meta[int],
        bscan_location_end_y: Meta[int],
        fovea_x: Pred[int],
        fovea_bscan_index: Pred[int],
    ) -> Tuple[ImgStat[float], ImgStat[float]]:
    # Calculates enface x,y position of fovea using information from current bscan only
    import numpy as np
    
    # Get bscan start and end positions of current b-scan
    bscan_start_x, bscan_start_y = bscan_location_start_x, bscan_location_start_y
    bscan_end_x, bscan_end_y = bscan_location_end_x, bscan_location_end_y
    bscan_width = scan_width_px * resolutions_mm_width # prefer using scan info
    bscan_depth = resolutions_mm_depth
    
    # Get fovea's OCT coordinates
    fovea_u, fovea_v = fovea_x, fovea_bscan_index
    
    # Calculate fovea enface x and y by projecting from current b-scan
    bscan_dir_x = bscan_end_x - bscan_start_x
    bscan_dir_y = bscan_end_y - bscan_start_y
    fovea_t = fovea_u / scan_width_px
    fovea_projection_x = bscan_start_x + fovea_t * bscan_dir_x
    fovea_projection_y = bscan_start_y + fovea_t * bscan_dir_y
    
    # Use the enface length to estimate the bscan spacing in enface px
    bscan_length = np.sqrt(bscan_dir_x**2 + bscan_dir_y**2)
    enface_pixel_per_mm = bscan_length / bscan_width
    bscan_spacing = enface_pixel_per_mm * bscan_depth
    perp_length = ( bscan_index - fovea_v ) * bscan_spacing
    
    # Add perpendicular component
    fovea_enface_x = fovea_projection_x - perp_length * (bscan_dir_y / bscan_length)
    fovea_enface_y = fovea_projection_y + perp_length * (bscan_dir_x / bscan_length)
    
    return fovea_enface_x, fovea_enface_y





################################################################################
########## FAF PIXEL COUNTS


@pyescan_metric(
  returns = ["distance_mask_<diameter>_mm"],
  parameters = ["diameter"],
)
def _get_distance_mask(
        scan_width_px: Meta[int],
        resolutions_mm_width: Meta[float],
        resolutions_mm_height: Meta[float],
        mask_width_px: MaskStat[int],
        mask_height_px: MaskStat[int],
        fovea_enface_x: Pred[int],
        fovea_enface_y: Pred[int],
        diameter: float,
    ) -> Tuple[Spec[np.array]]:
    from skimage.draw import disk

    radius  = float(diameter) / 2
    radius_scale_factor = mask_width_px / scan_width_px
    
    radius_scaled = radius / resolutions_mm_width #assuming the same in both dims
    radius_scaled *= radius_scale_factor
    
    distance_mask = np.zeros((mask_height_px, mask_width_px))
    center = fovea_enface_y, fovea_enface_x # y, x (y=0 top)
    shape = mask_height_px, mask_width_px # h, w
    rr, cc = disk(center, radius_scaled, shape=shape)
    distance_mask[rr,cc] = 1.
    
    return distance_mask,


@pyescan_metric()
def _get_quadrant_masks(
        mask_width_px: MaskStat[int],
        mask_height_px: MaskStat[int],
        fovea_enface_x: Pred[int],
        fovea_enface_y: Pred[int]
    ) -> Tuple[Spec[np.array], Spec[np.array], Spec[np.array], Spec[np.array]]:
    import numpy as np
    
    # Should probably normalise fovea position by actual image size, not just assume mask size = img size...

    im_w = mask_width_px
    im_h = mask_height_px
    
    center_x, center_y = fovea_enface_x, fovea_enface_y
    
    # 0,0 on enface corresponds to upper left (when facing image), i.e. y is inverted
    # upper left half is x < -y or -x > y, upper right is y < x or -y > -x
    
    y, x = np.ogrid[:im_h, :im_w]
    upper_left  = (center_y - y >= x - center_x) 
    upper_right = (center_y - y >= center_x - x) 
    
    # Get insection of mask pairs
    quadrant_mask_superior =  upper_left &  upper_right # UpL+UpR
    quadrant_mask_dexter   = ~upper_left &  upper_right # LowR+UpR
    quadrant_mask_inferior = ~upper_left & ~upper_right # LowR+LowL
    quadrant_mask_sinister =  upper_left & ~upper_right # UpL+LowL
    
    # Explicit return for automatic naming
    return ( quadrant_mask_superior,
             quadrant_mask_dexter, # Right as viewed
             quadrant_mask_inferior,
             quadrant_mask_sinister ) # Left as viewed


@pyescan_metric(
    requires=["spec:mask", "stat:quadrant_mask_<quadrant>"],
    returns=["pixel_count_<quadrant>"],
    parameters = ["quadrant"],
)
def get_pixel_count_by_quadrant(
        mask: Spec[np.array],
        quadrant_mask: Spec[np.array],
        quadrant: str,
    ) -> Tuple[MaskStat[int]]:
    masked_img = mask * quadrant_mask
    pixel_count = masked_img.sum()
    return pixel_count,


@pyescan_metric(
    requires=["spec:mask", "stat:distance_mask_<diameter>_mm"],
    returns=["mask_pixel_count_<diameter>_mm"],
    parameters = ["diameter"],
)
def get_pixel_count_by_distance(
        mask: Spec[np.array],
        distance_mask: Spec[np.array],
        diameter: float,
    ) -> Tuple[MaskStat[int]]:
    masked_img = mask * distance_mask
    pixel_count = masked_img.sum()
    return pixel_count,


@pyescan_metric(
    requires=[
        "spec:mask",
        "spec:distance_mask_<diameter>_mm",
        "spec:quadrant_mask_<quadrant>"
    ],
    returns=["mask_pixel_count_<diameter>_mm_<quadrant>"],
    parameters = ["diameter", "quadrant"],
)
def get_pixel_count_by_distance_quadrant(
        mask: Spec[np.array],
        distance_mask: Spec[np.array],
        quadrant_mask: Spec[np.array],
        diameter: float,
        quadrant: str,
    ) -> Tuple[MaskStat[int]]:
    masked_img = mask * distance_mask * quadrant_mask
    pixel_count = masked_img.sum()
    return pixel_count,





################################################################################
########## OCT PIXEL COUNTS


def _get_circle_line_intersection(circle_centre, radius, line_start, line_end):
    # Used for distance mask calculations
    # TODO: Move to utils?
    import math

    dx = line_end[0] - line_start[0]
    dy = line_end[1] - line_start[1]
    
    offset_x = line_start[0] - circle_centre[0]
    offset_y = line_start[1] - circle_centre[1]
    
    # Formula from ChatGPT
    # Could look up derivation, should probably double check
    # Is presumably line given as interpolation with t, then solve for t
    # in circle eqn
    a = dx**2 + dy**2
    b = 2 * (dx * offset_x + dy * offset_y) 
    c = offset_x**2 + offset_y**2 - radius**2

    discriminant = b**2 - 4 * a * c

    if discriminant < 0:
        # No intersection
        return []

    t1 = (-b - math.sqrt(discriminant)) / (2 * a)
    t2 = (-b + math.sqrt(discriminant)) / (2 * a)

    return t1, t2


@pyescan_metric(
    returns = ["distance_mask_slice_<diameter>_mm"],
    parameters = ["diameter"],
)
def _get_distance_mask_slice(
        bscan_index: Meta[int],
        scan_width_px: Meta[int],
        resolutions_mm_width: Meta[float],
        resolutions_mm_depth: Meta[float],
        mask_width_px: MaskStat[int],
        mask_height_px: MaskStat[int],
        fovea_x: Pred[int],
        fovea_bscan_index: Pred[int],
        diameter: float,
    ) -> Tuple[Spec[np.array]]:
    
    import numpy as np
    radius = float(diameter) / 2.
    
    bscan_start = 0, bscan_index * resolutions_mm_depth
    bscan_end = scan_width_px * resolutions_mm_width, bscan_index * resolutions_mm_depth

    fovea_location = fovea_x * resolutions_mm_width, \
                     fovea_bscan_index * resolutions_mm_depth
    
    intersection = _get_circle_line_intersection(fovea_location, radius, bscan_start, bscan_end)

    #mask = np.zeros((mask_height_px, mask_width_px))\
    mask = np.zeros((1, mask_width_px)) # Use broadcasting for performance
    if intersection:
        start = max(0.0, intersection[0])
        end = min(1.0, intersection[1])
        mask[:,int(start*mask_width_px):int(end*mask_width_px)-1] = 1.0
    distance_mask_slice = mask
    return distance_mask_slice,


@pyescan_metric()
def _get_quadrant_masks_slice(
        bscan_location_start_x: Meta[int],
        bscan_location_start_y: Meta[int],
        bscan_location_end_x: Meta[int],
        bscan_location_end_y: Meta[int],
        mask_width_px: MaskStat[int],
        mask_height_px: MaskStat[int],
        fovea_enface_x: Pred[int],
        fovea_enface_y: Pred[int],
    ) -> Tuple[Spec[np.array], Spec[np.array], Spec[np.array], Spec[np.array]]:
    import numpy as np
    
    # Idea - make masks for upper left and upper right and multiply them
    
    # Get bscan start and end positions of current b-scan
    bscan_start_x, bscan_start_y = bscan_location_start_x, bscan_location_start_y
    bscan_end_x, bscan_end_y = bscan_location_end_x, bscan_location_end_y
    
    # Get B-scan's start/end points relative to fovea
    start_x, start_y = bscan_start_x - fovea_enface_x, bscan_start_y - fovea_enface_y
    end_x, end_y = bscan_end_x - fovea_enface_x, bscan_end_y - fovea_enface_y
    
    # 0,0 on enface correspoponds to upper left (when facing image), i.e. y is inverted
    # upper left half is x < -y or -x > y, upper right is y < x
    # rearranging for co-ord transform 0 < -(y+x) and 0 < (x-y)
    
    # Do co-ordinate shift w=y-x, z=y+x (equiv to rotation + scale)
    start_w, start_z = - start_x - start_y, start_x - start_y
    end_w, end_z = - end_x - end_y, end_x - end_y
    
    def _get_intercept_mask(start_v, end_v):
        mask = np.zeros((1, mask_width_px)) # Use broadcasting for performance
        t = - start_v / (end_v - start_v) # axis intercept
        if 0 <= t <= 1:
            x_intercept = int(t * (mask_width_px - 1))
            if end_v > start_v:
                mask[...,x_intercept:] = 1.
            else:
                mask[...,:x_intercept] = 1.
        elif start_v > 0:
            mask[...] = 1.
        return mask
    
    # Calculate mask of upper and left quadrants, -y > x, w > 0
    upper_left = _get_intercept_mask(start_w, end_w)
    
    # Calculate mask of upper and right quadrants, -y > -x, z > 0
    upper_right = _get_intercept_mask(start_z, end_z)
    
    # Multiply masks together to get the intersection - just need to match names
    # su upper_left x upper_right lust gives upper, and so on
    quadrant_mask_slice_superior = upper_left     * upper_right     # Upper
    quadrant_mask_slice_dexter   = (1-upper_left) * upper_right     # Right (as viewed)
    quadrant_mask_slice_inferior = (1-upper_left) * (1-upper_right) # Lower
    quadrant_mask_slice_sinister = upper_left     * (1-upper_right) # left (as viewed)
    return ( quadrant_mask_slice_superior,
             quadrant_mask_slice_dexter, #Right as viewed
             quadrant_mask_slice_inferior,
             quadrant_mask_slice_sinister ) #Left as viewed


@pyescan_metric(
    requires=["spec:mask", "stat:quadrant_mask_slice_<quadrant>"],
    returns=["mask_pixel_count_slice_<quadrant>", "mask_pixel_count_projected_<quadrant>"],
    parameters = ["quadrant"],
)
def get_pixel_count_by_quadrant_slice(
        mask: Spec[np.array],
        quadrant_mask_slice: Spec[np.array],
        quadrant: str,
    ) -> Tuple[MaskStat[int], MaskStat[int]]:
    masked_img = mask * quadrant_mask_slice
    pixel_count = masked_img.sum()
    columns_count = masked_img.any(axis=0).sum()
    return pixel_count, columns_count


@pyescan_metric(
    requires=["spec:mask", "stat:distance_mask_slice_<diameter>_mm"],
    returns=["mask_pixel_count_slice_<diameter>_mm", "mask_pixel_count_projected_<diameter>_mm"],
    parameters = ["diameter"],
)
def get_pixel_count_by_distance_slice(
        mask: Spec[np.array],
        distance_mask_slice: Spec[np.array],
        diameter: float,
    ) -> Tuple[MaskStat[int], MaskStat[int]]:
    masked_img = mask * distance_mask_slice
    pixel_count = masked_img.sum()
    columns_count = masked_img.any(axis=0).sum()
    return pixel_count, columns_count


@pyescan_metric(
    requires=[
        "spec:mask",
        "spec:distance_mask_slice_<diameter>_mm",
        "spec:quadrant_mask_slice_<quadrant>"
    ],
    returns=[
        "mask_pixel_count_slice_<diameter>_mm_<quadrant>",
        "mask_pixel_count_projected_<diameter>_mm_<quadrant>",
    ],
    parameters = ["diameter", "quadrant"],
)
def get_pixel_count_by_distance_quadrant_slice(
        mask: Spec[np.array],
        distance_mask_slice: Spec[np.array],
        quadrant_mask_slice: Spec[np.array],
        diameter: float,
        quadrant: str,
    ) -> Tuple[MaskStat[int], MaskStat[int]]:
    masked_img = mask * distance_mask_slice * quadrant_mask_slice
    pixel_count = masked_img.sum()
    columns_count = masked_img.any(axis=0).sum()
    return pixel_count, columns_count





################################################################################
########## AREAS AND VOLUMES


@pyescan_metric()
def get_mask_area(
        mask_resolutions_mm_width: MaskStat[float],
        mask_resolutions_mm_height: MaskStat[float],
        mask_pixel_count: MaskStat[int],
        mask_columns_count: MaskStat[int],
        mask_rows_count: MaskStat[int]
    ) -> Tuple[MaskStat[float], MaskStat[float], MaskStat[float]]:
    mask_area = mask_pixel_count * mask_resolutions_mm_width * mask_resolutions_mm_height
    mask_horizontal_extent = mask_columns_count * mask_resolutions_mm_width
    mask_vertical_extent = mask_rows_count * mask_resolutions_mm_height
    return mask_area, mask_horizontal_extent, mask_vertical_extent


@pyescan_metric()
def get_mask_volume(
        mask_area: MaskStat[float],
        mask_horizontal_extent: MaskStat[float],
        resolutions_mm_depth: Meta[float], #bscan-spacing
    ) -> Tuple[MaskStat[float], MaskStat[float]]:

    mask_volume = mask_area * resolutions_mm_depth
    mask_enface_area = mask_horizontal_extent * resolutions_mm_depth
    return mask_volume, mask_enface_area


@pyescan_metric(
    requires=[
        "stat:mask_resolutions_mm_width",
        "stat:mask_resolutions_mm_height",
        "stat:mask_pixel_count_<diameter>_mm",
    ],
    returns=[
        "mask_area_<diameter>_mm",
    ],
    parameters = ["diameter"],
)
def get_area_by_distance(
        mask_resolutions_mm_width: MaskStat[float],
        mask_resolutions_mm_height: MaskStat[float],
        pixel_count_region: MaskStat[int],
        diameter: float,
    ) -> Tuple[MaskStat[float],]:
    mask_area = pixel_count_region * mask_resolutions_mm_width * mask_resolutions_mm_height
    return mask_area,


@pyescan_metric(
    requires=[
        "stat:mask_resolutions_mm_width",
        "stat:mask_resolutions_mm_height",
        "stat:mask_pixel_count_<quadrant>",
    ],
    returns=[
        "mask_area_<quadrant>",
    ],
    parameters = ["quadrant"],
)
def get_area_by_quadrant(
        mask_resolutions_mm_width: MaskStat[float],
        mask_resolutions_mm_height: MaskStat[float],
        pixel_count_region: MaskStat[int],
        quadrant: str,
    ) -> Tuple[MaskStat[float],]:
    mask_area = pixel_count_region * mask_resolutions_mm_width * mask_resolutions_mm_height
    return mask_area,


@pyescan_metric(
    requires=[
        "stat:mask_resolutions_mm_width",
        "stat:mask_resolutions_mm_height",
        "stat:mask_pixel_count_<diameter>_mm_<quadrant>",
    ],
    returns=[
        "mask_area_<diameter>_mm_<quadrant>",
    ],
    parameters = ["diameter", "quadrant"],
)
def get_area_by_distance_quadrant(
        mask_resolutions_mm_width: MaskStat[float],
        mask_resolutions_mm_height: MaskStat[float],
        pixel_count_region: MaskStat[int],
        diameter: float,
        quadrant: str,
    ) -> Tuple[MaskStat[float],]:
    mask_area = pixel_count_region * mask_resolutions_mm_width * mask_resolutions_mm_height
    return mask_area,


@pyescan_metric(
    requires=[
        "stat:mask_resolutions_mm_width",
        "stat:mask_resolutions_mm_height",
        "meta:resolutions_mm_depth",
        "stat:mask_pixel_count_slice_<diameter>_mm",
        "stat:mask_pixel_count_projected_<diameter>_mm",
    ],
    returns=[
        "mask_volume_<diameter>_mm",
        "mask_enface_area_<diameter>_mm",
    ],
    parameters = ["diameter"],
)
def get_volume_by_distance(
        mask_resolutions_mm_width: MaskStat[float],
        mask_resolutions_mm_height: MaskStat[float],
        resolutions_mm_depth: Meta[float],
        pixel_count_region: MaskStat[int],
        columns_count: MaskStat[int],
        diameter: float,
    ) -> Tuple[MaskStat[float], MaskStat[float]]:
    volume = pixel_count_region * mask_resolutions_mm_width * mask_resolutions_mm_height * resolutions_mm_depth
    enface_area = columns_count * mask_resolutions_mm_width * resolutions_mm_depth
    return volume, enface_area


@pyescan_metric(
    requires=[
        "stat:mask_resolutions_mm_width",
        "stat:mask_resolutions_mm_height",
        "meta:resolutions_mm_depth",
        "stat:mask_pixel_count_slice_<quadrant>",
        "stat:mask_pixel_count_projected_<quadrant>",
    ],
    returns=[
        "mask_volume_<quadrant>",
        "mask_enface_area_<quadrant>",
    ],
    parameters = ["diameter", "quadrant"],
)
def get_volume_by_quadrant(
        mask_resolutions_mm_width: MaskStat[float],
        mask_resolutions_mm_height: MaskStat[float],
        resolutions_mm_depth: Meta[float],
        pixel_count_region: MaskStat[int],
        columns_count: MaskStat[int],
        quadrant: str,
    ) -> Tuple[MaskStat[float], MaskStat[float]]:
    volume = pixel_count_region * mask_resolutions_mm_width * mask_resolutions_mm_height * resolutions_mm_depth
    enface_area = columns_count * mask_resolutions_mm_width * resolutions_mm_depth
    return volume, enface_area


@pyescan_metric(
    requires=[
        "stat:mask_resolutions_mm_width",
        "stat:mask_resolutions_mm_height",
        "meta:resolutions_mm_depth",
        "stat:mask_pixel_count_slice_<diameter>_mm_<quadrant>",
        "stat:mask_pixel_count_projected_<diameter>_mm_<quadrant>",
    ],
    returns=[
        "mask_volume_<diameter>_mm_<quadrant>",
        "mask_enface_area_<diameter>_mm_<quadrant>",
    ],
    parameters = ["diameter", "quadrant"],
)
def get_volume_by_distance_quadrant(
        mask_resolutions_mm_width: MaskStat[float],
        mask_resolutions_mm_height: MaskStat[float],
        resolutions_mm_depth: Meta[float],
        pixel_count_region: MaskStat[int],
        columns_count: MaskStat[int],
        diameter: float,
        quadrant: str,
    ) -> Tuple[MaskStat[float], MaskStat[float]]:
    volume = pixel_count_region * mask_resolutions_mm_width * mask_resolutions_mm_height * resolutions_mm_depth
    enface_area = columns_count * mask_resolutions_mm_width * resolutions_mm_depth
    return volume, enface_area