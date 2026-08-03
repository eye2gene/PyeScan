"""
High-level visualisation functions for PyeScan.

Provides publication-quality overlays and animations for scans with annotations.
Uses matplotlib for static outputs and ipywidgets for interactive viewing.

Usage:
    from pyescan.visualise import show_enface, show_oct, animate_oct, scan_summary
"""
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray
from PIL import Image as PILImage
from scipy.ndimage import gaussian_filter


def scan_summary(scan) -> str:
    """
    Compact one-line summary of a scan object.

    Returns
    -------
    str
        Summary like "OCTScan(source_id=OCT-0, 25 bscans, 512x496, enface=768x768)"
    """
    parts = [type(scan).__name__]
    try:
        parts.append(f"source_id={scan.source_id}")
    except Exception:
        pass
    try:
        parts.append(f"laterality={scan.laterality}")
    except Exception:
        pass

    # OCT specific
    if hasattr(scan, 'bscans'):
        parts.append(f"{len(scan.bscans)} bscans")
        try:
            h, w = scan.bscans[0].data.shape
            parts.append(f"{w}x{h}")
        except Exception:
            pass
        if hasattr(scan, '_enface') and scan._enface is not None:
            try:
                eh, ew = scan.enface.data.shape[:2]
                parts.append(f"enface={ew}x{eh}")
            except Exception:
                pass
    elif hasattr(scan, 'shape'):
        parts.append(f"{scan.shape[1]}x{scan.shape[0]}")

    if scan.annotations:
        parts.append(f"annotations={list(scan.annotations.keys())}")

    return ", ".join(parts)


# ---------------------------------------------------------------------------
# Projection helpers
# ---------------------------------------------------------------------------

def project_annotation_to_enface(scan, annotation,
                                 thickness: bool = False,
                                 smooth: float = 0) -> NDArray:
    """
    Project an OCT volume annotation onto the enface plane.

    Parameters
    ----------
    scan : OCTScan
        The scan providing the geometry for projection.
    annotation : Annotation
        Volume annotation to project.
    thickness : bool, default False
        If True, project sum along depth axis (thickness map).
        If False, project binary presence (any along depth).
    smooth : float, default 0
        Gaussian smoothing sigma applied after projection. 0 = no smoothing.

    Returns
    -------
    NDArray
        2D projected mask in enface coordinates.
    """
    from .core.utils import _pad_array

    data = annotation.data
    if data.ndim == 2:
        return data.astype(np.float64)

    feat_data = _pad_array(data.astype(np.uint8), len(scan))

    if thickness:
        projected = scan.transform_to_enface(feat_data.sum(axis=1).astype(np.float64))
    else:
        projected = scan.transform_to_enface(feat_data.any(axis=1).astype(np.float64))

    if smooth > 0:
        projected = gaussian_filter(projected, sigma=smooth)

    return projected


def postprocess_mask(mask: NDArray, smooth: float = 0,
                     threshold: float = 0.5) -> "np.ma.MaskedArray":
    """
    Smooth and threshold a projected mask for contour display.

    Parameters
    ----------
    mask : NDArray
        2D projected mask.
    smooth : float
        Gaussian smoothing sigma. 0 = no smoothing.
    threshold : float
        Fraction of max value below which to mask out.

    Returns
    -------
    np.ma.MaskedArray
        Masked array suitable for plt.contour / plt.imshow.
    """
    if smooth > 0:
        mask = gaussian_filter(mask.astype(np.float32), sigma=smooth)
    else:
        mask = mask.astype(np.float32)

    max_val = mask.max()
    if max_val == 0:
        return np.ma.masked_all_like(mask)

    hidden = mask <= threshold * max_val
    return np.ma.masked_where(hidden, np.ones_like(mask))


# ---------------------------------------------------------------------------
# ETDRS grid
# ---------------------------------------------------------------------------

def draw_etdrs(ax, fovea_x: float, fovea_y: float,
               px_per_mm: float = 174.5,
               radii_mm: Tuple[float, ...] = (0.5, 1.5, 3.0),
               color: str = "white", lw: float = 1.0, alpha: float = 0.5,
               fovea_marker: bool = True):
    """
    Draw ETDRS grid rings and quadrant lines on a matplotlib axes.

    Parameters
    ----------
    ax : matplotlib Axes
        The axes to draw on.
    fovea_x, fovea_y : float
        Fovea position in pixel coordinates.
    px_per_mm : float
        Scale factor (pixels per mm).
    radii_mm : tuple of float
        Ring radii in mm. Default: (0.5, 1.5, 3.0) for standard ETDRS.
    color : str
        Line color.
    lw : float
        Line width.
    alpha : float
        Transparency.
    fovea_marker : bool
        Whether to draw a cross at the fovea.
    """
    from matplotlib.patches import Circle

    for r_mm in radii_mm:
        r_px = r_mm * px_per_mm
        ax.add_patch(Circle((fovea_x, fovea_y), r_px,
                            fill=False, color=color, lw=lw, alpha=alpha))

    # Quadrant lines from inner to outer ring
    r_in = radii_mm[0] * px_per_mm
    r_out = radii_mm[-1] * px_per_mm
    for ang in [45, 135, 225, 315]:
        a = np.deg2rad(ang)
        dx, dy = np.cos(a), np.sin(a)
        ax.plot([fovea_x + r_in * dx, fovea_x + r_out * dx],
                [fovea_y + r_in * dy, fovea_y + r_out * dy],
                color=color, lw=lw, alpha=alpha)

    if fovea_marker:
        ax.plot(fovea_x, fovea_y, '+', color='lime', ms=18, mew=2)


# ---------------------------------------------------------------------------
# Enface overlay
# ---------------------------------------------------------------------------

def show_enface(scan,
                annotations: Optional[Dict[str, "Annotation"]] = None,
                smooth: float = 5,
                threshold: float = 0.3,
                fovea: Optional[Tuple[float, float]] = None,
                etdrs: bool = False,
                px_per_mm: Optional[float] = None,
                colors: Optional[Dict[str, str]] = None,
                contours: bool = True,
                filled: bool = True,
                alpha: float = 0.25,
                title: Optional[str] = None,
                figsize: Tuple[float, float] = (8, 8),
                ax=None,
                show: bool = True):
    """
    Display enface image with annotation overlays, contours, and optional ETDRS grid.

    Parameters
    ----------
    scan : OCTScan or EnfaceScan
        Scan to visualise. Must have .enface or .image.
    annotations : dict, optional
        Dict of {name: Annotation} to overlay. If None, uses scan.annotations.
    smooth : float
        Gaussian smoothing for projected masks.
    threshold : float
        Mask threshold for contour extraction (fraction of max).
    fovea : (x, y) tuple, optional
        Fovea position in pixels for ETDRS grid.
    etdrs : bool
        Whether to draw ETDRS grid (requires fovea).
    px_per_mm : float, optional
        Pixel scale for ETDRS. Estimated from scan metadata if not given.
    colors : dict, optional
        Dict of {annotation_name: color_string}. Auto-generated if not given.
    contours : bool
        Whether to draw contour outlines.
    filled : bool
        Whether to draw semi-transparent filled regions.
    alpha : float
        Fill transparency.
    title : str, optional
        Plot title.
    figsize : tuple
        Figure size.
    ax : matplotlib Axes, optional
        Existing axes to draw on. If None, creates a new figure.
    show : bool
        Whether to call plt.show().

    Returns
    -------
    matplotlib Axes
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.lines import Line2D

    # Get enface image
    if hasattr(scan, 'enface') and scan.enface is not None:
        enface_img = np.array(scan.enface.image)
    elif hasattr(scan, 'image'):
        enface_img = np.array(scan.image)
    else:
        raise ValueError("Scan has no enface or image attribute")

    # Use scan's annotations if none provided
    if annotations is None:
        annotations = scan.annotations if scan.annotations else {}

    # Default colors
    default_colors = ['lime', 'red', 'cyan', 'magenta', 'yellow', 'orange']
    if colors is None:
        colors = {name: default_colors[i % len(default_colors)]
                  for i, name in enumerate(annotations.keys())}

    # Create figure
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(enface_img, cmap='gray')

    # Overlay each annotation
    legend_handles = []
    for name, annotation in annotations.items():
        color = colors.get(name, 'lime')
        proj = project_annotation_to_enface(scan, annotation, smooth=smooth)
        masked = postprocess_mask(proj, threshold=threshold)

        if filled:
            ax.imshow(masked, cmap=ListedColormap([color]), alpha=alpha)
        if contours:
            try:
                ax.contour(masked.mask, levels=[0.5], colors=[color], linewidths=2.5)
            except ValueError:
                pass  # empty mask

        legend_handles.append(Line2D([0], [0], color=color, lw=2.5, label=name))

    # ETDRS grid
    if etdrs and fovea is not None:
        scale = px_per_mm
        if scale is None:
            # Try to estimate from scan metadata
            try:
                w = scan.enface.image.width
                fov_mm = scan.metadata.dimensions_mm_width
                scale = w / fov_mm
            except Exception:
                scale = enface_img.shape[1] / 6.0  # rough fallback for 6mm scan
        draw_etdrs(ax, fovea[0], fovea[1], px_per_mm=scale)

    if legend_handles:
        ax.legend(handles=legend_handles, loc='upper right', framealpha=0.7)

    ax.axis('off')
    if title:
        ax.set_title(title, fontsize=10, pad=10)

    if show:
        plt.tight_layout()
        plt.show()

    return ax


# ---------------------------------------------------------------------------
# OCT B-scan overlay
# ---------------------------------------------------------------------------

def show_oct(scan,
             bscan_index: Optional[int] = None,
             annotations: Optional[Dict[str, "Annotation"]] = None,
             colors: Optional[Dict[str, str]] = None,
             alpha: float = 0.4,
             title: Optional[str] = None,
             figsize: Tuple[float, float] = (12, 4),
             show_enface: bool = True,
             ax=None,
             show: bool = True):
    """
    Display a B-scan (or middle B-scan) with annotation mask overlays.

    Optionally shows the enface with the B-scan position marked.

    Parameters
    ----------
    scan : OCTScan
        OCT scan to visualise.
    bscan_index : int, optional
        Which B-scan to show. Default: middle.
    annotations : dict, optional
        Annotations to overlay. If None, uses scan.annotations.
    colors : dict, optional
        Color map for annotations.
    alpha : float
        Overlay transparency.
    title : str, optional
        Plot title.
    figsize : tuple
        Figure size.
    show_enface : bool
        Whether to show enface alongside with B-scan position marked.
    ax : Axes, optional
        Existing axes (only used if show_enface=False).
    show : bool
        Whether to call plt.show().
    """
    import matplotlib.pyplot as plt
    from .core.visualisation import draw_bscan_lines, overlay_masks, generate_distinct_colors

    if bscan_index is None:
        bscan_index = len(scan.bscans) // 2

    if annotations is None:
        annotations = scan.annotations if scan.annotations else {}

    default_colors = [(0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255)]
    if colors is None:
        color_list = [default_colors[i % len(default_colors)] for i in range(len(annotations))]
    else:
        color_list = [colors.get(name, default_colors[i % len(default_colors)])
                      for i, name in enumerate(annotations.keys())]

    # Get the B-scan image
    bscan_img = scan.bscans[bscan_index].image

    # Get masks for this B-scan
    masks = []
    for ann in annotations.values():
        if ann.is_volume and bscan_index < len(ann.slices):
            slice_data = ann.slices[bscan_index].data
            if slice_data is not None:
                masks.append(PILImage.fromarray(slice_data.astype(np.uint8)))
            else:
                masks.append(None)
        else:
            masks.append(None)

    if show_enface and hasattr(scan, 'enface') and scan._enface is not None:
        fig, (ax_enface, ax_bscan) = plt.subplots(1, 2, figsize=figsize)

        # Enface with B-scan position
        try:
            locations = scan.get_bscan_enface_locations()
            enface_marked = draw_bscan_lines(scan.enface.image, locations, bscan_index)
            ax_enface.imshow(enface_marked)
        except (AttributeError, TypeError):
            ax_enface.imshow(scan.enface.image, cmap='gray')
        ax_enface.axis('off')
        ax_enface.set_title(f"Enface (B-scan {bscan_index})", fontsize=9)

        # B-scan with overlay
        annotated = overlay_masks(bscan_img, masks, colors=color_list,
                                  feature_names=list(annotations.keys()), alpha=alpha)
        ax_bscan.imshow(annotated)
        ax_bscan.axis('off')
        ax_bscan.set_title(f"B-scan {bscan_index}", fontsize=9)
    else:
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        annotated = overlay_masks(bscan_img, masks, colors=color_list,
                                  feature_names=list(annotations.keys()), alpha=alpha)
        ax.imshow(annotated)
        ax.axis('off')

    if title:
        plt.suptitle(title, fontsize=11)
    if show:
        plt.tight_layout()
        plt.show()


# ---------------------------------------------------------------------------
# Animation (GIF export)
# ---------------------------------------------------------------------------

def animate_oct(scan,
                output_path: Optional[str] = None,
                annotations: Optional[Dict[str, "Annotation"]] = None,
                include_enface: bool = True,
                include_raw: bool = True,
                fps: int = 5,
                alpha: float = 0.4,
                progress: bool = True) -> Optional[List[PILImage.Image]]:
    """
    Create an animated GIF of an OCT volume with annotations.

    Shows annotated enface (with B-scan position) alongside annotated B-scans,
    optionally with raw versions underneath.

    Parameters
    ----------
    scan : OCTScan
        OCT scan to animate.
    output_path : str, optional
        Path to save the GIF. If None, returns frame list without saving.
    annotations : dict, optional
        Annotations to overlay. If None, uses scan.annotations.
    include_enface : bool
        Whether to include enface panel.
    include_raw : bool
        Whether to include raw (un-annotated) panels below.
    fps : int
        Frames per second for the GIF.
    alpha : float
        Overlay transparency.
    progress : bool
        Whether to show progress bar.

    Returns
    -------
    list[PIL.Image] or None
        List of frames if output_path is None.
    """
    from .core.visualisation import draw_bscan_lines, overlay_masks, generate_distinct_colors

    if annotations is None:
        annotations = scan.annotations if scan.annotations else {}

    # Attach annotations to scan temporarily for _annotated_enface
    for name, ann in annotations.items():
        if name not in scan.annotations:
            scan.add_annotation(name, ann)

    has_enface = include_enface and hasattr(scan, '_enface') and scan._enface is not None

    try:
        bscan_locations = scan.get_bscan_enface_locations() if has_enface else None
    except (AttributeError, TypeError):
        bscan_locations = None
        has_enface = False

    if has_enface:
        enface_base = scan._annotated_enface(contours=True, heatmap=False) if annotations else scan.enface.image
        enface_raw = scan.enface.image
    
    n_colors = len(annotations)
    default_colors = generate_distinct_colors(n_colors) if n_colors else []
    color_list = [ann.color or default_colors[i] for i, ann in enumerate(annotations.values())]

    frames = []
    iterator = range(len(scan.bscans))
    if progress:
        try:
            import tqdm
            iterator = tqdm.tqdm(iterator, desc="Rendering frames", leave=False)
        except ImportError:
            pass

    for i in iterator:
        panels = []
        panel_heights = []

        # Annotated B-scan
        bscan_img = scan.bscans[i].image
        masks = []
        for ann in annotations.values():
            if ann.is_volume and i < len(ann.slices):
                d = ann.slices[i].data
                masks.append(PILImage.fromarray(d.astype(np.uint8)) if d is not None else None)
            else:
                masks.append(None)
        annotated_bscan = overlay_masks(bscan_img, masks, colors=color_list, alpha=alpha)

        if has_enface:
            # Enface with B-scan line
            annotated_enface = draw_bscan_lines(enface_base, [bscan_locations[i]], 0)
            # Normalise heights
            target_h = min(annotated_enface.height, annotated_bscan.height) // (2 if include_raw else 1)
            enface_resized = annotated_enface.resize(
                (int(annotated_enface.width * target_h / annotated_enface.height), target_h))
            bscan_resized = annotated_bscan.resize(
                (int(annotated_bscan.width * target_h / annotated_bscan.height), target_h))
            panels.append((enface_resized, bscan_resized))
            panel_heights.append(target_h)

            if include_raw:
                raw_enface = draw_bscan_lines(enface_raw, [bscan_locations[i]], 0)
                raw_bscan = scan.bscans[i].image
                raw_enface_r = raw_enface.resize(
                    (int(raw_enface.width * target_h / raw_enface.height), target_h))
                raw_bscan_r = raw_bscan.convert('RGB').resize(
                    (int(raw_bscan.width * target_h / raw_bscan.height), target_h))
                panels.append((raw_enface_r, raw_bscan_r))
                panel_heights.append(target_h)
        else:
            target_h = annotated_bscan.height
            bscan_resized = annotated_bscan
            panels.append((bscan_resized,))
            panel_heights.append(target_h)
            if include_raw:
                raw_bscan = scan.bscans[i].image.convert('RGB')
                panels.append((raw_bscan,))
                panel_heights.append(target_h)

        # Combine panels into single frame
        total_h = sum(panel_heights)
        max_w = max(sum(p.width for p in row) for row in panels)
        frame = PILImage.new('RGB', (max_w, total_h))
        y_offset = 0
        for row in panels:
            x_offset = 0
            for panel in row:
                frame.paste(panel, (x_offset, y_offset))
                x_offset += panel.width
            y_offset += row[0].height
        frames.append(frame)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        duration = int(1000 / fps)
        frames[0].save(output_path, save_all=True, append_images=frames[1:],
                       duration=duration, loop=0)

    return frames


# ---------------------------------------------------------------------------
# Save annotated volume as images
# ---------------------------------------------------------------------------

def save_annotated_volume(scan,
                          output_dir: str,
                          annotations: Optional[Dict[str, "Annotation"]] = None,
                          alpha: float = 0.4,
                          format: str = "png",
                          prefix: Optional[str] = None,
                          as_animation: bool = False,
                          fps: int = 5,
                          include_enface: bool = True,
                          include_raw: bool = False,
                          progress: bool = True) -> str:
    """
    Save an annotated OCT volume as a series of images or a single animation.

    Parameters
    ----------
    scan : OCTScan
        OCT scan to render.
    output_dir : str
        Output directory. Created if it doesn't exist.
    annotations : dict, optional
        Annotations to overlay. If None, uses scan.annotations.
    alpha : float
        Overlay transparency.
    format : str
        Image format for individual frames ("png", "jpg", etc.).
    prefix : str, optional
        Filename prefix. Defaults to scan.source_id if available.
    as_animation : bool, default False
        If True, saves a single GIF instead of individual frames.
    fps : int
        Frames per second (only used if as_animation=True).
    include_enface : bool
        Whether to include enface panel alongside each B-scan.
    include_raw : bool
        Whether to include raw (un-annotated) panels.
    progress : bool
        Whether to show progress bar.

    Returns
    -------
    str
        Path to the output directory (or GIF file if as_animation=True).
    """
    if as_animation:
        # Use animate_oct and save as GIF
        name = prefix or getattr(scan, 'source_id', 'scan')
        gif_path = str(Path(output_dir) / f"{name}.gif")
        animate_oct(scan, output_path=gif_path, annotations=annotations,
                    include_enface=include_enface, include_raw=include_raw,
                    fps=fps, alpha=alpha, progress=progress)
        return gif_path

    # Save as individual frames
    from .core.visualisation import overlay_masks, generate_distinct_colors

    if annotations is None:
        annotations = scan.annotations if scan.annotations else {}

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    name = prefix or getattr(scan, 'source_id', 'scan')

    n_colors = len(annotations)
    default_colors = generate_distinct_colors(n_colors) if n_colors else []
    color_list = [ann.color or default_colors[i] for i, ann in enumerate(annotations.values())]

    iterator = range(len(scan.bscans))
    if progress:
        try:
            import tqdm
            iterator = tqdm.tqdm(iterator, desc="Saving frames", leave=False)
        except ImportError:
            pass

    for i in iterator:
        bscan_img = scan.bscans[i].image
        masks = []
        for ann in annotations.values():
            if ann.is_volume and i < len(ann.slices):
                d = ann.slices[i].data
                masks.append(PILImage.fromarray(d.astype(np.uint8)) if d is not None else None)
            else:
                masks.append(None)

        annotated = overlay_masks(bscan_img, masks, colors=color_list,
                                  feature_names=list(annotations.keys()), alpha=alpha)
        filename = f"{name}_{i}.{format}"
        annotated.save(out / filename)

    return str(out)
