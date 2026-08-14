"""High-level visualisation functions for PyeScan.

Provides publication-quality overlays and animations for scans with annotations.
Uses matplotlib for static outputs and ipywidgets for interactive viewing.

Usage:
    from pyescan.visualise import show_enface, show_oct, animate_oct, scan_summary
"""

import contextlib
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PIL import Image as PILImage


def scan_summary(scan) -> str:
    """Compact one-line summary of a scan object.

    Returns:
    -------
    str
        Summary like "OCTScan(source_id=OCT-0, 25 bscans, 512x496, enface=768x768)"
    """
    parts = [type(scan).__name__]
    with contextlib.suppress(Exception):
        parts.append(f"source_id={scan.source_id}")
    with contextlib.suppress(Exception):
        parts.append(f"laterality={scan.laterality}")

    # OCT specific
    if hasattr(scan, "bscans"):
        parts.append(f"{len(scan.bscans)} bscans")
        try:
            h, w = scan.bscans[0].data.shape
            parts.append(f"{w}x{h}")
        except Exception:
            pass
        if hasattr(scan, "_enface") and scan._enface is not None:
            try:
                eh, ew = scan.enface.data.shape[:2]
                parts.append(f"enface={ew}x{eh}")
            except Exception:
                pass
    elif hasattr(scan, "shape"):
        parts.append(f"{scan.shape[1]}x{scan.shape[0]}")

    if scan.annotations:
        parts.append(f"annotations={list(scan.annotations.keys())}")

    return ", ".join(parts)


# ---------------------------------------------------------------------------
# Mask postprocessing for display
# ---------------------------------------------------------------------------


def postprocess_mask(
    mask: NDArray, smooth: float = 0, threshold: float = 0.5
) -> "np.ma.MaskedArray":
    """Smooth and threshold a projected mask for contour display.

    Parameters
    ----------
    mask : NDArray
        2D projected mask.
    smooth : float
        Gaussian smoothing sigma. 0 = no smoothing.
    threshold : float
        Fraction of max value below which to mask out.

    Returns:
    -------
    np.ma.MaskedArray
        Masked array suitable for plt.contour / plt.imshow.
    """
    if smooth > 0:
        try:
            from scipy.ndimage import gaussian_filter
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Mask smoothing requires the 'metrics' extra; "
                "install it with `pip install 'pyescan[metrics]'`."
            ) from exc
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


def draw_etdrs(
    ax,
    fovea_x: float,
    fovea_y: float,
    px_per_mm: float = 174.5,
    radii_mm: tuple[float, ...] = (0.5, 1.5, 3.0),
    color: str = "white",
    lw: float = 1.0,
    alpha: float = 0.5,
    fovea_marker: bool = True,
):
    """Draw ETDRS grid rings and quadrant lines on a matplotlib axes.

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
        ax.add_patch(
            Circle(
                (fovea_x, fovea_y), r_px, fill=False, color=color, lw=lw, alpha=alpha
            )
        )

    # Quadrant lines from inner to outer ring
    r_in = radii_mm[0] * px_per_mm
    r_out = radii_mm[-1] * px_per_mm
    for ang in [45, 135, 225, 315]:
        a = np.deg2rad(ang)
        dx, dy = np.cos(a), np.sin(a)
        ax.plot(
            [fovea_x + r_in * dx, fovea_x + r_out * dx],
            [fovea_y + r_in * dy, fovea_y + r_out * dy],
            color=color,
            lw=lw,
            alpha=alpha,
        )

    if fovea_marker:
        ax.plot(fovea_x, fovea_y, "+", color="lime", ms=18, mew=2)


# ---------------------------------------------------------------------------
# Enface overlay
# ---------------------------------------------------------------------------


def show_enface(
    scan,
    annotations: dict[str, "Annotation"] | None = None,
    smooth: float = 5,
    threshold: float = 0.3,
    fovea: tuple[float, float] | None = None,
    etdrs: bool = False,
    px_per_mm: float | None = None,
    colors: dict[str, str] | None = None,
    contours: bool = True,
    filled: bool = True,
    alpha: float = 0.25,
    title: str | None = None,
    figsize: tuple[float, float] = (8, 8),
    ax=None,
    show: bool = True,
):
    """Display enface image with annotation overlays, contours, and optional ETDRS grid.

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

    Returns:
    -------
    matplotlib Axes
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.lines import Line2D

    # Get enface image
    if hasattr(scan, "enface") and scan.enface is not None:
        enface_img = np.array(scan.enface.image)
    elif hasattr(scan, "image"):
        enface_img = np.array(scan.image)
    else:
        raise ValueError("Scan has no enface or image attribute")

    # Use scan's annotations if none provided
    if annotations is None:
        annotations = scan.annotations if scan.annotations else {}

    # Default colors
    default_colors = ["lime", "red", "cyan", "magenta", "yellow", "orange"]
    if colors is None:
        colors = {
            name: default_colors[i % len(default_colors)]
            for i, name in enumerate(annotations.keys())
        }

    # Create figure
    if ax is None:
        _fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(enface_img, cmap="gray")

    # Overlay each annotation
    legend_handles = []
    for name, annotation in annotations.items():
        color = colors.get(name, "lime")
        # Project to enface — scan owns the geometry, visualise owns the reduction
        enface_ann = scan.annotation_to_enface(annotation)
        proj = enface_ann.data.astype(np.float32)
        # Smoothing and thresholding are display concerns
        masked = postprocess_mask(proj, smooth=smooth, threshold=threshold)

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
        ax.legend(handles=legend_handles, loc="upper right", framealpha=0.7)

    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=10, pad=10)

    if show:
        plt.tight_layout()
        plt.show()

    return ax


# ---------------------------------------------------------------------------
# OCT B-scan overlay
# ---------------------------------------------------------------------------


def show_oct(scan, bscan_index=None, **kwargs):
    """Display a single annotated B-scan frame. Shortcut for animate_oct with one frame.

    See animate_oct for full parameter documentation.
    Additional parameter: show (bool) — whether to call plt.show().
    """
    import matplotlib.pyplot as plt

    if bscan_index is None:
        bscan_index = len(scan.bscans) // 2

    show = kwargs.pop("show", True)
    figsize = kwargs.pop("figsize", (12, 4))
    title = kwargs.pop("title", None)

    # Render just this one frame
    frames = animate_oct(scan, bscan_indices=[bscan_index], progress=False, **kwargs)
    if not frames:
        return

    _fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(frames[0])
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=11)
    if show:
        plt.tight_layout()
        plt.show()
    return ax


# ---------------------------------------------------------------------------
# Animation (GIF export)
# ---------------------------------------------------------------------------


def animate_oct(
    scan,
    output_path: str | None = None,
    output_dir: str | None = None,
    bscan_indices: list[int] | None = None,
    annotations: dict[str, "Annotation"] | None = None,
    include_enface: bool = True,
    include_raw: bool = False,
    fps: int = 5,
    alpha: float = 0.4,
    format: str = "png",
    prefix: str | None = None,
    progress: bool = True,
) -> list[PILImage.Image] | None:
    """Render an annotated OCT volume as an animation or image series.

    Parameters
    ----------
    scan : OCTScan
        OCT scan to render.
    output_path : str, optional
        Path to save as GIF animation. If None and output_dir is None,
        returns frames without saving.
    output_dir : str, optional
        Directory to save individual frames (one PNG per B-scan).
        If given, saves frames as {prefix}_{index}.{format} instead of GIF.
    bscan_indices : list[int], optional
        Which B-scans to render. Defaults to all.
    annotations : dict, optional
        Annotations to overlay. If None, uses scan.annotations.
    include_enface : bool, default True
        Whether to include enface panel alongside each B-scan.
    include_raw : bool, default False
        Whether to include raw (un-annotated) panels below.
    fps : int, default 5
        Frames per second for GIF output.
    alpha : float, default 0.4
        Overlay transparency.
    format : str, default "png"
        Image format for individual frames (only used with output_dir).
    prefix : str, optional
        Filename prefix for individual frames. Defaults to scan.source_id.
    progress : bool, default True
        Whether to show progress bar.

    Returns:
    -------
    list[PIL.Image]
        List of rendered frames.
    """
    from .core.visualisation import (
        draw_bscan_lines,
        generate_distinct_colors,
        overlay_masks,
    )

    if annotations is None:
        annotations = scan.annotations if scan.annotations else {}

    # Attach annotations to scan temporarily for _annotated_enface
    for name, ann in annotations.items():
        if name not in scan.annotations:
            scan.add_annotation(name, ann)

    has_enface = (
        include_enface and hasattr(scan, "_enface") and scan._enface is not None
    )

    try:
        bscan_locations = scan.get_bscan_enface_locations() if has_enface else None
    except (AttributeError, TypeError):
        bscan_locations = None
        has_enface = False

    if has_enface:
        enface_base = (
            scan._annotated_enface(contours=True, heatmap=False)
            if annotations
            else scan.enface.image
        )
        enface_raw = scan.enface.image

    n_colors = len(annotations)
    default_colors = generate_distinct_colors(n_colors) if n_colors else []
    color_list = [
        ann.color or default_colors[i] for i, ann in enumerate(annotations.values())
    ]

    frames = []
    indices = (
        bscan_indices if bscan_indices is not None else list(range(len(scan.bscans)))
    )
    iterator = indices
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
                masks.append(
                    PILImage.fromarray(d.astype(np.uint8)) if d is not None else None
                )
            else:
                masks.append(None)
        annotated_bscan = overlay_masks(
            bscan_img,
            masks,
            colors=color_list,
            feature_names=list(annotations.keys()),
            alpha=alpha,
        )

        if has_enface:
            annotated_enface = draw_bscan_lines(enface_base, [bscan_locations[i]], 0)
            target_h = min(annotated_enface.height, annotated_bscan.height) // (
                2 if include_raw else 1
            )
            enface_resized = annotated_enface.resize(
                (
                    int(annotated_enface.width * target_h / annotated_enface.height),
                    target_h,
                )
            )
            bscan_resized = annotated_bscan.resize(
                (
                    int(annotated_bscan.width * target_h / annotated_bscan.height),
                    target_h,
                )
            )
            panels.append((enface_resized, bscan_resized))
            panel_heights.append(target_h)

            if include_raw:
                raw_enface = draw_bscan_lines(enface_raw, [bscan_locations[i]], 0)
                raw_bscan = scan.bscans[i].image
                raw_enface_r = raw_enface.resize(
                    (int(raw_enface.width * target_h / raw_enface.height), target_h)
                )
                raw_bscan_r = raw_bscan.convert("RGB").resize(
                    (int(raw_bscan.width * target_h / raw_bscan.height), target_h)
                )
                panels.append((raw_enface_r, raw_bscan_r))
                panel_heights.append(target_h)
        else:
            target_h = annotated_bscan.height
            bscan_resized = annotated_bscan
            panels.append((bscan_resized,))
            panel_heights.append(target_h)
            if include_raw:
                raw_bscan = scan.bscans[i].image.convert("RGB")
                panels.append((raw_bscan,))
                panel_heights.append(target_h)

        # Combine panels into single frame
        total_h = sum(panel_heights)
        max_w = max(sum(p.width for p in row) for row in panels)
        frame = PILImage.new("RGB", (max_w, total_h))
        y_offset = 0
        for row in panels:
            x_offset = 0
            for panel in row:
                frame.paste(panel, (x_offset, y_offset))
                x_offset += panel.width
            y_offset += row[0].height
        frames.append(frame)

    # Save output
    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        name = prefix or getattr(scan, "source_id", "scan")
        for i, frame in enumerate(frames):
            frame.save(out / f"{name}_{i}.{format}")
    elif output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        duration = int(1000 / fps)
        frames[0].save(
            output_path,
            save_all=True,
            append_images=frames[1:],
            duration=duration,
            loop=0,
        )

    return frames
