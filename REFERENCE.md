# PyeScan Quick Reference

## Installation

```bash
pip install -e .            # core
pip install -e ".[dev]"     # + pytest, ruff
pip install -e ".[interactive]"  # + ipywidgets, IPython
```

## Loading Scans

```python
from pyescan import load_record_from_CE, load_record_from_df, load_records_from_df

# From CrystalEye JSON export folder
scans = load_record_from_CE("/path/to/sdb_folder/")

# From DataFrame (single record)
scans = load_record_from_df(df, column_headings={"image_location": "file_path"})

# From DataFrame (multiple records, grouped)
scans = load_records_from_df(df, identifier_columns=["pat", "sdb"])

# With validation
scans = load_record_from_df(df, identity_col="volume_id", validate=True)
scans = load_record_from_df(df, validate=False)  # bypass checks
```

### Required DataFrame columns

| Internal name  | Default column   | Required for |
| -------------- | ---------------- | ------------ |
| source_id      | source_id        | always       |
| group          | group            | always       |
| modality       | modality         | always       |
| image_location | file_path        | always       |
| n_images       | number_of_images | OCT          |
| bscan_index    | bscan_index      | OCT          |

Use `column_headings={"internal_name": "your_col"}` to remap.

## Scan Objects

```python
scan = scans[0]  # OCTScan, FAFScan, or IRScan

# OCTScan
scan.enface  # EnfaceScan (IR image)
scan.bscans  # BScanArray (indexable, sliceable)
scan.bscans[0].data  # numpy array (H x W)
scan.data  # full volume as numpy array (N x H x W)
len(scan)  # number of B-scans

# Any scan
scan.image  # BaseImage
scan.data  # numpy array
scan.metadata  # MetadataView (dynamic attribute access)
scan.source_id  # str
scan.laterality  # str
scan.annotations  # dict[str, Annotation]

# Preload/unload for memory management
scan.preload()
scan.unload()
```

## Annotations

```python
from pyescan import Annotation, AnnotationSlice, AnnotationVolume

# Create from numpy array
ann = Annotation(slices=mask_3d, feature_name="GA", source_id="OCT-0")
ann = Annotation(slices=mask_2d, feature_name="GA")  # enface (single slice)

# Create from AnnotationSlices
slices = [AnnotationSlice(raster=arr) for arr in arrays]
ann = Annotation(slices=slices, feature_name="drusen")

# AnnotationSlice can hold mixed types
s = AnnotationSlice(raster=mask, contours=[polyline], points=landmarks)

# Properties
ann.data  # NDArray (2D for enface, 3D for volume)
ann.slices  # AnnotationVolume (indexable)
ann.is_volume  # True if multi-slice
ann.is_enface  # True if single-slice
ann.feature_name  # str
ann.source_id  # str
ann.color  # tuple (RGB)
ann[5]  # AnnotationSlice at index 5
```

### Loading annotations

```python
from pyescan import load_annotation_from_df, load_annotation_from_folder

# From DataFrame
ann = load_annotation_from_df(df, feature_col="feature", allow_gaps=True)
# Returns dict[str, AnnotationOCT] if feature_col given, else single AnnotationOCT

# From folder
anns = load_annotation_from_folder(
    "/path/to/masks/", folder_structure="{feature}/{source_id}_{bscan_index:\\d+}.png"
)
```

### Attaching to scans

```python
scan.add_annotation("GA", ann, color=(255, 0, 0))
scan.add_annotations({"GA": ann_ga, "drusen": ann_drusen})
```

### Projection (on OCTScan)

```python
# Volume -> enface
enface_ann = scan.annotation_to_enface(volume_ann)

# Enface -> B-scans (verticalised)
bscan_ann = scan.annotation_to_bscans(enface_ann)
```

### Saving

```python
from pyescan import save_annotation, save_annotations

# Single annotation -> feature/source_id_index.png
save_annotation(ann, "/output/dir/")

# All scan annotations
save_annotations(scan.annotations, "/output/dir/", source_id="OCT-0")
```

## Visualisation

```python
from pyescan.visualise import (
    scan_summary,
    show_enface,
    show_oct,
    animate_oct,
    project_annotation_to_enface,
    draw_etdrs,
)

# Quick summary
print(scan_summary(scan))
# "OCTScan, source_id=OCT-0, 25 bscans, 512x496, enface=768x768, annotations=['GA']"

# Enface overlay with contours
show_enface(scan, smooth=5, threshold=0.3, contours=True, filled=True)

# With ETDRS grid
show_enface(scan, fovea=(384, 384), etdrs=True, px_per_mm=87.3)

# B-scan overlay (shows enface position + annotated B-scan)
show_oct(scan, bscan_index=12)

# GIF animation
animate_oct(scan, output_path="output.gif", fps=5)
frames = animate_oct(scan)  # returns list of PIL frames without saving

# Low-level: project annotation to enface as numpy array
proj = project_annotation_to_enface(scan, ann, smooth=5, thickness=False)
```

## File Discovery

```python
from pyescan.tools.dataset_utils import summarise_dataset
from pyescan.tools.ce_metadata import parse_ce_metadata_json, scrape_ce_export

# Find files matching a pattern (glob-accelerated)
df = summarise_dataset(
    "/data/", structure="{pat}/{sdb}/{source_id}_{bscan_index:\\d+}.png"
)

# Parse a single CE metadata JSON
records = parse_ce_metadata_json("/path/to/metadata.json")

# Full export scrape with validation
df = scrape_ce_export(
    "/data/export/",
    file_structure="{pat}/{sdb}/metadata.json",
    on_missing="warn",  # or "raise" / "ignore"
    on_duplicate="warn",
)
```

## Metrics

```python
from pyescan.metrics import run_on_dataframe, pyescan_metric, PYESCAN_GLOBAL_METRICS

# Run metrics on a DataFrame
result = run_on_dataframe(
    df,
    stat_name=["mask_pixel_count", "mask_area"],
    col_mapping={"file_path_mask": "mask_col"},
    auto_merge=True,
)

# Define a custom metric
from pyescan.metrics.registry import Meta, MaskStat, Spec


@pyescan_metric()
def my_metric(mask: Spec[np.array], scale: Meta[float]) -> Tuple[MaskStat[float]]:
    result = mask.sum() * scale
    return (result,)
```

## Project Structure

```txt
pyescan/
├── __init__.py          # Public API exports
├── CELoader.py          # CrystalEye loading (JSON + DataFrame)
├── annotation_loader.py # Load annotations from df/folder
├── annotation_io.py     # Save annotations
├── visualise.py         # High-level visualisation (show_enface, animate_oct, etc.)
├── core/
│   ├── annotation.py    # Annotation, AnnotationSlice, AnnotationVolume
│   ├── image.py         # LazyImage, ImageVolume
│   ├── metadata.py      # MetadataRecord, MetadataView, parsers
│   ├── scan.py          # BaseScan, SingleImageScan
│   ├── scan_oct.py      # OCTScan, BScan, BScanArray
│   ├── scan_enface.py   # EnfaceScan, FAFScan, IRScan
│   ├── scan_building.py # Factory functions from metadata
│   ├── utils.py         # ArrayView, padding helpers
│   └── visualisation.py # Widget-based display (ipywidgets)
├── metrics/
│   ├── metric.py        # Metric class
│   ├── registry.py      # @pyescan_metric decorator, type annotations
│   ├── processor.py     # Dependency resolution engine
│   ├── metrics.py       # Built-in metric functions
│   └── helpers.py       # run_on_dataframe
└── tools/
    ├── dataset_utils.py # summarise_dataset, get_ce_export_summary
    ├── file_discovery.py# Glob-based file finding (internal)
    ├── ce_metadata.py   # parse_ce_metadata_json, scrape_ce_export
    └── cli.py           # Command-line tools
```
