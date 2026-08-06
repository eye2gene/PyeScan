# PyeScan Metrics System

The metrics system provides a dependency-driven computation framework for
extracting quantitative measurements from retinal scans and annotations.

## Core Concepts

**Metric**: A function that takes typed inputs and returns typed outputs.
The `@pyescan_metric` decorator registers it into the global registry.

**MetricProcessor**: Resolves dependency chains automatically. When you
request a statistic, the processor finds the metric that produces it, checks
what inputs that metric requires, and recursively resolves those too.

**Registry**: The global `PYESCAN_GLOBAL_METRICS` registry holds all
registered metrics. You can also create local registries for testing.

## Type Annotations

Inputs and outputs are annotated with semantic types that control how the
processor maps values:

| Annotation   | Meaning                                          |
|-------------|--------------------------------------------------|
| `Meta[T]`   | Scan-level metadata (from DataFrame or metadata) |
| `Spec[T]`   | Intermediate array (e.g. loaded mask/image)      |
| `MaskStat[T]` | Per-mask statistic (output)                    |
| `ImgStat[T]` | Per-image statistic (output)                    |
| `Pred[T]`   | Model prediction (e.g. fovea location)           |

## Naming Convention

The processor matches inputs to outputs by **name**. Variable names in the
function signature become the stat names in the dependency graph.

For parameterised metrics, use `<param>` placeholders in `returns`:

```python
@pyescan_metric(
    returns=["mask_pixel_count_<diameter>_mm"],
    parameters=["diameter"],
)
def get_pixel_count_by_distance(mask, distance_mask, diameter):
    ...
```

When you request `mask_pixel_count_4.0_mm`, the processor extracts
`diameter=4.0` and passes it to the function.

## Writing a Custom Metric

```python
from typing import Tuple
import numpy as np
from pyescan.metrics.registry import Meta, MaskStat, Spec, pyescan_metric


@pyescan_metric()
def get_lesion_count(mask: Spec[np.array]) -> Tuple[MaskStat[int]]:
    """Count connected components in a binary mask."""
    from scipy.ndimage import label
    labeled, n_features = label(mask)
    lesion_count = n_features
    return lesion_count,
```

Key rules:
1. Return a **tuple** (even for single values) — the decorator extracts
   variable names from the return statement via AST.
2. Name return variables descriptively — these become stat names.
3. Use the type annotations to tell the processor where each input comes from.

## Running Metrics on a DataFrame

```python
from pyescan.metrics import run_on_dataframe

result = run_on_dataframe(
    df,
    stat_name=["mask_area", "mask_pixel_count"],
    col_mapping={"file_path_mask": "your_mask_column"},
    auto_merge=True,
)
```

The `col_mapping` dict maps internal metric input names to your DataFrame
column names. The processor will load masks, compute intermediates, and
return the requested stats.

## CLI Usage

```bash
run_metric mask_area,mask_pixel_count input.csv output.csv \
    --mapping file_path_mask=mask_col,scan_width_px=width
```
