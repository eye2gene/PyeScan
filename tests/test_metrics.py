"""
Tests for the metrics system: registry, processor, and metric functions.
"""
import pytest
import numpy as np
import pandas as pd
from typing import Tuple

from pyescan.metrics.registry import (
    MetricRegistry,
    pyescan_metric,
    PYESCAN_GLOBAL_METRICS,
    Meta,
    MaskStat,
    Spec,
    Pred,
)
from pyescan.metrics.metric import Metric
from pyescan.metrics.processor import MetricProcessor
from pyescan.metrics.helpers import run_on_dataframe, PandasRowWrapperHelper


# ============================================================================
# Module-level metric definitions for testing
# (pyescan_metric uses inspect.getsource + ast.parse, which requires
#  the function to be defined at module/top level, not inside a method)
# ============================================================================

_test_registry = MetricRegistry()


@pyescan_metric(registry=_test_registry)
def my_test_metric(x: Meta[float]) -> Tuple[MaskStat[float]]:
    result = x * 2
    return result,


_callable_registry = MetricRegistry()


@pyescan_metric(registry=_callable_registry)
def double_it(value: Meta[float]) -> Tuple[MaskStat[float]]:
    doubled = value * 2
    return doubled,


_req_registry = MetricRegistry()


@pyescan_metric(registry=_req_registry)
def needs_inputs(a: Meta[float], b: MaskStat[float]) -> Tuple[MaskStat[float]]:
    c = a + b
    return c,


_returns_registry = MetricRegistry()


@pyescan_metric(registry=_returns_registry)
def produces_output(x: Meta[float]) -> Tuple[MaskStat[float], MaskStat[float]]:
    out_a = x * 2
    out_b = x * 3
    return out_a, out_b


_param_registry = MetricRegistry()


@pyescan_metric(
    registry=_param_registry,
    returns=["result_<multiplier>"],
    parameters=["multiplier"],
)
def multiply(x: Meta[float], multiplier: float) -> Tuple[MaskStat[float]]:
    result = x * multiplier
    return result,


# Chain metrics: raw_value -> base_value -> derived_value
_chain_registry = MetricRegistry()


@pyescan_metric(registry=_chain_registry)
def get_base(raw_value: Meta[float]) -> Tuple[MaskStat[float]]:
    base_value = raw_value * 2
    return base_value,


@pyescan_metric(registry=_chain_registry)
def get_derived(base_value: MaskStat[float]) -> Tuple[MaskStat[float]]:
    derived_value = base_value + 10
    return derived_value,


# Orphan metric (dependency cannot be resolved)
_orphan_registry = MetricRegistry()


@pyescan_metric(registry=_orphan_registry)
def orphan_metric(nonexistent_input: Meta[float]) -> Tuple[MaskStat[float]]:
    output = nonexistent_input
    return output,


# Simple metrics for run_on_dataframe tests
_df_registry = MetricRegistry()


@pyescan_metric(registry=_df_registry)
def get_mask(file_path_mask: Meta[str]) -> Tuple[Spec[np.array]]:
    mask = np.ones((10, 10), dtype=bool)
    return mask,


@pyescan_metric(registry=_df_registry)
def get_mask_pixel_counts(mask: Spec[np.array]) -> Tuple[MaskStat[float]]:
    mask_pixel_count = mask.sum()
    return mask_pixel_count,


_multi_registry = MetricRegistry()


@pyescan_metric(registry=_multi_registry)
def compute_stuff(input_val: Meta[float]) -> Tuple[MaskStat[float], MaskStat[float]]:
    stat_a = input_val * 2
    stat_b = input_val * 3
    return stat_a, stat_b


# ============================================================================
# Tests
# ============================================================================


class TestMetricRegistry:
    """Tests for the metric registry and decorator."""

    def test_global_registry_has_metrics(self):
        assert len(PYESCAN_GLOBAL_METRICS.metrics) > 0

    def test_register_custom_metric(self):
        assert len(_test_registry.metrics) == 1
        assert _test_registry.metrics[0].name == "my_test_metric"

    def test_metric_callable(self):
        metric = _callable_registry.metrics[0]
        result = metric(5.0)
        assert result == (10.0,)

    def test_metric_requirements_extracted(self):
        metric = _req_registry.metrics[0]
        assert "meta:a" in metric.requirements_template
        assert "stat:b" in metric.requirements_template

    def test_metric_returns_extracted(self):
        metric = _returns_registry.metrics[0]
        assert len(metric.returns_template) == 2
        assert "stat:out_a" in metric.returns_template
        assert "stat:out_b" in metric.returns_template

    def test_parametrised_metric(self):
        metric = _param_registry.metrics[0]
        assert "multiplier" in metric.parameters
        assert "stat:result_<multiplier>" in metric.returns_template


class TestMetricProcessor:
    """Tests for the MetricProcessor dependency resolution."""

    def test_resolve_dependency(self):
        processor = MetricProcessor(_chain_registry.metrics)
        metric, params = processor.get_metric_by_stat("derived_value")
        assert metric is not None
        assert metric.name == "get_derived"

    def test_process_single_metric(self):
        processor = MetricProcessor(_chain_registry.metrics)

        class MockData:
            def __contains__(self, key):
                return key == "raw_value"
            def __getitem__(self, key):
                if key == "raw_value":
                    return 5.0
                raise KeyError(key)
            def __getattr__(self, key):
                if key == "raw_value":
                    return 5.0
                raise AttributeError(key)

        mock = MockData()
        metric, params = processor.get_metric_by_stat("base_value")
        cache = processor._process_metric(mock, metric, params)

        assert "base_value" in cache["computed_stats"]
        assert cache["computed_stats"]["base_value"] == 10.0

    def test_process_with_dependency_chain(self):
        processor = MetricProcessor(_chain_registry.metrics)

        class MockData:
            def __contains__(self, key):
                return key == "raw_value"
            def __getitem__(self, key):
                if key == "raw_value":
                    return 5.0
                raise KeyError(key)
            def __getattr__(self, key):
                if key == "raw_value":
                    return 5.0
                raise AttributeError(key)

        mock = MockData()
        metric, params = processor.get_metric_by_stat("derived_value")
        cache = processor._process_metric(mock, metric, params)

        assert "derived_value" in cache["computed_stats"]
        # raw_value=5 -> base_value=10 -> derived_value=20
        assert cache["computed_stats"]["derived_value"] == 20.0

    def test_missing_dependency_raises(self):
        processor = MetricProcessor(_orphan_registry.metrics)

        class EmptyData:
            def __contains__(self, key):
                return False

        metric, params = processor.get_metric_by_stat("output")
        with pytest.raises(ValueError, match="No metric found"):
            processor._process_metric(EmptyData(), metric, params)


class TestPandasRowWrapper:
    """Tests for PandasRowWrapperHelper."""

    def test_basic_access(self):
        row = pd.Series({"a": 1, "b": 2, "c": 3})
        wrapper = PandasRowWrapperHelper(row)
        assert wrapper.a == 1
        assert wrapper["b"] == 2

    def test_column_mapping(self):
        row = pd.Series({"col_x": 42})
        wrapper = PandasRowWrapperHelper(row, column_map={"my_name": "col_x"})
        assert wrapper.my_name == 42

    def test_special_funcs(self):
        row = pd.Series({"value": 10})
        wrapper = PandasRowWrapperHelper(
            row, special_funcs={"doubled": lambda w: w._row.loc["value"] * 2}
        )
        assert wrapper.doubled == 20

    def test_contains(self):
        row = pd.Series({"a": 1, "b": 2})
        wrapper = PandasRowWrapperHelper(row, column_map={"mapped": "a"})
        assert "a" in wrapper
        assert "mapped" in wrapper
        assert "nonexistent" not in wrapper

    def test_missing_key_raises(self):
        row = pd.Series({"a": 1})
        wrapper = PandasRowWrapperHelper(row)
        with pytest.raises(KeyError):
            _ = wrapper.nonexistent


class TestRunOnDataFrame:
    """Tests for run_on_dataframe with dummy metrics."""

    def test_run_basic(self):
        df = pd.DataFrame({
            "file_path_mask": ["/fake/path1.png", "/fake/path2.png"],
        })
        result = run_on_dataframe(
            df,
            stat_name="mask_pixel_count",
            metric_list=_df_registry.metrics,
        )
        assert result is not None
        assert len(result) == 2
        assert "mask_pixel_count" in result.columns
        # Our fake mask is 10x10 all ones = 100
        assert result["mask_pixel_count"].iloc[0] == 100

    def test_run_with_auto_merge(self):
        df = pd.DataFrame({
            "file_path_mask": ["/fake/path.png"],
            "extra_col": ["hello"],
        })
        result = run_on_dataframe(
            df,
            stat_name="mask_pixel_count",
            metric_list=_df_registry.metrics,
            auto_merge=True,
        )
        assert "extra_col" in result.columns
        assert "mask_pixel_count" in result.columns

    def test_run_nonexistent_stat_returns_none(self):
        df = pd.DataFrame({"file_path_mask": ["/fake.png"]})
        result = run_on_dataframe(
            df,
            stat_name="totally_nonexistent_stat",
            metric_list=_df_registry.metrics,
        )
        assert result is None

    def test_run_multiple_stats(self):
        df = pd.DataFrame({"input_val": [5.0, 10.0]})
        result = run_on_dataframe(
            df,
            stat_name=["stat_a", "stat_b"],
            metric_list=_multi_registry.metrics,
        )
        assert "stat_a" in result.columns
        assert "stat_b" in result.columns
        assert result["stat_a"].iloc[0] == 10.0
        assert result["stat_b"].iloc[1] == 30.0
