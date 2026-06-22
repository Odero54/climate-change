"""Tests for drought/cdi_runner.py — pure utility functions (no GEE)."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from climate_change.drought.cdi_runner import (
    _classify_cdi_value,
    _normalize_drought_class,
    _temporally_fill_dataframe,
    build_drought_charts,
    compute_spatial_uncertainty,
)

# ── _classify_cdi_value ───────────────────────────────────────────────────────


class TestClassifyCdiValue:
    @pytest.mark.parametrize(
        "value,expected",
        [
            (0.3, "Extreme drought"),
            (0.55, "Severe drought"),
            (0.72, "Moderate drought"),
            (0.85, "Mild drought"),
            (1.0, "Near normal"),
            (1.15, "Mild wet"),
            (1.25, "Moderately wet"),
            (1.5, "Very wet"),
        ],
    )
    def test_classification(self, value, expected):
        assert _classify_cdi_value(value) == expected

    def test_boundary_0_50_is_severe(self):
        assert _classify_cdi_value(0.50) == "Severe drought"

    def test_boundary_1_10_is_mild_wet(self):
        assert _classify_cdi_value(1.10) == "Mild wet"


# ── _normalize_drought_class ──────────────────────────────────────────────────


class TestNormalizeDroughtClass:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("normal/wet", "Near normal"),
            ("near normal", "Near normal"),
            ("mild wet", "Mild wet"),
            ("MILD WET", "Mild wet"),  # case-insensitive? no — but raw is .lower()
            ("extreme drought", "Extreme drought"),
            ("severe drought", "Severe drought"),
            ("unknown label", "unknown label"),
        ],
    )
    def test_normalisation(self, raw, expected):
        assert _normalize_drought_class(raw) == expected


# ── _temporally_fill_dataframe ────────────────────────────────────────────────


class TestTemporallyFillDataframe:
    def _make_df(self, with_nan=True):
        idx = pd.date_range("2010-01-01", periods=24, freq="MS")
        data = {
            "PDI": np.where(np.arange(24) % 5 == 0, np.nan, 0.9) if with_nan else np.full(24, 0.9),
            "TDI": np.full(24, 0.85),
            "VDI": np.full(24, 1.0),
            "CDI": np.full(24, 0.95),
        }
        return pd.DataFrame(data, index=idx)

    def test_fills_nans(self):
        df = self._make_df(with_nan=True)
        result = _temporally_fill_dataframe(df)
        assert not result[["PDI", "TDI", "VDI", "CDI"]].isnull().any().any()

    def test_clean_df_unchanged(self):
        df = self._make_df(with_nan=False)
        result = _temporally_fill_dataframe(df)
        assert result["PDI"].tolist() == df["PDI"].tolist()

    def test_handles_missing_value_columns(self):
        idx = pd.date_range("2010-01-01", periods=5, freq="MS")
        df = pd.DataFrame({"other": [np.nan, 1.0, np.nan, 2.0, np.nan]}, index=idx)
        result = _temporally_fill_dataframe(df)
        assert not result.isnull().any().any()


# ── build_drought_charts ──────────────────────────────────────────────────────


def _make_features(n_months=60, n_time=5, n_lon=4, n_lat=3):
    idx = pd.date_range("2019-01-01", periods=n_months, freq="MS")
    rng = np.random.default_rng(7)
    df = pd.DataFrame(
        {
            "PDI": rng.uniform(0.6, 1.2, n_months),
            "TDI": rng.uniform(0.6, 1.2, n_months),
            "VDI": rng.uniform(0.6, 1.2, n_months),
            "CDI": rng.uniform(0.5, 1.3, n_months),
            "severity": ["Near normal"] * n_months,
        },
        index=idx,
    )
    times = pd.date_range("2019", periods=n_time, freq="YS")
    lons = np.linspace(36.0, 37.0, n_lon)
    lats = np.linspace(-1.0, 0.0, n_lat)
    ds = xr.Dataset(
        {
            "CDI": (["time", "lon", "lat"], rng.uniform(0.5, 1.3, (n_time, n_lon, n_lat))),
            "PDI": (["time", "lon", "lat"], rng.uniform(0.5, 1.3, (n_time, n_lon, n_lat))),
            "TDI": (["time", "lon", "lat"], rng.uniform(0.5, 1.3, (n_time, n_lon, n_lat))),
            "VDI": (["time", "lon", "lat"], rng.uniform(0.5, 1.3, (n_time, n_lon, n_lat))),
        },
        coords={"time": times, "lon": lons, "lat": lats},
    )
    return {"cdi_series": df, "cdi_maps": ds}


class TestBuildDroughtCharts:
    def test_keys_present(self):
        features = _make_features()
        result = build_drought_charts(features)
        for key in ("timeseries", "anomaly", "seasonal", "severity_distribution"):
            assert key in result

    def test_timeseries_has_four_datasets(self):
        features = _make_features()
        result = build_drought_charts(features)
        assert len(result["timeseries"]["datasets"]) == 4

    def test_seasonal_has_12_months(self):
        features = _make_features()
        result = build_drought_charts(features)
        assert result["seasonal"]["labels"] == list(range(1, 13))

    def test_severity_labels_nonempty(self):
        features = _make_features()
        result = build_drought_charts(features)
        assert len(result["severity_distribution"]["labels"]) > 0


# ── compute_spatial_uncertainty ───────────────────────────────────────────────


class TestComputeSpatialUncertainty:
    def test_keys_present(self):
        features = _make_features()
        result = compute_spatial_uncertainty(features)
        for key in (
            "lons",
            "lats",
            "temporal_std",
            "component_spread",
            "temporal_std_stats",
            "component_spread_stats",
        ):
            assert key in result

    def test_stats_have_min_max_mean(self):
        features = _make_features()
        result = compute_spatial_uncertainty(features)
        for stat_key in ("temporal_std_stats", "component_spread_stats"):
            assert "min" in result[stat_key]
            assert "max" in result[stat_key]
            assert "mean" in result[stat_key]

    def test_min_lte_mean_lte_max(self):
        features = _make_features()
        result = compute_spatial_uncertainty(features)
        for stat_key in ("temporal_std_stats", "component_spread_stats"):
            s = result[stat_key]
            assert s["min"] <= s["mean"] <= s["max"]
