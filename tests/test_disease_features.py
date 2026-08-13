"""Tests for disease/features.py — constants, _normalise_date_window, and sample_training_data guards."""

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest
import xarray as xr

from climate_change.disease.features import (
    DISEASE_CLASSES,
    DISEASE_COLORS,
    FEATURE_COLS,
    RISK_SCORE_THRESHOLDS,
    SCORE_WEIGHTS,
    _normalise_date_window,
    sample_training_data,
)


def _pop_density_da(value: float = 3.0, lons=None, lats=None) -> xr.DataArray:
    """A constant-value pop_density grid wide enough to cover the test fixtures'
    lon/lat points (default lon=30.0 + i*0.01, lat=0.0), for nearest-neighbour join."""
    lons = lons if lons is not None else np.arange(29.0, 31.0, 0.01)
    lats = lats if lats is not None else np.arange(-1.0, 1.0, 0.5)
    data = np.full((len(lats), len(lons)), value)
    return xr.DataArray(data, dims=("lat", "lon"), coords={"lat": lats, "lon": lons})


class _FakeSamples:
    def __init__(self, features):
        self._features = features

    def getInfo(self):
        return {"features": self._features}


class _FakeFeatureStack:
    """Stands in for an ee.Image — only `.sample(...)` is ever called on it."""

    def __init__(self, features):
        self._features = features

    def sample(self, **kwargs):
        return _FakeSamples(self._features)


def _feature(properties, lon=30.0, lat=0.0):
    return {"properties": properties, "geometry": {"coordinates": [lon, lat]}}


def _full_properties(**overrides):
    props = {
        "rainfall_4w": 40.0,
        "temp_mean": 26.0,
        "ndwi": -0.1,
        "elevation": 1200.0,
        "pop_density": 2.5,
        "ndvi": 0.4,
        "land_cover": 0.4,
    }
    props.update(overrides)
    return props


class TestFeatureCols:
    def test_seven_features(self):
        assert len(FEATURE_COLS) == 7

    def test_expected_features_present(self):
        for col in ("rainfall_4w", "temp_mean", "ndwi", "elevation", "pop_density"):
            assert col in FEATURE_COLS


class TestDiseaseClasses:
    def test_three_classes(self):
        assert len(DISEASE_CLASSES) == 3

    def test_risk_order(self):
        assert DISEASE_CLASSES[0] == "Low Risk"
        assert DISEASE_CLASSES[-1] == "High Risk"


class TestDiseaseColors:
    def test_one_color_per_class(self):
        assert len(DISEASE_COLORS) == len(DISEASE_CLASSES)

    def test_colors_are_hex(self):
        for c in DISEASE_COLORS:
            assert c.startswith("#")


class TestScoreWeights:
    def test_weights_sum_to_one(self):
        total = sum(SCORE_WEIGHTS.values())
        assert abs(total - 1.0) < 1e-9

    def test_all_positive(self):
        for w in SCORE_WEIGHTS.values():
            assert w > 0

    def test_expected_keys(self):
        for k in ("temp_suit", "rain_suit", "ndwi_score"):
            assert k in SCORE_WEIGHTS


class TestRiskScoreThresholds:
    def test_two_values(self):
        assert len(RISK_SCORE_THRESHOLDS) == 2

    def test_ordered(self):
        assert RISK_SCORE_THRESHOLDS[0] < RISK_SCORE_THRESHOLDS[1]

    def test_within_score_range(self):
        for t in RISK_SCORE_THRESHOLDS:
            assert 0.0 < t < 100.0


class TestNormaliseDateWindow:
    def test_future_end_capped_to_safe_date(self):
        far_future = (datetime.now(timezone.utc).date() + timedelta(days=30)).isoformat()
        start, end = _normalise_date_window("2024-01-01", far_future)
        safe = (datetime.now(timezone.utc).date() - timedelta(days=7)).isoformat()
        assert end <= safe

    def test_start_after_end_pushed_back(self):
        # start > latest_safe triggers automatic adjustment
        today = datetime.now(timezone.utc).date()
        start = today.isoformat()
        end = today.isoformat()
        s, e = _normalise_date_window(start, end, minimum_days=90)
        assert s < e

    def test_normal_dates_unchanged(self):
        start, end = _normalise_date_window("2022-01-01", "2022-06-01")
        assert start == "2022-01-01"
        assert end == "2022-06-01"

    def test_end_date_before_start_after_capping(self):
        # Even if end is capped, start must be before end
        far_future = (datetime.now(timezone.utc).date() + timedelta(days=1)).isoformat()
        s, e = _normalise_date_window(far_future, far_future, minimum_days=30)
        assert s < e


class TestSampleTrainingDataGuards:
    """
    Regression tests for a real production failure: a WorldPop population band
    with zero valid pixels over an AOI never appears as a property on any sampled
    GEE feature, so the column is absent from the DataFrame entirely (not just
    NaN). Without a guard, `.dropna(subset=FEATURE_COLS)` raised an opaque
    `KeyError: [...]` instead of an actionable error.

    pop_density itself no longer reaches sample_training_data via GEE sample
    properties at all (see build_gee_feature_stack's docstring) — it's joined
    in separately from pop_density_da, precisely because the WorldPop GEE
    mirror this guard was originally written for (WorldPop/GP/100m/pop) has
    coverage gaps for some countries/AOIs. The missing-column guard below is
    now exercised via one of the remaining GEE-sampled bands instead.
    """

    def test_no_samples_raises_clear_error(self):
        stack = _FakeFeatureStack([])
        with pytest.raises(ValueError, match="No pixels could be sampled"):
            sample_training_data(stack, aoi=None, pop_density_da=_pop_density_da())

    def test_missing_band_column_raises_clear_error(self):
        # rainfall_4w absent from every sampled point's properties, mirroring a
        # fully-masked source raster over the AOI.
        features = [
            _feature({k: v for k, v in _full_properties().items() if k != "rainfall_4w"})
            for _ in range(5)
        ]
        stack = _FakeFeatureStack(features)
        with pytest.raises(ValueError, match="rainfall_4w"):
            sample_training_data(stack, aoi=None, pop_density_da=_pop_density_da())

    def test_all_bands_present_builds_dataframe(self):
        features = [_feature(_full_properties(), lon=30.0 + i * 0.01) for i in range(10)]
        stack = _FakeFeatureStack(features)
        df = sample_training_data(stack, aoi=None, pop_density_da=_pop_density_da())
        assert len(df) == 10
        for col in FEATURE_COLS:
            assert col in df.columns
        assert "label" in df.columns and "risk_score" in df.columns

    def test_pop_density_joined_from_raster_not_gee_properties(self):
        """The exact shape of the real Uganda bug: pop_density is absent from
        every sampled point's GEE properties (fully-masked WorldPop band over
        this AOI), but sample_training_data must still succeed by joining it
        in from pop_density_da instead of requiring it as a GEE property."""
        features = [
            _feature(
                {k: v for k, v in _full_properties().items() if k != "pop_density"},
                lon=30.0 + i * 0.01,
            )
            for i in range(10)
        ]
        stack = _FakeFeatureStack(features)
        df = sample_training_data(stack, aoi=None, pop_density_da=_pop_density_da(value=7.5))
        assert len(df) == 10
        assert (df["pop_density"] == 7.5).all()
