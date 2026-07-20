"""Tests for core/population.py — population_by_class, at_risk_population.

fetch_population_count/fetch_population_count_safe require live GEE
credentials and aren't unit-tested here, consistent with how every other
fetch_* function across the domain features.py files is untested — see
this file's live-smoke-test note in the population-exposure feature plan.
"""

import numpy as np
import pytest
import xarray as xr
from rasterio.transform import from_bounds

from climate_change.core.population import (
    at_risk_population,
    population_by_class,
    population_exposure,
    upsample_class_grid,
)


class TestPopulationByClass:
    def test_int_coded_grid_sums_per_class(self):
        class_grid = np.array([[1, 1, 2], [2, 3, 3]])
        population_grid = np.array([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]])
        result = population_by_class(class_grid, population_grid, classes=[1, 2, 3])
        assert result == {"1": 30.0, "2": 70.0, "3": 110.0}

    def test_string_coded_grid_sums_per_class(self):
        class_grid = np.array(
            [["Extreme drought", "Severe drought"], ["Mild drought", "Extreme drought"]]
        )
        population_grid = np.array([[100.0, 200.0], [300.0, 400.0]])
        result = population_by_class(
            class_grid, population_grid, classes=["Extreme drought", "Severe drought"]
        )
        assert result == {"Extreme drought": 500.0, "Severe drought": 200.0}

    def test_nan_population_treated_as_zero(self):
        class_grid = np.array([1, 1, 1])
        population_grid = np.array([10.0, np.nan, 20.0])
        result = population_by_class(class_grid, population_grid, classes=[1])
        assert result == {"1": 30.0}

    def test_class_not_present_in_grid_returns_zero(self):
        class_grid = np.array([1, 1, 1])
        population_grid = np.array([10.0, 20.0, 30.0])
        result = population_by_class(class_grid, population_grid, classes=[1, 2])
        assert result == {"1": 60.0, "2": 0.0}

    def test_nodata_class_excluded_when_not_requested(self):
        """Class 0 (typical nodata encoding) is simply not summed unless
        explicitly requested in `classes` — callers control this by choosing
        which class values to pass, e.g. [1, 2, 3] to skip 0=nodata."""
        class_grid = np.array([0, 1, 1])
        population_grid = np.array([1000.0, 10.0, 20.0])
        result = population_by_class(class_grid, population_grid, classes=[1])
        assert result == {"1": 30.0}

    def test_shape_mismatch_raises(self):
        class_grid = np.zeros((2, 2))
        population_grid = np.zeros((3, 3))
        with pytest.raises(ValueError, match="must be aligned"):
            population_by_class(class_grid, population_grid, classes=[0])

    def test_empty_classes_returns_empty_dict(self):
        class_grid = np.array([1, 2, 3])
        population_grid = np.array([10.0, 20.0, 30.0])
        assert population_by_class(class_grid, population_grid, classes=[]) == {}


class TestUpsampleClassGrid:
    def test_upsamples_2x2_to_4x4_nearest_neighbour(self):
        # A 2x2 class grid over lon[0,2] x lat[0,2], upsampled 2x to 4x4.
        class_grid = np.array([[1, 2], [3, 4]], dtype=np.int32)
        src_transform = from_bounds(0, 0, 2, 2, 2, 2)
        dst_transform = from_bounds(0, 0, 2, 2, 4, 4)
        result = upsample_class_grid(class_grid, src_transform, (4, 4), dst_transform)
        # Each source cell should expand into a contiguous 2x2 block.
        assert result.shape == (4, 4)
        assert (result[:2, :2] == 1).all()
        assert (result[:2, 2:] == 2).all()
        assert (result[2:, :2] == 3).all()
        assert (result[2:, 2:] == 4).all()


class TestPopulationExposure:
    def _make_population_ds(self, values, lons, lats):
        return xr.Dataset(
            {"population": (["lat", "lon"], np.asarray(values, dtype=np.float64))},
            coords={"lon": lons, "lat": lats},
        )

    def test_matching_resolution_int_classes(self):
        class_grid = np.array([[1, 2], [1, 2]], dtype=np.int32)
        class_transform = from_bounds(0, 0, 2, 2, 2, 2)
        pop_ds = self._make_population_ds(
            [[10.0, 20.0], [30.0, 40.0]], lons=[0.5, 1.5], lats=[1.5, 0.5]
        )
        result = population_exposure(class_grid, class_transform, pop_ds, classes=[1, 2])
        assert sum(result.values()) == pytest.approx(100.0)
        assert set(result.keys()) == {"1", "2"}

    def test_coarser_class_grid_upsampled_onto_finer_population(self):
        """The realistic case: a coarse (e.g. 1km) risk classification grid
        and a finer (native ~100m) population grid — confirms the resolution
        mismatch is handled by upsampling the classification, and that the
        total is conserved (matches summing the population directly)."""
        # 1x2 coarse class grid: left half class 1, right half class 2.
        class_grid = np.array([[1, 2]], dtype=np.int32)
        class_transform = from_bounds(0, 0, 4, 2, 2, 1)
        # 2x4 population grid at 2x the class grid's resolution.
        pop_values = [[10.0, 20.0, 30.0, 40.0], [50.0, 60.0, 70.0, 80.0]]
        pop_ds = self._make_population_ds(pop_values, lons=[0.5, 1.5, 2.5, 3.5], lats=[1.5, 0.5])
        result = population_exposure(class_grid, class_transform, pop_ds, classes=[1, 2])
        total = sum(result.values())
        assert total == pytest.approx(sum(sum(row) for row in pop_values))
        # Left-half population (class 1) should be less than right-half (class 2)
        # since pop values increase left-to-right in this fixture.
        assert result["1"] < result["2"]

    def test_string_labelled_class_grid_encoded_and_decoded(self):
        """Drought's CDI classes are string-labelled — GDAL/rasterio's
        reprojection only supports numeric dtypes, so population_exposure
        must transparently encode/decode rather than erroring."""
        class_grid = np.array([["a", "b"], ["a", "b"]], dtype=object)
        class_transform = from_bounds(0, 0, 2, 2, 2, 2)
        pop_ds = self._make_population_ds(
            [[10.0, 20.0], [30.0, 40.0]], lons=[0.5, 1.5], lats=[1.5, 0.5]
        )
        result = population_exposure(class_grid, class_transform, pop_ds, classes=["a", "b"])
        assert set(result.keys()) == {"a", "b"}
        assert sum(result.values()) == pytest.approx(100.0)

    def test_class_not_present_returns_zero(self):
        class_grid = np.array([[1, 1], [1, 1]], dtype=np.int32)
        class_transform = from_bounds(0, 0, 2, 2, 2, 2)
        pop_ds = self._make_population_ds(
            [[10.0, 20.0], [30.0, 40.0]], lons=[0.5, 1.5], lats=[1.5, 0.5]
        )
        result = population_exposure(class_grid, class_transform, pop_ds, classes=[1, 2])
        assert result["2"] == 0.0
        assert result["1"] == pytest.approx(100.0)


class TestAtRiskPopulation:
    def test_sums_requested_labels(self):
        pop_by_class = {"Low": 100.0, "Medium": 50.0, "High": 25.0}
        assert at_risk_population(pop_by_class, ["Medium", "High"]) == 75.0

    def test_missing_label_treated_as_zero(self):
        pop_by_class = {"Low": 100.0}
        assert at_risk_population(pop_by_class, ["Medium", "High"]) == 0.0

    def test_empty_at_risk_labels_returns_zero(self):
        pop_by_class = {"Low": 100.0, "High": 50.0}
        assert at_risk_population(pop_by_class, []) == 0.0

    def test_single_label(self):
        pop_by_class = {"Degraded": 42.5, "Not Degraded": 100.0}
        assert at_risk_population(pop_by_class, ["Degraded"]) == 42.5
