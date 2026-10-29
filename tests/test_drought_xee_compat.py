"""Compatibility tests for the xee-backed drought map loader."""

from unittest.mock import MagicMock, patch

import pytest
import xarray as xr

from climate_change.drought.cdi_runner import (
    _adaptive_grid_scale,
    _normalize_spatial_coords,
    _yearly_drought_maps,
    _yearly_drought_maps_xee_v1,
)


def test_normalize_spatial_coords_renames_x_and_y():
    ds = xr.Dataset(coords={"time": [0], "y": [1.0], "x": [2.0]})

    result = _normalize_spatial_coords(ds)

    assert "lat" in result.dims
    assert "lon" in result.dims


def test_current_xee_dispatches_to_explicit_grid_adapter():
    expected = xr.Dataset()

    with patch(
        "climate_change.drought.cdi_runner._yearly_drought_maps_xee_v1",
        return_value=expected,
    ) as adapter:
        result = _yearly_drought_maps(
            {"type": "Polygon", "coordinates": []},
            [36.0, 1.0, 38.0, 3.0],
            2005,
            2023,
        )

    assert result is expected
    adapter.assert_called_once()


def test_dispatch_passes_an_adaptive_scale_not_the_flat_default():
    """A drawn AOI's CDI map used to render as a near-single-pixel block
    regardless of size — the flat 0.1 deg default was never adjusted."""
    small_bbox = [36.90, -1.20, 36.92, -1.18]  # ~2km across, like a drawn AOI

    with patch(
        "climate_change.drought.cdi_runner._yearly_drought_maps_xee_v1",
        return_value=xr.Dataset(),
    ) as adapter:
        _yearly_drought_maps({"type": "Polygon", "coordinates": []}, small_bbox, 2005, 2023)

    _, kwargs = adapter.call_args
    assert kwargs["scale"] < 0.1


class TestAdaptiveGridScale:
    def test_refines_small_aoi_well_below_default(self):
        scale = _adaptive_grid_scale([36.90, -1.20, 36.92, -1.18])
        assert scale < 0.1
        assert scale >= 0.001  # floor

    def test_never_exceeds_floor(self):
        tiny_bbox = [36.9000, -1.2000, 36.9001, -1.1999]
        assert _adaptive_grid_scale(tiny_bbox) == 0.001

    def test_never_coarsens_a_large_aoi_beyond_the_default(self):
        country_scale_bbox = [33.0, -5.0, 41.0, 5.0]
        assert _adaptive_grid_scale(country_scale_bbox) == 0.1

    def test_degenerate_bbox_falls_back_to_default(self):
        assert _adaptive_grid_scale([36.9, -1.2, 36.9, -1.2]) == 0.1


class TestYearlyDroughtMapsXeeV1EmptyCollectionGuard:
    """An empty ee.ImageCollection handed to xr.open_dataset(engine="ee", ...)
    doesn't fail cleanly — it raises an opaque "ValueError: __len__() should
    return >= 0" from deep inside xee instead of anything naming the real
    cause. Confirmed live for Borena District, Ethiopia: ERA5/MODIS had no
    imagery for the requested date range, and the raw xee crash surfaced as
    a generic 500 to the user instead of the app's existing friendly
    "no imagery was available" message. These tests confirm the explicit
    .size().getInfo() check added ahead of xr.open_dataset catches this
    before xee ever sees the empty collection."""

    @staticmethod
    def _mock_collection(size):
        collection = MagicMock()
        collection.filterDate.return_value = collection
        collection.filterBounds.return_value = collection
        collection.select.return_value = collection
        collection.size.return_value.getInfo.return_value = size
        return collection

    def test_raises_recognized_error_when_era5_empty(self):
        era5 = self._mock_collection(0)
        modis = self._mock_collection(10)

        with (
            patch("ee.ImageCollection", side_effect=[era5, modis]),
            patch("ee.Geometry", return_value=MagicMock()),
            patch("xee.helpers.fit_geometry", return_value={}),
            patch("shapely.geometry.box", return_value=MagicMock()),
            pytest.raises(ValueError, match="imagery was available"),
        ):
            _yearly_drought_maps_xee_v1({"type": "Polygon"}, [36.0, 1.0, 38.0, 3.0], 2005, 2023)

    def test_raises_recognized_error_when_modis_empty(self):
        era5 = self._mock_collection(10)
        modis = self._mock_collection(0)

        with (
            patch("ee.ImageCollection", side_effect=[era5, modis]),
            patch("ee.Geometry", return_value=MagicMock()),
            patch("xee.helpers.fit_geometry", return_value={}),
            patch("shapely.geometry.box", return_value=MagicMock()),
            pytest.raises(ValueError, match="imagery was available"),
        ):
            _yearly_drought_maps_xee_v1({"type": "Polygon"}, [36.0, 1.0, 38.0, 3.0], 2005, 2023)

    def test_guard_runs_before_open_dataset_when_both_collections_have_data(self):
        """Confirms the new .size().getInfo() guard only short-circuits when a
        collection is genuinely empty — with non-zero sizes it must fall
        through to the existing xr.open_dataset-based pipeline untouched."""
        era5 = self._mock_collection(10)
        modis = self._mock_collection(10)

        with (
            patch("ee.ImageCollection", side_effect=[era5, modis]),
            patch("ee.Geometry", return_value=MagicMock()),
            patch("xee.helpers.fit_geometry", return_value={}),
            patch("shapely.geometry.box", return_value=MagicMock()),
            patch("xarray.open_dataset", side_effect=RuntimeError("reached open_dataset")),
            pytest.raises(RuntimeError, match="reached open_dataset"),
        ):
            _yearly_drought_maps_xee_v1({"type": "Polygon"}, [36.0, 1.0, 38.0, 3.0], 2005, 2023)
