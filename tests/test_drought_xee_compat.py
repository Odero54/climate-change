"""Compatibility tests for the xee-backed drought map loader."""

from unittest.mock import patch

import xarray as xr

from climate_change.drought.cdi_runner import (
    _normalize_spatial_coords,
    _yearly_drought_maps,
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
