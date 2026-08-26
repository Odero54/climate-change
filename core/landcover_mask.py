from __future__ import annotations

import io
from typing import TYPE_CHECKING, cast

from requests import HTTPError

from climate_change.core.http import get_with_retry

if TYPE_CHECKING:
    import ee
    import xarray as xr

# ESA WorldCover v200 class codes (https://esa-worldcover.org)
WATER_CLASS = 80  # Permanent water bodies
BUILTUP_CLASS = 50  # Built-up

WORLDCOVER_COLLECTION = "ESA/WorldCover/v200"


def worldcover_map(aoi: ee.Geometry) -> ee.Image:
    """Raw ESA WorldCover v200 'Map' band (class codes 10-100), clipped to aoi."""
    import ee  # local import — ee must be initialised before calling this

    return ee.ImageCollection(WORLDCOVER_COLLECTION).first().select("Map").clip(aoi)


def exclusion_mask(aoi: ee.Geometry, exclude_classes: list[int]) -> ee.Image:
    """
    Boolean ee.Image (1 = keep, 0 = excluded) derived from ESA WorldCover landcover
    classes. Used to mask specific landcover classes (e.g. water bodies, built-up
    areas) out of an AOI before feature datasets are built or sampled.
    """
    lc = worldcover_map(aoi)
    mask = lc.neq(exclude_classes[0])
    for class_value in exclude_classes[1:]:
        mask = mask.And(lc.neq(class_value))
    return mask


def fetch_landcover_class_array(aoi_geojson: dict, scale: int = 100) -> xr.DataArray:
    """
    Download raw ESA WorldCover class codes over aoi_geojson as an xr.DataArray
    with (lon, lat) dims. Used where the AOI is only available as GeoJSON (no
    live ee.Geometry), e.g. to mask an already-computed spatial dataset.
    """
    import ee
    import rioxarray as rxr

    from climate_change.core.base_use_case import _ee_geometry_from_geojson

    aoi = _ee_geometry_from_geojson(aoi_geojson)
    lc = ee.ImageCollection(WORLDCOVER_COLLECTION).first().select("Map").clip(aoi)
    url = lc.getDownloadURL(
        {"region": aoi, "scale": scale, "crs": "EPSG:4326", "format": "GEO_TIFF"}
    )

    resp = get_with_retry(url, timeout=600)
    try:
        resp.raise_for_status()
    except HTTPError as exc:
        body = resp.text[:1000] if resp.text else ""
        raise HTTPError(f"{exc}. Earth Engine response: {body}", response=resp) from exc

    da = cast("xr.DataArray", rxr.open_rasterio(io.BytesIO(resp.content))).squeeze()
    return da.rename({"x": "lon", "y": "lat"})
