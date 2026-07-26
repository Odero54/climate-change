from __future__ import annotations

import io
import logging
import math
from typing import TYPE_CHECKING, cast

import numpy as np
import requests
from requests import HTTPError

if TYPE_CHECKING:
    import ee
    import xarray as xr

_log = logging.getLogger(__name__)

POPULATION_COLLECTION = "WorldPop/GP/100m/pop"
_NATIVE_SCALE = 100  # WorldPop's native pixel size in metres

# GEE's synchronous getDownloadURL rejects any request whose exported raster
# would exceed 50,331,648 bytes (observed live: a 400 response quoting that
# exact figure). Tile with a safety margin under it rather than against it
# exactly, since _bbox_pixel_bytes is only an estimate (equirectangular
# metres-per-degree, uncompressed float32 pixel count) and the actual GeoTIFF
# adds header/tag overhead on top.
_MAX_DOWNLOAD_BYTES = 40_000_000
_BYTES_PER_PIXEL = 4  # WorldPop population band is float32
_METRES_PER_DEGREE_LAT = 110_540  # ~constant; a degree of longitude scales by cos(latitude)


def _collection_size(collection: ee.ImageCollection) -> int:
    return int(cast(int, collection.size().getInfo()))


def _require_population_image(collection: ee.ImageCollection, year: int) -> ee.Image:
    """
    Guard against an empty WorldPop filter result. Calling `.first()` on an
    empty collection silently yields a bandless image — every downstream
    `.select()` on it then fails opaquely, or worse, the exposure computation
    would just report 0 people instead of raising a clear data-availability
    error.
    """
    if _collection_size(collection) == 0:
        raise ValueError(f"No WorldPop population data was available for {year}.")
    return collection.first()


def _bbox_pixel_bytes(
    min_lon: float, min_lat: float, max_lon: float, max_lat: float, scale: int
) -> int:
    """Estimate the uncompressed byte size of a GEO_TIFF export covering this
    bounding box at the given scale (metres/pixel), used to decide whether a
    single getDownloadURL call would exceed GEE's per-request size cap."""
    mean_lat_rad = math.radians((min_lat + max_lat) / 2)
    width_m = (max_lon - min_lon) * _METRES_PER_DEGREE_LAT * math.cos(mean_lat_rad)
    height_m = (max_lat - min_lat) * _METRES_PER_DEGREE_LAT
    width_px = max(1, math.ceil(width_m / scale))
    height_px = max(1, math.ceil(height_m / scale))
    return width_px * height_px * _BYTES_PER_PIXEL


def _split_bbox_into_tiles(
    min_lon: float,
    min_lat: float,
    max_lon: float,
    max_lat: float,
    scale: int,
    max_bytes: int = _MAX_DOWNLOAD_BYTES,
) -> list[tuple[float, float, float, float]]:
    """
    Split a bounding box into a regular grid of sub-boxes, each estimated to
    stay under `max_bytes` when exported at `scale`, so a large AOI can be
    fetched as several within-limit GEE downloads instead of one oversized
    one. Returns the whole bbox as a single tile when it already fits.
    """
    total_bytes = _bbox_pixel_bytes(min_lon, min_lat, max_lon, max_lat, scale)
    if total_bytes <= max_bytes:
        return [(min_lon, min_lat, max_lon, max_lat)]

    grid_n = math.ceil(math.sqrt(total_bytes / max_bytes))
    lon_step = (max_lon - min_lon) / grid_n
    lat_step = (max_lat - min_lat) / grid_n

    return [
        (
            min_lon + i * lon_step,
            min_lat + j * lat_step,
            min_lon + (i + 1) * lon_step,
            min_lat + (j + 1) * lat_step,
        )
        for i in range(grid_n)
        for j in range(grid_n)
    ]


def fetch_population_count(aoi: ee.Geometry, scale: int, year: int = 2020) -> xr.Dataset:
    """
    WorldPop GP 100 m population COUNT (people per pixel — an additive
    quantity, not a density) for population-exposure reporting.

    Deliberately separate from any domain's log-transformed pop_density ML
    feature (e.g. disease/features.py::fetch_pop_density) — this returns raw
    counts suitable for summing within a risk zone, never model input.

    Always fetches at (up to) the native ~100 m resolution regardless of the
    requested `scale` — i.e. min(scale, 100). An earlier version of this
    function used ee.Image.reduceResolution(Reducer.sum()) to downsample to a
    domain's coarser working scale before download, on the assumption that
    would sum native pixels into each output pixel. Verified against live GEE
    data this was wrong: reduceResolution's default area-weighting turns
    sum() into something closer to a coverage fraction (single-pixel test:
    output value was ~1 instead of the true ~25/~100 contributing pixels),
    and even with `.unweighted()` the boundary handling still over-counted by
    20-60%+ against a server-side reduceRegion ground truth, including at a
    realistic ~100km county-sized AOI (not just a small edge-effect-prone
    test box). There is no reliable way found to make GEE aggregate this
    correctly server-side, so this function never downsamples population at
    all — see population_exposure() below for how the resolution mismatch
    with a domain's coarser risk-classification grid is actually handled (by
    upsampling the *classification*, not downsampling the population).

    Returns Dataset with variable 'population', values in people-per-pixel,
    already masked (0.0) where WorldPop's -99999 nodata sentinel appears —
    verified live that the GeoTIFF export does not set a GDAL/rasterio
    nodata tag (da.rio.nodata comes back None), so summing the raw array
    would otherwise corrupt totals by many orders of magnitude.

    A large AOI at native ~100 m resolution can serialize to more bytes than
    GEE's getDownloadURL allows in one request (50,331,648 bytes — observed
    live from a 400 response for a ~54.5 MB export). Rather than coarsening
    the scale to fit (see the reduceResolution discussion above for why that
    silently corrupts totals), the AOI's bounding box is split into a grid of
    sub-regions that each fit under the cap, fetched individually at the same
    native scale, and mosaicked back together — resolution and correctness
    are unaffected, just the number of HTTP round trips.
    """
    import ee
    import xarray as xr

    pop_collection = (
        ee.ImageCollection(POPULATION_COLLECTION)
        .filterBounds(aoi)
        .filter(ee.Filter.eq("year", year))
    )
    pop_img = _require_population_image(pop_collection, year).select("population").clip(aoi)
    fetch_scale = min(scale, _NATIVE_SCALE)

    bounds_coords = cast(list, aoi.bounds().coordinates().getInfo())[0]
    lons = [c[0] for c in bounds_coords]
    lats = [c[1] for c in bounds_coords]
    tiles = _split_bbox_into_tiles(min(lons), min(lats), max(lons), max(lats), fetch_scale)

    if len(tiles) == 1:
        url = pop_img.getDownloadURL(
            {"region": aoi, "scale": fetch_scale, "crs": "EPSG:4326", "format": "GEO_TIFF"}
        )
        da = _download_band(url).squeeze()
    else:
        _log.info(
            "Population raster for this AOI exceeds GEE's per-request size cap; "
            "fetching in %d tiles at scale=%dm.",
            len(tiles),
            fetch_scale,
        )
        tile_arrays = []
        for min_lon, min_lat, max_lon, max_lat in tiles:
            tile_region = ee.Geometry.Rectangle(
                [min_lon, min_lat, max_lon, max_lat], proj="EPSG:4326", geodesic=False
            )
            url = pop_img.getDownloadURL(
                {
                    "region": tile_region,
                    "scale": fetch_scale,
                    "crs": "EPSG:4326",
                    "format": "GEO_TIFF",
                }
            )
            tile_arrays.append(_download_band(url).squeeze())
        da = _merge_tiles(tile_arrays)

    # Population can't be negative — clamp WorldPop's unmasked -99999
    # sentinel (and any other negative noise) to 0 rather than relying on a
    # nodata tag that isn't actually present in the export.
    da = da.copy(data=np.where(da.values < 0, 0.0, da.values))
    return xr.Dataset({"population": da.rename({"x": "lon", "y": "lat"})})


def fetch_population_count_safe(
    aoi: ee.Geometry, scale: int, year: int = 2020
) -> xr.Dataset | None:
    """
    Best-effort wrapper around fetch_population_count.

    Population exposure is an enrichment on top of the core risk analysis,
    not a required input — WorldPop being unavailable for a given AOI/year
    (or a transient network failure) must not crash the whole analysis the
    way a missing rainfall/temperature/NDVI band legitimately should. Returns
    None on any failure; callers should skip exposure reporting in that case.
    """
    try:
        return fetch_population_count(aoi, scale=scale, year=year)
    except Exception:
        _log.warning(
            "Population fetch failed; population exposure stats will be omitted",
            exc_info=True,
        )
        return None


def _download_band(url: str, timeout: int = 600) -> xr.DataArray:
    """GET a GEE download URL and return a rioxarray DataArray."""
    import rioxarray as rxr

    resp = requests.get(url, timeout=timeout)
    try:
        resp.raise_for_status()
    except HTTPError as exc:
        body = resp.text[:1000] if resp.text else ""
        raise HTTPError(f"{exc}. Earth Engine response: {body}", response=resp) from exc
    return cast("xr.DataArray", rxr.open_rasterio(io.BytesIO(resp.content)))


def _merge_tiles(tiles: list[xr.DataArray]) -> xr.DataArray:
    """Mosaic adjacent same-scale rioxarray tiles (from _split_bbox_into_tiles)
    back into a single DataArray on one regular grid."""
    from rioxarray.merge import merge_arrays

    return cast("xr.DataArray", merge_arrays(tiles))


def population_by_class(
    class_grid: np.ndarray,
    population_grid: np.ndarray,
    classes: list,
) -> dict[str, float]:
    """
    Zonal sum of population per value present in class_grid.

    class_grid and population_grid must already be the same shape and cover
    the same grid. In practice, prefer population_exposure() below, which
    handles the (typical) case where the risk-classification grid is coarser
    than population's native resolution — calling this directly requires the
    caller to have already resolved that mismatch. Works with int-coded risk
    grids (disease/flood/food_security/land_degradation) or string-labelled
    class grids (drought's CDI severity classes) alike — `classes` is
    whatever set of class tokens is meaningful for that grid's dtype.
    Returned dict is keyed by str(class token).
    """
    if class_grid.shape != population_grid.shape:
        raise ValueError(
            f"class_grid shape {class_grid.shape} != population_grid shape "
            f"{population_grid.shape} — must be aligned to the same grid. "
            "Use population_exposure() if the two grids have different resolutions."
        )
    pop_valid = np.nan_to_num(population_grid.astype(np.float64), nan=0.0)
    return {str(cls): round(float(pop_valid[class_grid == cls].sum()), 1) for cls in classes}


def upsample_class_grid(
    class_grid: np.ndarray,
    class_transform,
    target_shape: tuple[int, int],
    target_transform,
    crs: str = "EPSG:4326",
) -> np.ndarray:
    """
    Nearest-neighbour upsample a (coarse) classified grid onto a finer target
    grid — typically population's native ~100 m resolution. Nearest-neighbour
    is the semantically correct choice for a categorical class grid (each
    finer output pixel just inherits its parent coarse cell's class), unlike
    downsampling an additive quantity like population counts, which is why
    the resolution mismatch is resolved this direction and not the other.
    """
    import rasterio.warp

    dst = np.zeros(target_shape, dtype=class_grid.dtype)
    rasterio.warp.reproject(
        source=class_grid,
        destination=dst,
        src_transform=class_transform,
        src_crs=crs,
        dst_transform=target_transform,
        dst_crs=crs,
        resampling=rasterio.warp.Resampling.nearest,
    )
    return dst


def population_exposure(
    class_grid: np.ndarray,
    class_transform,
    population_ds: xr.Dataset,
    classes: list,
) -> dict[str, float]:
    """
    Zonal sum of population per class, handling the resolution mismatch
    between a domain's (coarser) risk-classification grid and population's
    native ~100 m grid: upsamples class_grid onto population's grid via
    nearest-neighbour, then delegates to population_by_class. This is the
    primary entry point domains should use — see population_by_class's
    docstring for why population itself is never downsampled instead.

    class_grid may hold int-coded classes (disease/flood/food_security/
    land_degradation's risk grids) or string labels (drought's CDI severity
    classes) — GDAL/rasterio's reprojection (used internally to upsample
    class_grid) only supports numeric dtypes, so a non-numeric class_grid is
    transparently encoded to small integers before upsampling and decoded
    back to the original `classes` labels in the returned dict.
    """
    from rasterio.transform import from_bounds

    pop_da = population_ds["population"]
    lons = np.asarray(pop_da["lon"].values)
    lats = np.asarray(pop_da["lat"].values)
    pop_transform = from_bounds(
        float(lons.min()),
        float(lats.min()),
        float(lons.max()),
        float(lats.max()),
        len(lons),
        len(lats),
    )
    pop_grid = np.asarray(pop_da.values).astype(np.float64)

    if np.issubdtype(class_grid.dtype, np.number):
        upsampled_class = upsample_class_grid(
            class_grid, class_transform, pop_grid.shape, pop_transform
        )
        return population_by_class(upsampled_class, pop_grid, classes=classes)

    label_to_idx = {label: i for i, label in enumerate(classes)}
    encoded = np.full(class_grid.shape, -1, dtype=np.int32)
    for label, idx in label_to_idx.items():
        encoded[class_grid == label] = idx
    upsampled_encoded = upsample_class_grid(encoded, class_transform, pop_grid.shape, pop_transform)
    by_idx = population_by_class(upsampled_encoded, pop_grid, classes=list(label_to_idx.values()))
    return {label: by_idx[str(idx)] for label, idx in label_to_idx.items()}


def at_risk_population(pop_by_class: dict[str, float], at_risk_labels: list[str]) -> float:
    """Sum population across the given at-risk class labels (keys of pop_by_class)."""
    return round(sum(pop_by_class.get(label, 0.0) for label in at_risk_labels), 1)
