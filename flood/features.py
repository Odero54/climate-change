from __future__ import annotations

import io
import math
from typing import cast

import ee
import numpy as np
import pandas as pd
import requests  # type: ignore[import-untyped]
import rioxarray as rxr
import xarray as xr
from requests import HTTPError
from xarray.core.types import InterpOptions

FEATURE_COLS: list[str] = [
    "elevation",
    "twi",
    "dist_river",
    "vv_change",
    "rainfall_7d",
    "rainfall_30d",
    "mndwi",
    "landcover",
    "longitude",
    "latitude",
]

# Risk-class integer encoding used throughout the package
RISK_INT: dict[str, int] = {"Low": 1, "Medium": 2, "High": 3, "Very High": 4}
_FLOOD_DEFAULTS = {
    "pre_flood_start": "2022-05-01",
    "pre_flood_end": "2022-07-31",
    "post_flood_start": "2022-10-01",
    "post_flood_end": "2023-01-31",
    "rain_7d_start": "2022-08-18",
    "rain_7d_end": "2022-08-25",
    "rain_30d_start": "2022-08-01",
    "rain_30d_end": "2022-08-31",
    "flood_label_start": "2021-01-01",
    "flood_label_end": "2021-12-31",
}
_GEE_DOWNLOAD_LIMIT_MARKERS = (
    "Total request size",
    "must be less than or equal to",
    "Request payload size exceeds",
)
_MAX_DOWNLOAD_SCALE_RETRIES = 4
_SAMPLE_TILE_SCALES = (4, 8, 16)
_MIN_SAMPLE_PIXELS = 500


def _is_gee_download_too_large(exc: Exception) -> bool:
    message = str(exc)
    response = getattr(exc, "response", None)
    if response is not None:
        message = f"{message}\n{getattr(response, 'text', '')}"
        if (
            getattr(response, "status_code", None) == 400
            and "earthengine.googleapis.com" in str(getattr(response, "url", ""))
            and "getPixels" in str(getattr(response, "url", ""))
        ):
            return True
    return all(marker in message for marker in _GEE_DOWNLOAD_LIMIT_MARKERS[:2]) or any(
        marker in message for marker in _GEE_DOWNLOAD_LIMIT_MARKERS
    )


def _next_download_scale(scale: int | float, attempt: int) -> int:
    multiplier = 1.6 if attempt == 0 else 2.0
    return int(math.ceil(max(float(scale) + 1, float(scale) * multiplier)))


def _download_image(
    image: "ee.Image",
    aoi: "ee.Geometry",
    scale: int,
    *,
    band_names: list[str] | None = None,
) -> xr.DataArray:
    """
    Download a GEE image, retrying at coarser resolution if Earth Engine rejects
    the request for exceeding its ~50 MB getDownloadURL payload limit.
    """
    current_scale = int(scale)
    last_exc: Exception | None = None
    attempted_scales: list[int] = []
    for attempt in range(_MAX_DOWNLOAD_SCALE_RETRIES + 1):
        attempted_scales.append(current_scale)
        try:
            url = image.getDownloadURL(
                {
                    "region": aoi,
                    "scale": current_scale,
                    "crs": "EPSG:4326",
                    "format": "GEO_TIFF",
                }
            )
            da = _download_band(url)
            da.attrs["gee_download_scale"] = current_scale
            if band_names is not None:
                da = da.assign_coords(band=band_names)
            return da
        except Exception as exc:
            last_exc = exc
            if not _is_gee_download_too_large(exc):
                raise
            if attempt >= _MAX_DOWNLOAD_SCALE_RETRIES:
                break
            current_scale = _next_download_scale(current_scale, attempt)

    assert last_exc is not None
    raise RuntimeError(
        "Earth Engine flood raster download exceeded request limits after "
        f"retrying scales {attempted_scales}."
    ) from last_exc


def _window(config: dict, key: str, legacy_key: str | None = None) -> str:
    value = config.get(key)
    if value:
        return str(value)
    if legacy_key and config.get(legacy_key):
        return str(config[legacy_key])
    return _FLOOD_DEFAULTS[key]


def _pre_flood_window(config: dict) -> tuple[str, str]:
    return (
        _window(config, "pre_flood_start", "pre_sar_start"),
        _window(config, "pre_flood_end", "pre_sar_end"),
    )


def _post_flood_window(config: dict) -> tuple[str, str]:
    return (
        _window(config, "post_flood_start", "flood_sar_start"),
        _window(config, "post_flood_end", "flood_sar_end"),
    )


def _rain_7d_window(config: dict) -> tuple[str, str]:
    return _window(config, "rain_7d_start"), _window(config, "rain_7d_end")


def _rain_30d_window(config: dict) -> tuple[str, str]:
    return _window(config, "rain_30d_start"), _window(config, "rain_30d_end")


def _mndwi_window(config: dict) -> tuple[str, str]:
    post_start, post_end = _post_flood_window(config)
    return str(config.get("mndwi_start") or post_start), str(
        config.get("mndwi_end") or post_end
    )


def _label_window(config: dict) -> tuple[str, str]:
    return _window(config, "flood_label_start"), _window(config, "flood_label_end")


# Individual band fetchers
def fetch_terrain(aoi: "ee.Geometry", scale: int = 90) -> xr.Dataset:
    """
    Download SRTM elevation from GEE.
    Returns Dataset with variable 'elevation' and (lat, lon) coords.
    """
    dem = ee.Image("USGS/SRTMGL1_003").select("elevation").clip(aoi)
    da = _download_image(dem, aoi, scale).squeeze()
    return xr.Dataset({"elevation": da.rename({"x": "lon", "y": "lat"})})


def fetch_twi(aoi: "ee.Geometry", scale: int = 500) -> xr.Dataset:
    """
    Compute Topographic Wetness Index from HydroSHEDS flow accumulation + SRTM slope.
    Returns Dataset with variables 'twi' and 'flow_acc_log'.
    """
    dem = (
        ee.Image("USGS/SRTMGL1_003")
        .select("elevation")
        .reproject("EPSG:4326", None, 500)
    )
    slope_rad = (
        ee.Terrain.products(dem)
        .select("slope")
        .clip(aoi)
        .multiply(np.pi / 180)
        .max(ee.Image(0.001))
    )
    flow_acc = ee.Image("WWF/HydroSHEDS/30ACC").select("b1").clip(aoi)
    twi = flow_acc.multiply(810_000).divide(slope_rad.tan()).log().rename("twi")
    flow_acc_log = flow_acc.add(1).log().rename("flow_acc_log")

    raw = _download_image(
        twi.addBands(flow_acc_log), aoi, scale, band_names=["twi", "flow_acc_log"]
    )
    ds = raw.to_dataset(dim="band").rename({"x": "lon", "y": "lat"})
    return ds


def fetch_sar_change(
    aoi: "ee.Geometry",
    pre_start: str,
    pre_end: str,
    flood_start: str,
    flood_end: str,
    scale: int = 90,
) -> xr.Dataset:
    """
    Compute Sentinel-1 VV backscatter change (pre − flood).
    Positive values indicate a backscatter drop → open water / flood signal.
    Returns Dataset with variable 'vv_change'.
    """
    sar = (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterBounds(aoi)
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .select("VV")
    )
    vv_change = (
        sar.filterDate(pre_start, pre_end)
        .mean()
        .unmask(0)
        .clip(aoi)
        .subtract(sar.filterDate(flood_start, flood_end).mean().unmask(0).clip(aoi))
        .rename("vv_change")
    )
    da = _download_image(vv_change, aoi, scale).squeeze()
    return xr.Dataset({"vv_change": da.rename({"x": "lon", "y": "lat"})})


def fetch_rainfall(
    aoi: "ee.Geometry",
    start_7d: str,
    end_7d: str,
    start_30d: str,
    end_30d: str,
    scale: int = 500,
) -> xr.Dataset:
    """
    Download CHIRPS cumulative rainfall for two windows.
    Returns Dataset with variables 'rainfall_7d' and 'rainfall_30d'.
    """
    chirps = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY").filterBounds(aoi)
    rain = (
        chirps.filterDate(start_7d, end_7d)
        .sum()
        .clip(aoi)
        .rename("rainfall_7d")
        .addBands(
            chirps.filterDate(start_30d, end_30d).sum().clip(aoi).rename("rainfall_30d")
        )
    )
    raw = _download_image(
        rain,
        aoi,
        scale,
        band_names=["rainfall_7d", "rainfall_30d"],
    )
    return raw.to_dataset(dim="band").rename({"x": "lon", "y": "lat"})


def fetch_landcover(aoi: "ee.Geometry", scale: int = 100) -> xr.Dataset:
    """
    Download ESA WorldCover 2021 land cover.
    Returns Dataset with variable 'Map' (raw class values 10–95).
    """
    worldcover = (
        ee.ImageCollection("ESA/WorldCover/v200").first().select("Map").clip(aoi)
    )
    da = _download_image(worldcover, aoi, scale).squeeze()
    return xr.Dataset({"Map": da.rename({"x": "lon", "y": "lat"})})


def fetch_dist_river(aoi: "ee.Geometry", scale: int = 90) -> xr.Dataset:
    """
    Compute Euclidean distance to permanent water (JRC occurrence ≥ 70 %).
    Returns Dataset with variable 'dist_river' in metres.
    """
    water_mask = (
        ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("occurrence").gte(70).clip(aoi)
    )
    dist_river = (
        water_mask.fastDistanceTransform(256, "pixels")
        .sqrt()
        .multiply(ee.Image.pixelArea().sqrt())
        .rename("dist_river")
        .clip(aoi)
    )
    da = _download_image(dist_river, aoi, scale).squeeze()
    return xr.Dataset({"dist_river": da.rename({"x": "lon", "y": "lat"})})


def fetch_mndwi(
    aoi: "ee.Geometry",
    start_date: str,
    end_date: str,
    scale: int = 90,
) -> xr.Dataset:
    """
    Compute Sentinel-2 MNDWI = (Green − SWIR1) / (Green + SWIR1).
    Returns Dataset with variable 'mndwi'.
    """
    s2 = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(aoi)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 30))
        .select(["B3", "B11"])
        .median()
        .clip(aoi)
    )
    mndwi = s2.normalizedDifference(["B3", "B11"]).rename("mndwi")
    da = _download_image(mndwi, aoi, scale).squeeze()
    return xr.Dataset({"mndwi": da.rename({"x": "lon", "y": "lat"})})


# ── Private GEE image builders (run in background threads) ─────────────────────


def _build_static_image(aoi: "ee.Geometry") -> "ee.Image":
    """Elevation, TWI, dist_river, landcover_norm, coords — event-independent."""
    scale_500 = 500
    dem = (
        ee.Image("USGS/SRTMGL1_003")
        .select("elevation")
        .reproject("EPSG:4326", None, scale_500)
    )
    slope_rad = (
        ee.Terrain.products(dem)
        .select("slope")
        .clip(aoi)
        .multiply(np.pi / 180)
        .max(ee.Image(0.001))
    )
    flow_acc = ee.Image("WWF/HydroSHEDS/30ACC").select("b1").clip(aoi)
    twi = flow_acc.multiply(810_000).divide(slope_rad.tan()).log().rename("twi")
    water_mask = (
        ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("occurrence").gte(70).clip(aoi)
    )
    dist_river = (
        water_mask.fastDistanceTransform(256, "pixels")
        .sqrt()
        .multiply(ee.Image.pixelArea().sqrt())
        .rename("dist_river")
        .clip(aoi)
    )
    worldcover = (
        ee.ImageCollection("ESA/WorldCover/v200")
        .filterBounds(aoi)
        .first()
        .select("Map")
        .unmask(30)
        .clip(aoi)
    )
    landcover_norm = worldcover.divide(10).subtract(1).rename("landcover")
    elevation = ee.Image("USGS/SRTMGL1_003").select("elevation").clip(aoi)
    coords = ee.Image.pixelLonLat().rename(["longitude", "latitude"]).clip(aoi)
    return ee.Image.cat([elevation, twi, dist_river, landcover_norm, coords])


def _build_dynamic_image(aoi: "ee.Geometry", config: dict) -> "ee.Image":
    """vv_change, rainfall_7d, rainfall_30d, mndwi — event-specific."""
    pre_start, pre_end = _pre_flood_window(config)
    post_start, post_end = _post_flood_window(config)
    rain_7d_start, rain_7d_end = _rain_7d_window(config)
    rain_30d_start, rain_30d_end = _rain_30d_window(config)
    mndwi_start, mndwi_end = _mndwi_window(config)
    sar = (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterBounds(aoi)
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .select("VV")
    )
    vv_change = (
        sar.filterDate(pre_start, pre_end)
        .mean()
        .unmask(0)
        .clip(aoi)
        .subtract(sar.filterDate(post_start, post_end).mean().unmask(0).clip(aoi))
        .rename("vv_change")
    )
    chirps = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY").filterBounds(aoi)
    rainfall_7d = (
        chirps.filterDate(rain_7d_start, rain_7d_end)
        .sum()
        .clip(aoi)
        .rename("rainfall_7d")
    )
    rainfall_30d = (
        chirps.filterDate(rain_30d_start, rain_30d_end)
        .sum()
        .clip(aoi)
        .rename("rainfall_30d")
    )
    s2 = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(aoi)
        .filterDate(mndwi_start, mndwi_end)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 30))
        .select(["B3", "B11"])
        .median()
        .unmask(0)
        .clip(aoi)
    )
    mndwi = s2.normalizedDifference(["B3", "B11"]).rename("mndwi")
    return ee.Image.cat([vv_change, rainfall_7d, rainfall_30d, mndwi])


def _build_jrc_label(
    aoi: "ee.Geometry",
    flood_start: str,
    flood_end: str,
) -> "ee.Image":
    """JRC flood-year label with fallback for years beyond v1.4 coverage (> 2021)."""
    jrc_yearly = ee.ImageCollection("JRC/GSW1_4/YearlyHistory")
    jrc_filtered = jrc_yearly.filterDate(flood_start, flood_end)
    flood_year = ee.Image(
        ee.Algorithms.If(
            jrc_filtered.size().gt(0),
            jrc_filtered.first(),
            jrc_yearly.sort("system:time_start", False).first(),
        )
    ).clip(aoi)
    permanent_baseline = (
        ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("occurrence").gte(75).clip(aoi)
    )
    return (
        flood_year.eq(2)
        .And(permanent_baseline.Not())
        .rename("is_flooded")
        .toInt()
        .clip(aoi)
    )


# Feature stack
def build_feature_datasets(aoi: "ee.Geometry", config: dict) -> dict[str, xr.Dataset]:
    """
    Download all feature bands from GEE in parallel.
    The seven fetch calls are independent HTTP requests; they run concurrently
    via DaskEngine.run_io_parallel (ThreadPoolExecutor) sharing the GEE session.
    The dict is keyed by band group name and consumed by both sample_training_data
    and cog_export.export_flood_cog.
    """
    from climate_change.core.dask_engine import DaskEngine

    scale = config.get("scale", 90)
    pre_start, pre_end = _pre_flood_window(config)
    post_start, post_end = _post_flood_window(config)
    rain_7d_start, rain_7d_end = _rain_7d_window(config)
    rain_30d_start, rain_30d_end = _rain_30d_window(config)
    mndwi_start, mndwi_end = _mndwi_window(config)

    return DaskEngine.run_io_parallel(
        {
            "terrain": lambda: fetch_terrain(aoi, scale=scale),
            "twi": lambda: fetch_twi(aoi, scale=500),
            "sar": lambda: fetch_sar_change(
                aoi,
                pre_start,
                pre_end,
                post_start,
                post_end,
                scale=scale,
            ),
            "rainfall": lambda: fetch_rainfall(
                aoi,
                rain_7d_start,
                rain_7d_end,
                rain_30d_start,
                rain_30d_end,
                scale=500,
            ),
            "landcover": lambda: fetch_landcover(aoi, scale=100),
            "dist_river": lambda: fetch_dist_river(aoi, scale=scale),
            "mndwi": lambda: fetch_mndwi(
                aoi,
                mndwi_start,
                mndwi_end,
                scale=scale,
            ),
        }
    )


def build_gee_feature_stack(aoi: "ee.Geometry", config: dict) -> "ee.Image":
    """
    Assemble the 10-band GEE image used for stratified sampling.
    Static (event-independent) and dynamic (event-specific) bands are built
    concurrently in background threads, then selected into FEATURE_COLS order.
    """
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=2) as pool:
        f_static = pool.submit(_build_static_image, aoi)
        f_dynamic = pool.submit(_build_dynamic_image, aoi, config)
        static_img = f_static.result()
        dynamic_img = f_dynamic.result()

    return ee.Image.cat([static_img, dynamic_img]).select(FEATURE_COLS)


# Training data sampling
def sample_training_data(
    aoi: "ee.Geometry",
    config: dict,
    n_pixels: int = 5000,
    scale: int = 90,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Build the 10-band feature stack and derive JRC flood labels concurrently in
    background threads, then sample n_pixels flooded + n_pixels non-flooded pixels.

    Static features (elevation, TWI, dist_river, landcover, coords), dynamic
    features (vv_change, rainfall, mndwi), and the JRC label image are all
    fetched in parallel — no external feature_stack argument required.

    Labels: 1 = seasonal flood water not part of the permanent baseline
            0 = all other land
    Returns a DataFrame with columns = FEATURE_COLS + ['is_flooded'].
    """
    from concurrent.futures import ThreadPoolExecutor

    flood_start, flood_end = _label_window(config)

    with ThreadPoolExecutor(max_workers=3) as pool:
        f_static = pool.submit(_build_static_image, aoi)
        f_dynamic = pool.submit(_build_dynamic_image, aoi, config)
        f_label = pool.submit(_build_jrc_label, aoi, flood_start, flood_end)
        static_img = f_static.result()
        dynamic_img = f_dynamic.result()
        flood_label = f_label.result()

    feature_stack = ee.Image.cat([static_img, dynamic_img]).select(FEATURE_COLS)
    labeled = feature_stack.addBands(flood_label)
    records = _sample_labeled_pixels(
        labeled=labeled,
        flood_label=flood_label,
        aoi=aoi,
        scale=scale,
        n_pixels=n_pixels,
        seed=seed,
    )
    df = pd.DataFrame([f["properties"] for f in records]).dropna()
    df["is_flooded"] = df["is_flooded"].astype(int)
    return df


def _sample_labeled_pixels(
    *,
    labeled: "ee.Image",
    flood_label: "ee.Image",
    aoi: "ee.Geometry",
    scale: int,
    n_pixels: int,
    seed: int,
) -> list[dict]:
    last_exc: Exception | None = None
    requested = max(_MIN_SAMPLE_PIXELS, int(n_pixels))
    sample_sizes = [
        requested,
        max(_MIN_SAMPLE_PIXELS, requested // 2),
        max(_MIN_SAMPLE_PIXELS, requested // 4),
    ]

    for sample_size in dict.fromkeys(sample_sizes):
        for tile_scale in _SAMPLE_TILE_SCALES:
            try:
                sample_flooded = labeled.updateMask(flood_label).sample(
                    region=aoi,
                    scale=scale,
                    numPixels=sample_size,
                    seed=seed,
                    dropNulls=True,
                    geometries=False,
                    tileScale=tile_scale,
                )
                sample_dry = labeled.updateMask(flood_label.Not()).sample(
                    region=aoi,
                    scale=scale,
                    numPixels=sample_size,
                    seed=seed,
                    dropNulls=True,
                    geometries=False,
                    tileScale=tile_scale,
                )
                return sample_flooded.merge(sample_dry).getInfo()["features"]  # type: ignore[index]
            except Exception as exc:
                last_exc = exc
                if "User memory limit exceeded" not in str(exc):
                    raise

    raise RuntimeError(
        "Earth Engine could not sample this area because the request is too large. "
        "Try a smaller area of interest or a shorter flood date window."
    ) from last_exc


def _download_band(url: str, timeout: int = 600) -> xr.DataArray:
    """GET a GEE download URL and return a rioxarray DataArray."""
    resp = requests.get(url, timeout=timeout)
    try:
        resp.raise_for_status()
    except HTTPError as exc:
        body = resp.text[:1000] if resp.text else ""
        raise HTTPError(
            f"{exc}. Earth Engine response: {body}",
            response=resp,
        ) from exc
    return cast(xr.DataArray, rxr.open_rasterio(io.BytesIO(resp.content)))


def align_datasets(
    datasets: dict[str, xr.Dataset],
    ref_key: str = "terrain",
    method_continuous: InterpOptions = "linear",
    method_categorical: InterpOptions = "nearest",
) -> dict[str, xr.Dataset]:
    """
    Interpolate all datasets onto the reference grid (terrain at 90 m by default).
    Landcover is treated as categorical and uses nearest-neighbour interpolation.

    Each dataset is chunked before interpolation so that xarray produces
    Dask-backed lazy arrays; dask.compute() then materialises them all
    in a single parallel scheduler pass.
    """
    import dask

    _CHUNK = {"lat": 256, "lon": 256}
    ref = datasets[ref_key]

    lazy: dict[str, xr.Dataset] = {ref_key: ref}
    for key, ds in datasets.items():
        if key == ref_key:
            continue
        method = method_categorical if key == "landcover" else method_continuous
        lazy[key] = ds.chunk(_CHUNK).interp(lat=ref.lat, lon=ref.lon, method=method)

    keys = list(lazy)
    computed = dask.compute(*[lazy[k] for k in keys])
    return dict(zip(keys, computed))
