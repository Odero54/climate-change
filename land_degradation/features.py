from __future__ import annotations

import io
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
    "ndvi_slope",
    "ndvi_mean",
    "ndvi_cv",
    "bsi",
    "ndti",
    "slope_terrain",
    "rainfall_anom",
    "land_cover",
]

DEGRADATION_CLASSES: list[str] = ["Not Degraded", "Degraded"]
DEGRADATION_COLORS: list[str] = ["#2ECC71", "#E74C3C"]

# Composite score weights (must sum to 1.0)
SCORE_WEIGHTS: dict[str, float] = {
    "s_ndvi_slope": 0.35,
    "s_bsi": 0.20,
    "s_ndti": 0.15,
    "s_rainfall_anom": 0.12,
    "s_slope_terrain": 0.10,
    "s_ndvi_cv": 0.08,
}

# Pixels scoring above this percentile are labelled Degraded
DEGRADED_PERCENTILE: int = 70


def fetch_ndvi_stack(
    aoi: "ee.Geometry",
    start: str,
    end: str,
    scale: int = 1000,
) -> xr.Dataset:
    """
    Download MODIS MOD13A3 pixel-wise NDVI trend at `scale` metres.
    Returns Dataset with variables ndvi_slope, ndvi_mean, ndvi_cv.
    """

    def _scale_ndvi(img: "ee.Image") -> "ee.Image":
        return img.multiply(0.0001).copyProperties(img, ["system:time_start"])

    def _add_time(img: "ee.Image") -> "ee.Image":
        t = img.date().difference(ee.Date(start), "year")
        return img.addBands(ee.Image(t).rename("time").float())

    modis = (
        ee.ImageCollection("MODIS/061/MOD13A3")
        .filterBounds(aoi)
        .filterDate(start, end)
        .select("NDVI")
        .map(_scale_ndvi)
    )
    trend = modis.map(_add_time).select(["time", "NDVI"]).reduce(ee.Reducer.linearFit())
    ndvi_slope = trend.select("scale").rename("ndvi_slope")
    ndvi_mean = modis.mean().rename("ndvi_mean")
    ndvi_std = modis.reduce(ee.Reducer.stdDev())
    ndvi_cv = ndvi_std.divide(ndvi_mean.abs().add(1e-6)).rename("ndvi_cv")

    stack = ndvi_slope.addBands(ndvi_mean).addBands(ndvi_cv)
    url = stack.getDownloadURL(
        {"region": aoi, "scale": scale, "crs": "EPSG:4326", "format": "GEO_TIFF"}
    )
    raw = _download_band(url)
    raw = raw.assign_coords(band=["ndvi_slope", "ndvi_mean", "ndvi_cv"])
    return raw.to_dataset(dim="band").rename({"x": "lon", "y": "lat"})


def fetch_ndvi_timeseries(
    aoi: "ee.Geometry",
    start: str,
    end: str,
) -> pd.Series:
    """
    Fetch area-averaged monthly MODIS NDVI and return as an annual pandas Series.
    Index = integer years; values = annual mean NDVI.
    """

    def _scale_ndvi(img: "ee.Image") -> "ee.Image":
        return img.multiply(0.0001).copyProperties(img, ["system:time_start"])

    def _mean_feat(img: "ee.Image") -> "ee.Feature":
        v = img.reduceRegion(ee.Reducer.mean(), aoi, 500, maxPixels=1e9)  # type: ignore[arg-type]
        return cast(
            "ee.Feature", ee.Feature(None, v).set("date", img.date().format("YYYY-MM"))
        )

    modis = (
        ee.ImageCollection("MODIS/061/MOD13A3")
        .filterBounds(aoi)
        .filterDate(start, end)
        .select("NDVI")
        .map(_scale_ndvi)
    )
    records = cast(dict, ee.FeatureCollection(modis.map(_mean_feat)).getInfo())[
        "features"
    ]
    ts_df = (
        pd.DataFrame([f["properties"] for f in records]).sort_values("date").dropna()
    )
    ndvi_monthly = pd.Series(
        ts_df["NDVI"].values,
        index=pd.to_datetime(ts_df["date"]),
        name="NDVI",
    )
    ndvi_annual = ndvi_monthly.resample("YE").mean()
    ndvi_annual.index = pd.DatetimeIndex(ndvi_annual.index).year.astype(int)
    return ndvi_annual


def fetch_s2_indices(
    aoi: "ee.Geometry",
    start: str,
    end: str,
    scale: int = 5000,
) -> xr.Dataset:
    """
    Compute annual Sentinel-2 BSI and NDTI composites (cloud < 30 %).
    Downloads all years as a single multi-band GeoTIFF and returns
    the temporal mean as a 2-variable Dataset with bsi and ndti.
    """
    years = list(range(int(start[:4]), int(end[:4]) + 1))

    def _annual(year: int) -> tuple["ee.Image", "ee.Image"]:
        col = (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(aoi)
            .filterDate(f"{year}-01-01", f"{year}-12-31")
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 30))
            .select(["B2", "B4", "B8", "B11", "B12"])
            .map(lambda i: i.multiply(0.0001))
            .median()
            .clip(aoi)
        )
        bsi = col.expression(
            "(SWIR1 + RED - NIR - BLUE) / (SWIR1 + RED + NIR + BLUE)",
            {
                "SWIR1": col.select("B11"),
                "RED": col.select("B4"),
                "NIR": col.select("B8"),
                "BLUE": col.select("B2"),
            },
        ).rename(f"bsi_{year}")
        ndti = col.expression(
            "(SWIR1 - SWIR2) / (SWIR1 + SWIR2)",
            {"SWIR1": col.select("B11"), "SWIR2": col.select("B12")},
        ).rename(f"ndti_{year}")
        return bsi, ndti

    bsi_list, ndti_list = [], []
    for yr in years:
        b, n = _annual(yr)
        bsi_list.append(b)
        ndti_list.append(n)

    band_names = [f"bsi_{yr}" for yr in years] + [f"ndti_{yr}" for yr in years]
    combined = ee.Image.cat(bsi_list + ndti_list)
    url = combined.getDownloadURL(
        {"region": aoi, "scale": scale, "crs": "EPSG:4326", "format": "GEO_TIFF"}
    )
    raw = _download_band(url)
    raw = raw.assign_coords(band=band_names)
    ds = raw.to_dataset(dim="band").rename({"x": "lon", "y": "lat"})

    bsi_mean = xr.concat([ds[f"bsi_{yr}"] for yr in years], dim="year").mean(dim="year")
    ndti_mean = xr.concat([ds[f"ndti_{yr}"] for yr in years], dim="year").mean(
        dim="year"
    )
    return xr.Dataset({"bsi": bsi_mean, "ndti": ndti_mean})


def fetch_terrain_slope(aoi: "ee.Geometry", scale: int = 1000) -> xr.Dataset:
    """Download SRTM-derived slope at `scale` metres."""
    slope = (
        ee.Terrain.slope(ee.Image("USGS/SRTMGL1_003")).clip(aoi).rename("slope_terrain")
    )
    url = slope.getDownloadURL(
        {"region": aoi, "scale": scale, "crs": "EPSG:4326", "format": "GEO_TIFF"}
    )
    da = _download_band(url).squeeze()
    return xr.Dataset({"slope_terrain": da.rename({"x": "lon", "y": "lat"})})


def fetch_rainfall_anomaly(
    aoi: "ee.Geometry",
    start: str,
    end: str,
    scale: int = 1000,
) -> xr.Dataset:
    """
    Pixel-wise CHIRPS precipitation linear trend (mm yr⁻¹) over [start, end].
    Returns Dataset with variable rainfall_anom.
    """

    def _add_t(img: "ee.Image") -> "ee.Image":
        t = img.date().difference(ee.Date(start), "year")
        return img.addBands(ee.Image(t).rename("time").float())

    chirps = (
        ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")
        .filterBounds(aoi)
        .filterDate(start, end)
        .select("precipitation")
    )
    rain_anom = (
        chirps.map(_add_t)
        .select(["time", "precipitation"])
        .reduce(ee.Reducer.linearFit())
        .select("scale")
        .rename("rainfall_anom")
        .clip(aoi)
    )
    url = rain_anom.getDownloadURL(
        {"region": aoi, "scale": scale, "crs": "EPSG:4326", "format": "GEO_TIFF"}
    )
    da = _download_band(url).squeeze()
    return xr.Dataset({"rainfall_anom": da.rename({"x": "lon", "y": "lat"})})


def fetch_landcover(aoi: "ee.Geometry", scale: int = 100) -> xr.Dataset:
    """Download ESA WorldCover 2021. Returns Dataset with variable Map."""
    worldcover = (
        ee.ImageCollection("ESA/WorldCover/v200").first().select("Map").clip(aoi)
    )
    url = worldcover.getDownloadURL(
        {"region": aoi, "scale": scale, "crs": "EPSG:4326", "format": "GEO_TIFF"}
    )
    da = _download_band(url).squeeze()
    return xr.Dataset({"Map": da.rename({"x": "lon", "y": "lat"})})


def build_feature_datasets(aoi: "ee.Geometry", config: dict) -> dict[str, xr.Dataset]:
    """
    Download all feature bands from GEE in parallel.
    The five fetch calls are independent HTTP requests; they run concurrently
    via DaskEngine.run_io_parallel (ThreadPoolExecutor) sharing the GEE session.
    Keyed by band group; consumed by both sample_training_data and cog_export.
    """
    from climate_change.core.dask_engine import DaskEngine

    scale = config.get("scale", 1000)
    start = config.get("start_date", "2015-01-01")
    end = config.get("end_date", "2024-12-31")

    return DaskEngine.run_io_parallel(
        {
            "ndvi": lambda: fetch_ndvi_stack(aoi, start, end, scale=scale),
            "s2": lambda: fetch_s2_indices(aoi, start, end),  # fixed at 5000 m
            "terrain": lambda: fetch_terrain_slope(aoi, scale=scale),
            "rainfall": lambda: fetch_rainfall_anomaly(aoi, start, end, scale=scale),
            "landcover": lambda: fetch_landcover(aoi, scale=100),
        }
    )


def build_gee_feature_stack(aoi: "ee.Geometry", config: dict) -> "ee.Image":
    """
    Assemble the 8-band GEE image used for stratified pixel sampling.
    Uses the same band order as FEATURE_COLS.
    """
    start = config.get("start_date", "2015-01-01")
    end = config.get("end_date", "2024-12-31")
    scale = config.get("scale", 1000)
    years = list(range(int(start[:4]), int(end[:4]) + 1))

    # MODIS NDVI trend bands
    def _scale_ndvi(img: "ee.Image") -> "ee.Image":
        return img.multiply(0.0001).copyProperties(img, ["system:time_start"])

    def _add_time(img: "ee.Image") -> "ee.Image":
        t = img.date().difference(ee.Date(start), "year")
        return img.addBands(ee.Image(t).rename("time").float())

    modis = (
        ee.ImageCollection("MODIS/061/MOD13A3")
        .filterBounds(aoi)
        .filterDate(start, end)
        .select("NDVI")
        .map(_scale_ndvi)
    )
    trend = modis.map(_add_time).select(["time", "NDVI"]).reduce(ee.Reducer.linearFit())
    ndvi_slope = trend.select("scale").rename("ndvi_slope")
    ndvi_mean = modis.mean().rename("ndvi_mean")
    ndvi_std = modis.reduce(ee.Reducer.stdDev())
    ndvi_cv = ndvi_std.divide(ndvi_mean.abs().add(1e-6)).rename("ndvi_cv")

    # S2 annual BSI + NDTI → temporal mean
    def _annual_gee(year: int) -> tuple["ee.Image", "ee.Image"]:
        col = (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(aoi)
            .filterDate(f"{year}-01-01", f"{year}-12-31")
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 30))
            .select(["B2", "B4", "B8", "B11", "B12"])
            .map(lambda i: i.multiply(0.0001))
            .median()
            .clip(aoi)
        )
        bsi = col.expression(
            "(SWIR1 + RED - NIR - BLUE) / (SWIR1 + RED + NIR + BLUE)",
            {
                "SWIR1": col.select("B11"),
                "RED": col.select("B4"),
                "NIR": col.select("B8"),
                "BLUE": col.select("B2"),
            },
        )
        ndti = col.expression(
            "(SWIR1 - SWIR2) / (SWIR1 + SWIR2)",
            {"SWIR1": col.select("B11"), "SWIR2": col.select("B12")},
        )
        return bsi, ndti

    bsi_list, ndti_list = [], []
    for yr in years:
        b, n = _annual_gee(yr)
        bsi_list.append(b)
        ndti_list.append(n)

    bsi_img = ee.Image.cat(bsi_list).reduce(ee.Reducer.mean()).rename("bsi")
    ndti_img = ee.Image.cat(ndti_list).reduce(ee.Reducer.mean()).rename("ndti")

    # Terrain slope
    slope = (
        ee.Terrain.slope(ee.Image("USGS/SRTMGL1_003"))
        .clip(aoi)
        .reproject("EPSG:4326", None, scale)
        .rename("slope_terrain")
    )

    # CHIRPS rainfall anomaly
    def _add_t(img: "ee.Image") -> "ee.Image":
        t = img.date().difference(ee.Date(start), "year")
        return img.addBands(ee.Image(t).rename("time").float())

    rain_anom = (
        ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")
        .filterBounds(aoi)
        .filterDate(start, end)
        .select("precipitation")
        .map(_add_t)
        .select(["time", "precipitation"])
        .reduce(ee.Reducer.linearFit())
        .select("scale")
        .rename("rainfall_anom")
        .clip(aoi)
        .reproject("EPSG:4326", None, scale)
    )

    # ESA WorldCover (normalised)
    lc_norm = (
        ee.ImageCollection("ESA/WorldCover/v200")
        .first()
        .select("Map")
        .divide(10)
        .subtract(1)
        .rename("land_cover")
        .clip(aoi)
        .reproject("EPSG:4326", None, scale)
    )

    return ee.Image.cat(
        [
            ndvi_slope,
            ndvi_mean,
            ndvi_cv,
            bsi_img.reproject("EPSG:4326", None, scale),
            ndti_img.reproject("EPSG:4326", None, scale),
            slope,
            rain_anom,
            lc_norm,
        ]
    )


def _compute_deg_score(df: pd.DataFrame) -> np.ndarray:
    """Compute composite degradation score (0–100) from a feature DataFrame."""

    def _mm(arr: np.ndarray, lo: float, hi: float) -> np.ndarray:
        return np.clip((arr - lo) / (hi - lo + 1e-8) * 100, 0, 100)

    s = (
        SCORE_WEIGHTS["s_ndvi_slope"] * _mm(-df["ndvi_slope"].to_numpy(), -0.03, 0.03)
        + SCORE_WEIGHTS["s_bsi"] * _mm(df["bsi"].to_numpy(), -0.40, 0.60)
        + SCORE_WEIGHTS["s_ndti"] * _mm(-df["ndti"].to_numpy(), -0.50, 0.30)
        + SCORE_WEIGHTS["s_rainfall_anom"]
        * _mm(-df["rainfall_anom"].to_numpy(), -0.50, 0.50)
        + SCORE_WEIGHTS["s_slope_terrain"]
        * _mm(df["slope_terrain"].to_numpy(), 0.00, 20.00)
        + SCORE_WEIGHTS["s_ndvi_cv"] * _mm(df["ndvi_cv"].to_numpy(), 0.00, 0.50)
    )
    return s


def sample_training_data(
    feature_stack: "ee.Image",
    aoi: "ee.Geometry",
    n_pixels: int = 3000,
    scale: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Sample n_pixels from the GEE feature stack and assign binary degradation labels.
    Labels: top DEGRADED_PERCENTILE % by composite score → 1 (Degraded),
            remainder → 0 (Not Degraded).
    Returns DataFrame with FEATURE_COLS + ['deg_score', 'deg_class'].
    """
    sample = feature_stack.sample(
        region=aoi,
        scale=scale,
        numPixels=n_pixels,
        seed=seed,
        dropNulls=True,
        geometries=False,
    )
    records = sample.getInfo()["features"]  # type: ignore[index]
    df = pd.DataFrame([f["properties"] for f in records]).dropna()
    df = df[FEATURE_COLS].copy()

    scores = _compute_deg_score(df)
    threshold = float(np.percentile(scores, DEGRADED_PERCENTILE))
    df["deg_score"] = scores
    df["deg_class"] = (scores >= threshold).astype(int)
    return df


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
    ref_key: str = "ndvi",
    method_continuous: InterpOptions = "linear",
    method_categorical: InterpOptions = "nearest",
) -> dict[str, xr.Dataset]:
    """
    Interpolate all datasets onto the NDVI reference grid.
    Land cover is treated as categorical (nearest-neighbour).

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
