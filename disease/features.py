from __future__ import annotations

import io
from datetime import datetime, timedelta, timezone
from typing import cast

import ee
import numpy as np
import pandas as pd
import requests
import rioxarray as rxr
import xarray as xr
from requests import HTTPError
from xarray.core.types import InterpOptions

FEATURE_COLS: list[str] = [
    "rainfall_4w",  # CHIRPS 28-day cumulative rainfall (mm)
    "temp_mean",  # MODIS daytime LST (°C)
    "ndwi",  # Sentinel-2 MNDWI — surface / standing water
    "elevation",  # SRTM elevation (m)
    "pop_density",  # WorldPop log(1 + pop density)
    "ndvi",  # MODIS NDVI — vegetation / humidity proxy
    "land_cover",  # ESA WorldCover normalised (0–1)
]

# Risk class definitions
DISEASE_CLASSES: list[str] = ["Low Risk", "Medium Risk", "High Risk"]
DISEASE_COLORS: list[str] = ["#2ECC71", "#F1C40F", "#E74C3C"]

# Composite risk score weights (must sum to 1.0)
# Weights: temperature suitability 40 %, rainfall trigger 35 %, standing water 25 %
SCORE_WEIGHTS: dict[str, float] = {
    "temp_suit": 0.40,
    "rain_suit": 0.35,
    "ndwi_score": 0.25,
}

# Tercile threshold for labelling (top 1/3 = High Risk)
RISK_PERCENTILES: tuple[float, float] = (1 / 3, 2 / 3)


def _normalise_date_window(start: str, end: str, minimum_days: int = 90) -> tuple[str, str]:
    """
    Keep requested disease windows inside the likely available satellite archive.
    Users can accidentally select today/future dates; some GEE collections lag by a
    few days, and empty collections produce invalid zero-band images.
    """
    latest_safe = datetime.now(timezone.utc).date() - timedelta(days=7)
    start_date = datetime.strptime(start, "%Y-%m-%d").date()
    end_date = datetime.strptime(end, "%Y-%m-%d").date()
    if end_date > latest_safe:
        end_date = latest_safe
    if start_date >= end_date:
        start_date = end_date - timedelta(days=minimum_days)
    return start_date.isoformat(), end_date.isoformat()


def _collection_size(collection: ee.ImageCollection) -> int:
    return int(cast(int, collection.size().getInfo()))


def _require_images(
    collection: ee.ImageCollection, label: str, start: str, end: str
) -> ee.ImageCollection:
    if _collection_size(collection) == 0:
        raise ValueError(f"No {label} imagery was available for {start} to {end}.")
    return collection


def _sentinel2_mndwi_collection(aoi: ee.Geometry, start: str, end: str) -> ee.ImageCollection:
    def _add_mndwi(img: ee.Image) -> ee.Image:
        return (
            img.normalizedDifference(["B3", "B11"])
            .rename("ndwi")
            .copyProperties(img, ["system:time_start"])
        )

    base = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(aoi).filterDate(start, end)
    )
    cloud_filtered = base.filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 30))
    if _collection_size(cloud_filtered) == 0:
        cloud_filtered = base.filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 80))
    return _require_images(cloud_filtered, "Sentinel-2 surface-water", start, end).map(_add_mndwi)


def fetch_rainfall_4w(
    aoi: ee.Geometry,
    end_date: str,
    scale: int = 1000,
) -> xr.Dataset:
    """
    CHIRPS 28-day cumulative rainfall ending on end_date.
    Returns Dataset with variable 'rainfall_4w'.
    """
    end_dt = ee.Date(end_date)
    start_4w = end_dt.advance(-28, "day")
    collection = _require_images(
        ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")
        .filterBounds(aoi)
        .filterDate(start_4w, end_dt)
        .select("precipitation"),
        "CHIRPS rainfall",
        cast(str, start_4w.format("YYYY-MM-dd").getInfo()),
        end_date,
    )
    chirps_4w = collection.sum().rename("rainfall_4w").clip(aoi)
    url = chirps_4w.getDownloadURL(
        {"region": aoi, "scale": scale, "crs": "EPSG:4326", "format": "GEO_TIFF"}
    )
    da = _download_band(url).squeeze()
    return xr.Dataset({"rainfall_4w": da.rename({"x": "lon", "y": "lat"})})


def fetch_lst_mean(
    aoi: ee.Geometry,
    start: str,
    end: str,
    scale: int = 1000,
) -> xr.Dataset:
    """
    MODIS Terra MOD11A2 daytime LST mean over [start, end] in °C.
    Returns Dataset with variable 'temp_mean'.
    """

    def _scale_lst(img: ee.Image) -> ee.Image:
        return img.multiply(0.02).subtract(273.15).copyProperties(img, ["system:time_start"])

    lst_col = _require_images(
        ee.ImageCollection("MODIS/061/MOD11A2")
        .filterBounds(aoi)
        .filterDate(start, end)
        .select("LST_Day_1km")
        .map(_scale_lst),
        "MODIS land-surface-temperature",
        start,
        end,
    )
    temp_mean = lst_col.mean().rename("temp_mean").clip(aoi)
    url = temp_mean.getDownloadURL(
        {"region": aoi, "scale": scale, "crs": "EPSG:4326", "format": "GEO_TIFF"}
    )
    da = _download_band(url).squeeze()
    return xr.Dataset({"temp_mean": da.rename({"x": "lon", "y": "lat"})})


def fetch_ndwi(
    aoi: ee.Geometry,
    start: str,
    end: str,
    scale: int = 1000,
) -> xr.Dataset:
    """
    Sentinel-2 MNDWI = (Green − SWIR1) / (Green + SWIR1) median composite.
    Returns Dataset with variable 'ndwi'.
    """

    s2 = _sentinel2_mndwi_collection(aoi, start, end)
    ndwi_img = s2.median().clip(aoi)
    url = ndwi_img.getDownloadURL(
        {"region": aoi, "scale": scale, "crs": "EPSG:4326", "format": "GEO_TIFF"}
    )
    da = _download_band(url).squeeze()
    return xr.Dataset({"ndwi": da.rename({"x": "lon", "y": "lat"})})


def fetch_elevation(aoi: ee.Geometry, scale: int = 1000) -> xr.Dataset:
    """USGS SRTM 30 m elevation. Returns Dataset with variable 'elevation'."""
    elev = ee.Image("USGS/SRTMGL1_003").select("elevation").clip(aoi)
    url = elev.getDownloadURL(
        {"region": aoi, "scale": scale, "crs": "EPSG:4326", "format": "GEO_TIFF"}
    )
    da = _download_band(url).squeeze()
    return xr.Dataset({"elevation": da.rename({"x": "lon", "y": "lat"})})


def fetch_pop_density(aoi: ee.Geometry, year: int = 2020, scale: int = 1000) -> xr.Dataset:
    """
    WorldPop GP 100 m population density, log-transformed: log(1 + pop).
    Returns Dataset with variable 'pop_density'.
    """
    pop_raw = (
        ee.ImageCollection("WorldPop/GP/100m/pop")
        .filterBounds(aoi)
        .filter(ee.Filter.eq("year", year))
        .first()
        .select("population")
        .clip(aoi)
    )
    pop_log = pop_raw.add(1).log().rename("pop_density")
    url = pop_log.getDownloadURL(
        {"region": aoi, "scale": scale, "crs": "EPSG:4326", "format": "GEO_TIFF"}
    )
    da = _download_band(url).squeeze()
    return xr.Dataset({"pop_density": da.rename({"x": "lon", "y": "lat"})})


def fetch_ndvi_mean(
    aoi: ee.Geometry,
    start: str,
    end: str,
    scale: int = 1000,
) -> xr.Dataset:
    """
    MODIS MOD13A3 monthly NDVI mean over [start, end] (scale factor 0.0001).
    Returns Dataset with variable 'ndvi'.
    """

    def _scale_ndvi(img: ee.Image) -> ee.Image:
        return img.multiply(0.0001).copyProperties(img, ["system:time_start"])

    ndvi_col = _require_images(
        ee.ImageCollection("MODIS/061/MOD13A3")
        .filterBounds(aoi)
        .filterDate(start, end)
        .select("NDVI")
        .map(_scale_ndvi),
        "MODIS vegetation",
        start,
        end,
    )
    ndvi_img = ndvi_col.mean().rename("ndvi").clip(aoi)
    url = ndvi_img.getDownloadURL(
        {"region": aoi, "scale": scale, "crs": "EPSG:4326", "format": "GEO_TIFF"}
    )
    da = _download_band(url).squeeze()
    return xr.Dataset({"ndvi": da.rename({"x": "lon", "y": "lat"})})


def fetch_landcover(aoi: ee.Geometry, scale: int = 1000) -> xr.Dataset:
    """
    ESA WorldCover v200 normalised to [0, 1].
    Returns Dataset with variable 'land_cover'.
    """
    lc = (
        ee.ImageCollection("ESA/WorldCover/v200")
        .first()
        .select("Map")
        .divide(100)
        .rename("land_cover")
        .clip(aoi)
    )
    url = lc.getDownloadURL(
        {"region": aoi, "scale": scale, "crs": "EPSG:4326", "format": "GEO_TIFF"}
    )
    da = _download_band(url).squeeze()
    return xr.Dataset({"land_cover": da.rename({"x": "lon", "y": "lat"})})


def build_feature_datasets(aoi: ee.Geometry, config: dict) -> dict[str, xr.Dataset]:
    """
    Download all seven disease feature bands from GEE in parallel.
    The seven fetch calls are independent HTTP requests; they run concurrently
    via DaskEngine.run_io_parallel (ThreadPoolExecutor) sharing the GEE session.
    Returns a dict keyed by band group name, consumed by cog_export.
    """
    from climate_change.core.dask_engine import DaskEngine

    scale = config.get("scale", 1000)
    start = config.get("start_date", "2021-01-01")
    end = config.get("end_date", "2023-12-31")
    start, end = _normalise_date_window(start, end)

    return DaskEngine.run_io_parallel(
        {
            "rainfall": lambda: fetch_rainfall_4w(aoi, end_date=end, scale=scale),
            "temperature": lambda: fetch_lst_mean(aoi, start, end, scale=scale),
            "ndwi": lambda: fetch_ndwi(aoi, start, end, scale=scale),
            "elevation": lambda: fetch_elevation(aoi, scale=scale),
            "population": lambda: fetch_pop_density(aoi, scale=scale),
            "ndvi": lambda: fetch_ndvi_mean(aoi, start, end, scale=scale),
            "landcover": lambda: fetch_landcover(aoi, scale=scale),
        }
    )


def build_gee_feature_stack(aoi: ee.Geometry, config: dict) -> ee.Image:
    """
    Assemble the 7-band GEE image used for stratified pixel sampling.
    Band order matches FEATURE_COLS. geometries=True preserved in sampling
    call so centroids are available for DBSCAN hotspot detection.
    """
    start = config.get("start_date", "2021-01-01")
    end = config.get("end_date", "2023-12-31")
    start, end = _normalise_date_window(start, end)
    scale = config.get("scale", 1000)

    # CHIRPS 28-day rainfall
    end_dt = ee.Date(end)
    start_4w = end_dt.advance(-28, "day")
    rain_collection = _require_images(
        ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")
        .filterBounds(aoi)
        .filterDate(start_4w, end_dt)
        .select("precipitation"),
        "CHIRPS rainfall",
        cast(str, start_4w.format("YYYY-MM-dd").getInfo()),
        end,
    )
    rain_4w = rain_collection.sum().rename("rainfall_4w").clip(aoi)

    # MODIS LST → °C
    def _scale_lst(img: ee.Image) -> ee.Image:
        return img.multiply(0.02).subtract(273.15).copyProperties(img, ["system:time_start"])

    temp_collection = _require_images(
        ee.ImageCollection("MODIS/061/MOD11A2")
        .filterBounds(aoi)
        .filterDate(start, end)
        .select("LST_Day_1km")
        .map(_scale_lst),
        "MODIS land-surface-temperature",
        start,
        end,
    )
    temp_mean = temp_collection.mean().rename("temp_mean").clip(aoi)

    # Sentinel-2 MNDWI
    def _add_mndwi(img: ee.Image) -> ee.Image:
        return (
            img.normalizedDifference(["B3", "B11"])
            .rename("ndwi")
            .copyProperties(img, ["system:time_start"])
        )

    ndwi = _sentinel2_mndwi_collection(aoi, start, end).median().clip(aoi)

    # SRTM elevation
    elevation = ee.Image("USGS/SRTMGL1_003").select("elevation").clip(aoi)

    # WorldPop log(1 + pop)
    pop_density = (
        ee.ImageCollection("WorldPop/GP/100m/pop")
        .filterBounds(aoi)
        .filter(ee.Filter.eq("year", 2020))
        .first()
        .select("population")
        .add(1)
        .log()
        .rename("pop_density")
        .clip(aoi)
    )

    # MODIS NDVI mean
    def _scale_ndvi(img: ee.Image) -> ee.Image:
        return img.multiply(0.0001).copyProperties(img, ["system:time_start"])

    ndvi_collection = _require_images(
        ee.ImageCollection("MODIS/061/MOD13A3")
        .filterBounds(aoi)
        .filterDate(start, end)
        .select("NDVI")
        .map(_scale_ndvi),
        "MODIS vegetation",
        start,
        end,
    )
    ndvi = ndvi_collection.mean().rename("ndvi").clip(aoi)

    # ESA WorldCover normalised
    land_cover = (
        ee.ImageCollection("ESA/WorldCover/v200")
        .first()
        .select("Map")
        .divide(100)
        .rename("land_cover")
        .clip(aoi)
        .reproject("EPSG:4326", None, scale)
    )

    return ee.Image.cat([rain_4w, temp_mean, ndwi, elevation, pop_density, ndvi, land_cover])


def _fetch_ndvi_monthly(aoi: ee.Geometry, start: str, end: str) -> pd.DataFrame:
    """Monthly area-mean MODIS NDVI over [start, end]."""

    def _scale_ndvi(img: ee.Image) -> ee.Image:
        return img.multiply(0.0001).copyProperties(img, ["system:time_start"])

    def _ndvi_mean(img: ee.Image) -> ee.Feature:
        v = img.reduceRegion(ee.Reducer.mean(), aoi, 1000, maxPixels=int(1e9)).get("NDVI")
        return ee.Feature(None, {"date": img.date().format("YYYY-MM"), "ndvi": v})

    ndvi_col = (
        ee.ImageCollection("MODIS/061/MOD13A3")
        .filterBounds(aoi)
        .filterDate(start, end)
        .select("NDVI")
        .map(_scale_ndvi)
    )
    records = cast(dict, ee.FeatureCollection(ndvi_col.map(_ndvi_mean)).getInfo())["features"]
    return (
        pd.DataFrame([f["properties"] for f in records])
        .dropna()
        .groupby("date")[["ndvi"]]
        .mean()
        .sort_index()
    )


def _fetch_rain_monthly(aoi: ee.Geometry, start: str, end: str) -> pd.DataFrame:
    """Monthly area-mean CHIRPS rainfall over [start, end]."""
    n_months = int(cast(int, ee.Date(end).difference(ee.Date(start), "month").round().getInfo()))
    months = ee.List.sequence(0, n_months - 1)

    def _monthly_rain(offset: ee.Number) -> ee.Feature:
        s = ee.Date(start).advance(offset, "month")
        e = s.advance(1, "month")
        v = (
            ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")
            .filterBounds(aoi)
            .filterDate(s, e)
            .select("precipitation")
            .sum()
            .reduceRegion(ee.Reducer.mean(), aoi, 1000, maxPixels=int(1e9))
            .get("precipitation", 0)
        )
        return ee.Feature(None, {"date": s.format("YYYY-MM"), "rain_mm": v})

    records = cast(dict, ee.FeatureCollection(months.map(_monthly_rain)).getInfo())["features"]
    return (
        pd.DataFrame([f["properties"] for f in records])
        .dropna()
        .groupby("date")[["rain_mm"]]
        .mean()
        .sort_index()
    )


def _fetch_lst_monthly(aoi: ee.Geometry, start: str, end: str) -> pd.DataFrame:
    """Monthly area-mean MODIS LST (°C) over [start, end]."""

    def _scale_lst(img: ee.Image) -> ee.Image:
        return img.multiply(0.02).subtract(273.15).copyProperties(img, ["system:time_start"])

    def _lst_mean(img: ee.Image) -> ee.Feature:
        v = img.reduceRegion(ee.Reducer.mean(), aoi, 1000, maxPixels=int(1e9)).get("LST_Day_1km")
        return ee.Feature(None, {"date": img.date().format("YYYY-MM"), "lst": v})

    lst_col = (
        ee.ImageCollection("MODIS/061/MOD11A2")
        .filterBounds(aoi)
        .filterDate(start, end)
        .select("LST_Day_1km")
        .map(_scale_lst)
    )
    records = cast(dict, lst_col.map(_lst_mean).getInfo())["features"]
    return (
        pd.DataFrame([f["properties"] for f in records])
        .dropna()
        .groupby("date")[["lst"]]
        .mean()
        .sort_index()
    )


def fetch_monthly_timeseries(
    aoi: ee.Geometry,
    start: str,
    end: str,
) -> dict[str, pd.DataFrame]:
    """
    Fetch monthly area-mean NDVI, rainfall, and LST time series in parallel.
    The three GEE queries are independent and run concurrently via
    DaskEngine.run_io_parallel (ThreadPoolExecutor) sharing the GEE session.
    Returns dict keyed by variable name with a DatetimeIndex DataFrame.
    """
    from climate_change.core.dask_engine import DaskEngine

    return DaskEngine.run_io_parallel(
        {
            "ndvi": lambda: _fetch_ndvi_monthly(aoi, start, end),
            "rain": lambda: _fetch_rain_monthly(aoi, start, end),
            "lst": lambda: _fetch_lst_monthly(aoi, start, end),
        }
    )


def _compute_risk_score(df: pd.DataFrame) -> np.ndarray:
    """
    Composite malaria environmental suitability score (0–100).

    score = 0.40 × TempSuit + 0.35 × RainSuit + 0.25 × NDWIScore

    TempSuit: Gaussian centred at 27.5 °C (σ=4) — Bayoh & Lindsay 2004 model
    RainSuit: clip(rain_4w / 100, 0, 1) × 100
    NDWIScore: clip((ndwi + 0.5) / 1.0, 0, 1) × 100
    """
    temp_suit = np.exp(-0.5 * ((df["temp_mean"].to_numpy() - 27.5) / 4.0) ** 2) * 100
    rain_suit = np.clip(df["rainfall_4w"].to_numpy() / 100, 0, 1) * 100
    ndwi_score = np.clip((df["ndwi"].to_numpy() + 0.5) / 1.0, 0, 1) * 100

    return (
        SCORE_WEIGHTS["temp_suit"] * temp_suit
        + SCORE_WEIGHTS["rain_suit"] * rain_suit
        + SCORE_WEIGHTS["ndwi_score"] * ndwi_score
    )


def sample_training_data(
    feature_stack: ee.Image,
    aoi: ee.Geometry,
    n_pixels: int = 3000,
    scale: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Sample n_pixels from the GEE feature stack and assign 3-class disease risk labels.
    Labels: tercile thresholds on the composite risk score.
      0 = Low Risk  (bottom 1/3)
      1 = Medium Risk (middle 1/3)
      2 = High Risk (top 1/3)
    Returns DataFrame with FEATURE_COLS + ['lon', 'lat', 'risk_score', 'label'].
    """
    samples = feature_stack.sample(
        region=aoi,
        scale=scale,
        numPixels=n_pixels,
        seed=seed,
        geometries=True,
        dropNulls=False,
    )
    sample_list = cast(dict, samples.getInfo())["features"]
    df = (
        pd.DataFrame(
            [
                {
                    **f["properties"],
                    "lon": f["geometry"]["coordinates"][0] if f.get("geometry") else None,
                    "lat": f["geometry"]["coordinates"][1] if f.get("geometry") else None,
                }
                for f in sample_list
            ]
        )
        .dropna(subset=FEATURE_COLS + ["lon", "lat"])[FEATURE_COLS + ["lon", "lat"]]
        .reset_index(drop=True)
    )

    scores = _compute_risk_score(df)
    t33 = float(np.percentile(scores, RISK_PERCENTILES[0] * 100))
    t66 = float(np.percentile(scores, RISK_PERCENTILES[1] * 100))
    labels = np.zeros(len(df), dtype=np.intp)
    labels[scores >= t33] = 1
    labels[scores >= t66] = 2

    df["risk_score"] = scores
    df["label"] = labels
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
    ref_key: str = "elevation",
    method_continuous: InterpOptions = "linear",
    method_categorical: InterpOptions = "nearest",
) -> dict[str, xr.Dataset]:
    """
    Interpolate all datasets onto the reference grid.
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
