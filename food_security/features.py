from __future__ import annotations

import io
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
    "vci",  # Vegetation Condition Index (0–100)
    "tci",  # Temperature Condition Index (0–100)
    "rainfall_anom_pct",  # CHIRPS rainfall anomaly vs LT baseline (%)
    "ndvi_slope",  # Long-term MODIS NDVI trend (linearFit, NDVI/yr)
    "mndwi",  # Sentinel-2 MNDWI — surface water availability
    "slope_terrain",  # SRTM terrain slope (degrees)
    "land_cover",  # ESA WorldCover normalised (0–1)
]

# Food insecurity risk class definitions (3-class composite label)
FOOD_CLASSES: list[str] = ["Low Risk", "Medium Risk", "High Risk"]
FOOD_COLORS: list[str] = ["#184c09", "#ffcc36", "#f22d06"]

# Composite food stress score weights (must sum to 1.0)
SCORE_WEIGHTS: dict[str, float] = {
    "vci_stress": 0.40,  # (100 − VCI): low vegetation condition = high stress
    "tci_stress": 0.25,  # (100 − TCI): heat stress
    "rain_deficit": 0.20,  # clip(−rainfall_anom_pct, 0, 100): rainfall deficit
    "slope_inv": 0.15,  # inverted normalised NDVI slope: declining = high stress
}

# Long-term baseline end date (used for VCI/TCI min-max computation)
LT_BASELINE_START = "2001-01-01"

# Tercile thresholds for labelling
RISK_PERCENTILES: tuple[float, float] = (1 / 3, 2 / 3)


def fetch_vci_tci(
    aoi: ee.Geometry,
    lt_start: str,
    study_start: str,
    study_end: str,
    scale: int = 1000,
) -> xr.Dataset:
    """
    Compute VCI and TCI from MODIS NDVI and LST relative to the long-term baseline.

    VCI = (NDVI_current − NDVI_min) / (NDVI_max − NDVI_min) × 100
    TCI = (LST_max − LST_current) / (LST_max − LST_min) × 100

    Returns Dataset with variables 'vci' and 'tci'.
    """

    def _scale_ndvi(img: ee.Image) -> ee.Image:
        return img.multiply(0.0001).copyProperties(img, ["system:time_start"])

    def _scale_lst(img: ee.Image) -> ee.Image:
        return img.multiply(0.02).subtract(273.15).copyProperties(img, ["system:time_start"])

    # Long-term NDVI baseline
    modis_lt = (
        ee.ImageCollection("MODIS/061/MOD13A3")
        .filterBounds(aoi)
        .filterDate(lt_start, study_start)
        .select("NDVI")
        .map(_scale_ndvi)
    )
    ndvi_min = modis_lt.min().rename("ndvi_min")
    ndvi_max = modis_lt.max().rename("ndvi_max")

    # Study-period NDVI current
    ndvi_current = (
        ee.ImageCollection("MODIS/061/MOD13A3")
        .filterBounds(aoi)
        .filterDate(study_start, study_end)
        .select("NDVI")
        .map(_scale_ndvi)
        .mean()
        .rename("ndvi_current")
    )
    vci_img = (
        ndvi_current.subtract(ndvi_min)
        .divide(ndvi_max.subtract(ndvi_min).add(1e-6))
        .multiply(100)
        .rename("vci")
        .clip(aoi)
    )

    # Long-term LST baseline (10th and 90th percentiles)
    lst_lt = (
        ee.ImageCollection("MODIS/061/MOD11A2")
        .filterBounds(aoi)
        .filterDate(lt_start, study_end)
        .select("LST_Day_1km")
        .map(_scale_lst)
    )
    lst_min = lst_lt.reduce(ee.Reducer.percentile([10])).rename("lst_min")
    lst_max = lst_lt.reduce(ee.Reducer.percentile([90])).rename("lst_max")

    # Study-period LST current
    lst_current = (
        ee.ImageCollection("MODIS/061/MOD11A2")
        .filterBounds(aoi)
        .filterDate(study_start, study_end)
        .select("LST_Day_1km")
        .map(_scale_lst)
        .mean()
        .rename("lst_current")
    )
    tci_img = (
        lst_max.subtract(lst_current)
        .divide(lst_max.subtract(lst_min).add(1e-6))
        .multiply(100)
        .rename("tci")
        .clip(aoi)
    )

    stack = vci_img.addBands(tci_img)
    url = stack.getDownloadURL(
        {"region": aoi, "scale": scale, "crs": "EPSG:4326", "format": "GEO_TIFF"}
    )
    raw = _download_band(url)
    raw = raw.assign_coords(band=["vci", "tci"])
    return raw.to_dataset(dim="band").rename({"x": "lon", "y": "lat"})


def fetch_rainfall_anomaly_pct(
    aoi: ee.Geometry,
    lt_start: str,
    study_start: str,
    study_end: str,
    scale: int = 1000,
) -> xr.Dataset:
    """
    Pixel-wise CHIRPS rainfall anomaly as percentage vs long-term daily mean.
    anomaly_pct = (study_mean − LT_mean) / (LT_mean + ε) × 100

    Returns Dataset with variable 'rainfall_anom_pct'.
    """
    chirps = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY").filterBounds(aoi)
    lt_mean = (
        chirps.filterDate(lt_start, study_start)
        .select("precipitation")
        .mean()
        .multiply(365)
        .rename("lt_mean")
    )
    study_mean = (
        chirps.filterDate(study_start, study_end)
        .select("precipitation")
        .mean()
        .multiply(365)
        .rename("study_mean")
    )
    rain_anom = (
        study_mean.subtract(lt_mean)
        .divide(lt_mean.add(1e-6))
        .multiply(100)
        .rename("rainfall_anom_pct")
        .clip(aoi)
    )
    url = rain_anom.getDownloadURL(
        {"region": aoi, "scale": scale, "crs": "EPSG:4326", "format": "GEO_TIFF"}
    )
    da = _download_band(url).squeeze()
    return xr.Dataset({"rainfall_anom_pct": da.rename({"x": "lon", "y": "lat"})})


def fetch_ndvi_slope_img(
    aoi: ee.Geometry,
    lt_start: str,
    study_end: str,
    scale: int = 1000,
) -> xr.Dataset:
    """
    Pixel-wise MODIS NDVI linear trend (NDVI per year) from lt_start to study_end.
    Returns Dataset with variable 'ndvi_slope'.
    """

    def _scale_ndvi(img: ee.Image) -> ee.Image:
        return img.multiply(0.0001).copyProperties(img, ["system:time_start"])

    def _add_time(img: ee.Image) -> ee.Image:
        t = img.date().difference(ee.Date(lt_start), "year")
        return img.addBands(ee.Image(t).rename("time").float())

    modis = (
        ee.ImageCollection("MODIS/061/MOD13A3")
        .filterBounds(aoi)
        .filterDate(lt_start, study_end)
        .select("NDVI")
        .map(_scale_ndvi)
    )
    trend = modis.map(_add_time).select(["time", "NDVI"]).reduce(ee.Reducer.linearFit())
    ndvi_slope = trend.select("scale").rename("ndvi_slope").clip(aoi)

    url = ndvi_slope.getDownloadURL(
        {"region": aoi, "scale": scale, "crs": "EPSG:4326", "format": "GEO_TIFF"}
    )
    da = _download_band(url).squeeze()
    return xr.Dataset({"ndvi_slope": da.rename({"x": "lon", "y": "lat"})})


def fetch_mndwi(
    aoi: ee.Geometry,
    start: str,
    end: str,
    scale: int = 1000,
) -> xr.Dataset:
    """
    Sentinel-2 MNDWI = (Green − SWIR1) / (Green + SWIR1) median composite.
    Returns Dataset with variable 'mndwi'.
    """

    def _add_mndwi(img: ee.Image) -> ee.Image:
        return (
            img.normalizedDifference(["B3", "B11"])
            .rename("mndwi")
            .copyProperties(img, ["system:time_start"])
        )

    s2 = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(aoi)
        .filterDate(start, end)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 30))
        .map(_add_mndwi)
    )
    mndwi_img = s2.median().clip(aoi)
    url = mndwi_img.getDownloadURL(
        {"region": aoi, "scale": scale, "crs": "EPSG:4326", "format": "GEO_TIFF"}
    )
    da = _download_band(url).squeeze()
    return xr.Dataset({"mndwi": da.rename({"x": "lon", "y": "lat"})})


def fetch_terrain_slope(aoi: ee.Geometry, scale: int = 1000) -> xr.Dataset:
    """SRTM-derived terrain slope in degrees. Returns Dataset with variable 'slope_terrain'."""
    slope = ee.Terrain.slope(ee.Image("USGS/SRTMGL1_003")).clip(aoi).rename("slope_terrain")
    url = slope.getDownloadURL(
        {"region": aoi, "scale": scale, "crs": "EPSG:4326", "format": "GEO_TIFF"}
    )
    da = _download_band(url).squeeze()
    return xr.Dataset({"slope_terrain": da.rename({"x": "lon", "y": "lat"})})


def fetch_landcover(aoi: ee.Geometry, scale: int = 1000) -> xr.Dataset:
    """
    ESA WorldCover v200 normalised to [0, 1] (class value / 100).
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
    Download all seven food-security feature bands from GEE in parallel.
    The six fetch calls are independent HTTP requests; they run concurrently
    via DaskEngine.run_io_parallel (ThreadPoolExecutor) sharing the GEE session.
    Returns a dict keyed by band group name, consumed by cog_export.
    """
    from climate_change.core.dask_engine import DaskEngine

    scale = config.get("scale", 1000)
    study_start = config.get("start_date", "2018-01-01")
    study_end = config.get("end_date", "2023-12-31")
    lt_start = config.get("lt_baseline_start", LT_BASELINE_START)
    s2_start = config.get("s2_start", "2020-01-01")

    return DaskEngine.run_io_parallel(
        {
            "vci_tci": lambda: fetch_vci_tci(aoi, lt_start, study_start, study_end, scale=scale),
            "rainfall": lambda: fetch_rainfall_anomaly_pct(
                aoi, lt_start, study_start, study_end, scale=scale
            ),
            "ndvi_slope": lambda: fetch_ndvi_slope_img(aoi, lt_start, study_end, scale=scale),
            "mndwi": lambda: fetch_mndwi(aoi, s2_start, study_end, scale=scale),
            "terrain": lambda: fetch_terrain_slope(aoi, scale=scale),
            "landcover": lambda: fetch_landcover(aoi, scale=scale),
        }
    )


def build_gee_feature_stack(aoi: ee.Geometry, config: dict) -> ee.Image:
    """
    Assemble the 7-band GEE image used for stratified pixel sampling.
    Band order matches FEATURE_COLS.
    """
    lt_start = config.get("lt_baseline_start", LT_BASELINE_START)
    study_start = config.get("start_date", "2018-01-01")
    study_end = config.get("end_date", "2023-12-31")
    s2_start = config.get("s2_start", "2020-01-01")
    scale = config.get("scale", 1000)

    def _scale_ndvi(img: ee.Image) -> ee.Image:
        return img.multiply(0.0001).copyProperties(img, ["system:time_start"])

    def _scale_lst(img: ee.Image) -> ee.Image:
        return img.multiply(0.02).subtract(273.15).copyProperties(img, ["system:time_start"])

    def _add_time(img: ee.Image) -> ee.Image:
        t = img.date().difference(ee.Date(lt_start), "year")
        return img.addBands(ee.Image(t).rename("time").float())

    # NDVI long-term min/max for VCI
    modis_lt = (
        ee.ImageCollection("MODIS/061/MOD13A3")
        .filterBounds(aoi)
        .filterDate(lt_start, study_start)
        .select("NDVI")
        .map(_scale_ndvi)
    )
    ndvi_min = modis_lt.min()
    ndvi_max = modis_lt.max()
    ndvi_current = (
        ee.ImageCollection("MODIS/061/MOD13A3")
        .filterBounds(aoi)
        .filterDate(study_start, study_end)
        .select("NDVI")
        .map(_scale_ndvi)
        .mean()
    )
    vci = (
        ndvi_current.subtract(ndvi_min)
        .divide(ndvi_max.subtract(ndvi_min).add(1e-6))
        .multiply(100)
        .rename("vci")
        .clip(aoi)
    )

    # LST long-term baseline for TCI
    lst_lt = (
        ee.ImageCollection("MODIS/061/MOD11A2")
        .filterBounds(aoi)
        .filterDate(lt_start, study_end)
        .select("LST_Day_1km")
        .map(_scale_lst)
    )
    lst_min = lst_lt.reduce(ee.Reducer.percentile([10]))
    lst_max = lst_lt.reduce(ee.Reducer.percentile([90]))
    lst_current = (
        ee.ImageCollection("MODIS/061/MOD11A2")
        .filterBounds(aoi)
        .filterDate(study_start, study_end)
        .select("LST_Day_1km")
        .map(_scale_lst)
        .mean()
    )
    tci = (
        lst_max.subtract(lst_current)
        .divide(lst_max.subtract(lst_min).add(1e-6))
        .multiply(100)
        .rename("tci")
        .clip(aoi)
    )

    # CHIRPS rainfall anomaly %
    chirps = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY").filterBounds(aoi)
    lt_rain_mean = (
        chirps.filterDate(lt_start, study_start).select("precipitation").mean().multiply(365)
    )
    study_rain_mean = (
        chirps.filterDate(study_start, study_end).select("precipitation").mean().multiply(365)
    )
    rain_anom = (
        study_rain_mean.subtract(lt_rain_mean)
        .divide(lt_rain_mean.add(1e-6))
        .multiply(100)
        .rename("rainfall_anom_pct")
        .clip(aoi)
    )

    # MODIS NDVI linearFit slope
    modis_full = (
        ee.ImageCollection("MODIS/061/MOD13A3")
        .filterBounds(aoi)
        .filterDate(lt_start, study_end)
        .select("NDVI")
        .map(_scale_ndvi)
    )
    trend = modis_full.map(_add_time).select(["time", "NDVI"]).reduce(ee.Reducer.linearFit())
    ndvi_slope = trend.select("scale").rename("ndvi_slope").clip(aoi)

    # Sentinel-2 MNDWI
    def _add_mndwi(img: ee.Image) -> ee.Image:
        return (
            img.normalizedDifference(["B3", "B11"])
            .rename("mndwi")
            .copyProperties(img, ["system:time_start"])
        )

    mndwi = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(aoi)
        .filterDate(s2_start, study_end)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 30))
        .map(_add_mndwi)
        .median()
        .clip(aoi)
    )

    # SRTM slope
    slope = (
        ee.Terrain.slope(ee.Image("USGS/SRTMGL1_003"))
        .clip(aoi)
        .reproject("EPSG:4326", None, scale)
        .rename("slope_terrain")
    )

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

    return ee.Image.cat([vci, tci, rain_anom, ndvi_slope, mndwi, slope, land_cover])


def fetch_ndvi_monthly_timeseries(
    aoi: ee.Geometry,
    start: str,
    end: str,
) -> pd.DataFrame:
    """
    Monthly area-mean MODIS NDVI over [start, end].
    Returns DataFrame with DatetimeIndex and column 'ndvi'.
    """

    def _scale_ndvi(img: ee.Image) -> ee.Image:
        return img.multiply(0.0001).copyProperties(img, ["system:time_start"])

    def _mean_feat(img: ee.Image) -> ee.Feature:
        v = img.reduceRegion(ee.Reducer.mean(), aoi, 1000, maxPixels=int(1e9)).get("NDVI")
        return ee.Feature(None, {"date": img.date().format("YYYY-MM"), "ndvi": v})

    modis = (
        ee.ImageCollection("MODIS/061/MOD13A3")
        .filterBounds(aoi)
        .filterDate(start, end)
        .select("NDVI")
        .map(_scale_ndvi)
    )
    records = cast(dict, ee.FeatureCollection(modis.map(_mean_feat)).getInfo())["features"]
    return (
        pd.DataFrame([f["properties"] for f in records])
        .dropna()
        .groupby("date")[["ndvi"]]
        .mean()
        .sort_index()
    )


def fetch_monthly_rainfall(
    aoi: ee.Geometry,
    start: str,
    end: str,
) -> pd.DataFrame:
    """
    Monthly area-mean CHIRPS rainfall sum over [start, end].
    Returns DataFrame with DatetimeIndex and column 'rain_mm'.
    """
    n_months = int(cast(int, ee.Date(end).difference(ee.Date(start), "month").round().getInfo()))
    months = ee.List.sequence(0, n_months - 1)

    def _monthly_sum(offset: ee.Number) -> ee.Feature:
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

    records = cast(dict, ee.FeatureCollection(months.map(_monthly_sum)).getInfo())["features"]
    return (
        pd.DataFrame([f["properties"] for f in records])
        .dropna()
        .groupby("date")[["rain_mm"]]
        .mean()
        .sort_index()
    )


def _compute_food_score(df: pd.DataFrame) -> np.ndarray:
    """
    Composite food stress score (0–100):
      score = 0.40 × (100 − VCI)
            + 0.25 × (100 − TCI)
            + 0.20 × clip(−rainfall_anom_pct, 0, 100)
            + 0.15 × inverted_normalised_ndvi_slope

    Higher score → greater food insecurity stress.
    """
    vci_stress = 100 - df["vci"].to_numpy()
    tci_stress = 100 - df["tci"].to_numpy()
    rain_deficit = np.clip(-df["rainfall_anom_pct"].to_numpy(), 0, 100)

    slope_arr = df["ndvi_slope"].to_numpy()
    slope_min, slope_max = slope_arr.min(), slope_arr.max()
    slope_norm = (slope_arr - slope_min) / (slope_max - slope_min + 1e-8) * 100
    slope_inv = 100 - slope_norm  # declining NDVI → high score

    return (
        SCORE_WEIGHTS["vci_stress"] * vci_stress
        + SCORE_WEIGHTS["tci_stress"] * tci_stress
        + SCORE_WEIGHTS["rain_deficit"] * rain_deficit
        + SCORE_WEIGHTS["slope_inv"] * slope_inv
    )


def sample_training_data(
    feature_stack: ee.Image,
    aoi: ee.Geometry,
    n_pixels: int = 3000,
    scale: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Sample n_pixels from the GEE feature stack and assign 3-class food insecurity labels.
    Labels: tercile thresholds on the composite food stress score.
      0 = Low Risk  (bottom 1/3)
      1 = Medium Risk (middle 1/3)
      2 = High Risk (top 1/3)
    Returns DataFrame with FEATURE_COLS + ['food_score', 'label'].
    """
    samples = feature_stack.sample(
        region=aoi,
        scale=scale,
        numPixels=n_pixels,
        seed=seed,
        geometries=False,
        dropNulls=False,
    )
    records = cast(dict, samples.getInfo())["features"]
    df = (
        pd.DataFrame([f["properties"] for f in records])
        .dropna(subset=FEATURE_COLS)[FEATURE_COLS]
        .reset_index(drop=True)
    )

    scores = _compute_food_score(df)
    t33 = float(np.percentile(scores, RISK_PERCENTILES[0] * 100))
    t66 = float(np.percentile(scores, RISK_PERCENTILES[1] * 100))
    labels = np.zeros(len(df), dtype=np.intp)
    labels[scores >= t33] = 1
    labels[scores >= t66] = 2

    df["food_score"] = scores
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
    ref_key: str = "vci_tci",
    method_continuous: InterpOptions = "linear",
    method_categorical: InterpOptions = "nearest",
) -> dict[str, xr.Dataset]:
    """
    Interpolate all datasets onto the VCI/TCI reference grid.
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
