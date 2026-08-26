from __future__ import annotations

import logging
from pathlib import Path
from typing import TypedDict

import numpy as np
import rasterio
import xarray as xr
from rasterio.features import geometry_mask
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import Booster, DMatrix

from climate_change.core.landcover_mask import WATER_CLASS
from climate_change.core.population import population_exposure

from .features import FEATURE_COLS, align_datasets
from .model import pad_gbm_proba

_log = logging.getLogger(__name__)


class DiseaseCogResult(TypedDict):
    disease_risk: str
    risk_pct: list[float]
    population_by_class: dict[str, float]
    total_population: float | None


# fetch_landcover() normalises ESA WorldCover class codes to [0, 1] (code / 100)
_WATER_LAND_COVER = WATER_CLASS / 100

# Integer encoding for COG (0 = nodata)
RISK_INT: dict[int, str] = {1: "Low Risk", 2: "Medium Risk", 3: "High Risk"}


def export_disease_cog(
    gbm_model: GradientBoostingClassifier | None,
    xgb_model: Booster | None,
    scaler: StandardScaler,
    datasets: dict[str, xr.Dataset],
    output_dir: str,
    prefix: str,
    model_type: str = "gbm",
    aoi_geojson: dict | None = None,
) -> DiseaseCogResult:
    """
    Build the full-AOI feature matrix from in-memory xarray Datasets,
    run inference with the selected model, classify into 3 risk classes,
    and write a Cloud Optimised GeoTIFF.

    Parameters
    ----------
    gbm_model   : trained GradientBoostingClassifier
    xgb_model   : trained XGBoost Booster
    scaler      : fitted StandardScaler (from DiseaseModel.scaler)
    datasets    : dict returned by build_feature_datasets
    output_dir  : local directory for COG output
    prefix      : file prefix
    model_type  : "gbm" | "xgboost" | "ensemble"

    Returns
    -------
    dict with keys:
      'disease_risk' → path string
      'risk_pct'     → [low_pct, medium_pct, high_pct] over the AOI's valid
                       (non-nodata) pixels, matching DISEASE_CLASSES order.
                       This reflects every pixel actually painted on the map,
                       unlike the training-sample-based stats DiseaseModel.predict()
                       computes, which only covers the few hundred sampled points
                       and can diverge sharply from the full-AOI distribution.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    cog_path = output_path / f"{prefix}_disease_risk.tif"

    # Align all bands to the elevation reference grid
    aligned = align_datasets(datasets, ref_key="elevation")
    elev_ds = aligned["elevation"]
    ref_lat = elev_ds.lat
    ref_lon = elev_ds.lon
    n_lat, n_lon = len(ref_lat), len(ref_lon)

    # Extract feature arrays in FEATURE_COLS order
    rain4w = aligned["rainfall"]["rainfall_4w"].values.astype(np.float32)
    temp = aligned["temperature"]["temp_mean"].values.astype(np.float32)
    ndwi = aligned["ndwi"]["ndwi"].values.astype(np.float32)
    elev = elev_ds["elevation"].values.astype(np.float32)
    pop = aligned["population"]["pop_density"].values.astype(np.float32)
    ndvi = aligned["ndvi"]["ndvi"].values.astype(np.float32)
    lc = aligned["landcover"]["land_cover"].values.astype(np.float32)

    bands = np.stack([rain4w, temp, ndwi, elev, pop, ndvi, lc], axis=0)  # (7, H, W)
    X_full = bands.reshape(len(FEATURE_COLS), -1).T  # (n_pixels, 7)
    valid_mask = ~np.isnan(X_full).any(axis=1)
    # Permanent water bodies are excluded from the disease-risk output.
    water = np.isclose(lc.reshape(-1), _WATER_LAND_COVER)
    valid_mask &= ~water
    X_valid = scaler.transform(X_full[valid_mask])

    # Inference
    if model_type == "gbm":
        assert gbm_model is not None
        proba = gbm_model.predict_proba(X_valid)
    elif model_type == "xgboost":
        assert xgb_model is not None
        proba = xgb_model.predict(DMatrix(X_valid))
    else:  # ensemble
        assert gbm_model is not None
        assert xgb_model is not None
        proba_gbm = pad_gbm_proba(gbm_model, gbm_model.predict_proba(X_valid))
        proba = (proba_gbm + xgb_model.predict(DMatrix(X_valid))) / 2.0

    pred_classes = np.argmax(proba, axis=1).astype(np.uint8) + 1  # 1-indexed

    transform = elev_ds["elevation"].rio.transform()
    crs = elev_ds["elevation"].rio.crs or "EPSG:4326"
    risk_grid: np.ndarray = np.zeros(n_lat * n_lon, dtype=np.uint8)
    risk_grid[valid_mask] = pred_classes
    risk_grid = risk_grid.reshape(n_lat, n_lon)
    if aoi_geojson:
        inside_aoi = geometry_mask(
            [aoi_geojson],
            out_shape=(n_lat, n_lon),
            transform=transform,
            invert=True,
        )
        risk_grid[~inside_aoi] = 0

    population_by_class: dict[str, float] = {}
    total_population: float | None = None
    pop_ds = aligned.get("population_count")
    if pop_ds is not None:
        by_int = population_exposure(risk_grid, transform, pop_ds, classes=[1, 2, 3])
        population_by_class = {RISK_INT[v]: by_int[str(v)] for v in (1, 2, 3)}
        total_population = round(sum(population_by_class.values()), 1)

    with rasterio.open(
        cog_path,
        "w",
        driver="COG",
        height=n_lat,
        width=n_lon,
        count=1,
        dtype=np.uint8,
        crs=crs,
        transform=transform,
        nodata=0,
        compress="lzw",
    ) as dst:
        dst.write(risk_grid, 1)
        dst.update_tags(
            RISK_CLASSES="0=nodata,1=Low Risk,2=Medium Risk,3=High Risk",
            MODEL_TYPE=model_type,
        )

    _log.info("COG exported: %s  (%d×%d px)", cog_path, n_lat, n_lon)

    class_counts = np.array([(risk_grid == v).sum() for v in (1, 2, 3)], dtype=np.float64)
    n_valid = class_counts.sum()
    risk_pct = (class_counts / n_valid * 100).round(1).tolist() if n_valid > 0 else [0.0, 0.0, 0.0]

    return {
        "disease_risk": str(cog_path),
        "risk_pct": risk_pct,
        "population_by_class": population_by_class,
        "total_population": total_population,
    }
