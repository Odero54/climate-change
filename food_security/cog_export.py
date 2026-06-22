from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import rasterio
import xarray as xr
from rasterio.features import geometry_mask
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from climate_change.core.base_use_case import _aoi_geometries

from .features import FEATURE_COLS, align_datasets

_log = logging.getLogger(__name__)

# Integer encoding for COG (0 = nodata)
RISK_INT: dict[int, str] = {1: "Low Risk", 2: "Medium Risk", 3: "High Risk"}


def predict_food_security_grid(
    rf_model: RandomForestClassifier,
    xgb_model: XGBClassifier,
    scaler: StandardScaler,
    datasets: dict[str, xr.Dataset],
    model_type: str = "rf",
    aoi_geojson: dict | None = None,
) -> dict:
    aligned = align_datasets(datasets, ref_key="vci_tci")
    vci_tci_ds = aligned["vci_tci"]
    ref_lat = vci_tci_ds.lat
    ref_lon = vci_tci_ds.lon
    n_lat, n_lon = len(ref_lat), len(ref_lon)

    vci = vci_tci_ds["vci"].values.astype(np.float32)
    tci = vci_tci_ds["tci"].values.astype(np.float32)
    rain_anom = aligned["rainfall"]["rainfall_anom_pct"].values.astype(np.float32)
    ndvi_slope = aligned["ndvi_slope"]["ndvi_slope"].values.astype(np.float32)
    mndwi = aligned["mndwi"]["mndwi"].values.astype(np.float32)
    slope = aligned["terrain"]["slope_terrain"].values.astype(np.float32)
    lc = aligned["landcover"]["land_cover"].values.astype(np.float32)

    bands = np.stack([vci, tci, rain_anom, ndvi_slope, mndwi, slope, lc], axis=0)
    X_full = bands.reshape(len(FEATURE_COLS), -1).T
    valid_mask = ~np.isnan(X_full).any(axis=1)

    transform = vci_tci_ds["vci"].rio.transform()
    crs = vci_tci_ds["vci"].rio.crs or "EPSG:4326"
    geometries = _aoi_geometries(aoi_geojson)
    if geometries:
        inside_aoi = geometry_mask(
            geometries,
            out_shape=(n_lat, n_lon),
            transform=transform,
            invert=True,
        ).reshape(-1)
        valid_mask = valid_mask & inside_aoi

    risk_grid = np.zeros(n_lat * n_lon, dtype=np.uint8)
    if valid_mask.any():
        X_valid = scaler.transform(X_full[valid_mask])
        if model_type == "rf":
            proba = rf_model.predict_proba(X_valid)
        elif model_type == "xgboost":
            proba = xgb_model.predict_proba(X_valid)
        else:
            proba = (
                rf_model.predict_proba(X_valid) + xgb_model.predict_proba(X_valid)
            ) / 2.0
        pred_classes = np.argmax(proba, axis=1).astype(np.uint8) + 1
        risk_grid[valid_mask] = pred_classes

    risk_grid = risk_grid.reshape(n_lat, n_lon)
    valid_values = risk_grid[risk_grid > 0]
    counts = np.array(
        [(valid_values == class_id).sum() for class_id in (1, 2, 3)],
        dtype=np.float64,
    )
    total = int(valid_values.size)
    percentages = (counts / total * 100).round(1).tolist() if total else [0.0, 0.0, 0.0]

    return {
        "risk_grid": risk_grid,
        "transform": transform,
        "crs": crs,
        "height": n_lat,
        "width": n_lon,
        "counts": counts.astype(int).tolist(),
        "percentages": percentages,
        "valid_pixel_count": total,
    }


def export_food_security_cog(
    rf_model: RandomForestClassifier,
    xgb_model: XGBClassifier,
    scaler: StandardScaler,
    datasets: dict[str, xr.Dataset],
    output_dir: str,
    prefix: str,
    model_type: str = "rf",
    aoi_geojson: dict | None = None,
) -> dict[str, str]:
    """
    Build the full-AOI feature matrix from in-memory xarray Datasets,
    run inference with the selected model, classify into 3 food insecurity risk classes,
    and write a Cloud Optimised GeoTIFF.

    Parameters
    ----------
    rf_model    : trained RandomForestClassifier
    xgb_model   : trained XGBClassifier
    scaler      : fitted StandardScaler (from FoodSecurityModel.scaler)
    datasets    : dict returned by build_feature_datasets
    output_dir  : local directory for COG output
    prefix      : file prefix
    model_type  : "rf" | "xgboost" | "ensemble"
    aoi_geojson : optional AOI polygon used to mask pixels outside the true AOI

    Returns
    -------
    dict with key 'food_security_risk' → path string
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    cog_path = output_path / f"{prefix}_food_security_risk.tif"

    grid = predict_food_security_grid(
        rf_model,
        xgb_model,
        scaler,
        datasets,
        model_type=model_type,
        aoi_geojson=aoi_geojson,
    )
    risk_grid = grid["risk_grid"]

    with rasterio.open(
        cog_path,
        "w",
        driver="COG",
        height=grid["height"],
        width=grid["width"],
        count=1,
        dtype=np.uint8,
        crs=grid["crs"],
        transform=grid["transform"],
        nodata=0,
        compress="lzw",
    ) as dst:
        dst.write(risk_grid, 1)
        dst.update_tags(
            RISK_CLASSES="0=nodata,1=Low Risk,2=Medium Risk,3=High Risk",
            MODEL_TYPE=model_type,
        )

    _log.info("COG exported: %s  (%d×%d px)", cog_path, grid["height"], grid["width"])
    return {"food_security_risk": str(cog_path)}
