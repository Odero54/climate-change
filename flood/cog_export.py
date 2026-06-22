from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import rasterio
import xarray as xr
from rasterio.features import geometry_mask
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from climate_change.core.base_use_case import _aoi_geometries

from .features import FEATURE_COLS, RISK_INT, align_datasets
from .model import VALID_MODEL_TYPES, classify_flood_risk

_log = logging.getLogger(__name__)

RISK_ORDER = ("Very High", "High", "Medium", "Low")
RISK_VALUE_BY_LABEL = dict(RISK_INT)
RISK_LABEL_BY_VALUE = {value: label for label, value in RISK_INT.items()}


def flood_raster_distribution(path: str | Path) -> dict:
    with rasterio.open(path) as src:
        data = src.read(1)

    valid_values = data[data > 0]
    counts = {
        label: int((valid_values == RISK_VALUE_BY_LABEL[label]).sum()) for label in RISK_ORDER
    }
    total = int(valid_values.size)
    percentages = {
        label: round((count / total * 100), 1) if total else 0.0 for label, count in counts.items()
    }
    return {
        "labels": list(RISK_ORDER),
        "counts": [counts[label] for label in RISK_ORDER],
        "percentages": [percentages[label] for label in RISK_ORDER],
        "valid_pixel_count": total,
    }


def export_flood_cog(
    rf_model: RandomForestClassifier,
    xgb_model: XGBClassifier,
    datasets: dict[str, xr.Dataset],
    output_dir: str,
    prefix: str,
    model_type: str = "ensemble",
    aoi_geojson: dict | None = None,
) -> dict[str, str]:
    """
    Build the full-AOI feature matrix from in-memory xarray Datasets,
    run inference with the selected model, classify into 4 risk classes,
    and write a Cloud Optimised GeoTIFF.
    """
    if model_type not in VALID_MODEL_TYPES:
        raise ValueError(f"model_type must be one of {VALID_MODEL_TYPES}, got '{model_type}'")

    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    cog_path = output_dir_path / f"{prefix}_flood_risk.tif"

    # Align all bands to the terrain (90 m) reference grid
    aligned = align_datasets(datasets, ref_key="terrain")
    terrain = aligned["terrain"]
    ref_lat = terrain.lat
    ref_lon = terrain.lon
    n_lat, n_lon = len(ref_lat), len(ref_lon)

    # Extract feature arrays (all float32)
    elevation = terrain["elevation"].values.astype(np.float32)
    twi_ = aligned["twi"]["twi"].values.astype(np.float32)
    dist_river_ = aligned["dist_river"]["dist_river"].values.astype(np.float32)
    vv_change_ = aligned["sar"]["vv_change"].values.astype(np.float32)
    rain7_ = aligned["rainfall"]["rainfall_7d"].values.astype(np.float32)
    rain30_ = aligned["rainfall"]["rainfall_30d"].values.astype(np.float32)
    mndwi_ = aligned["mndwi"]["mndwi"].values.astype(np.float32)
    landcover_ = aligned["landcover"]["Map"].values.astype(np.float32)
    landcover_ = (landcover_ / 10.0) - 1.0  # normalise: match §8

    lon_grid = np.broadcast_to(ref_lon.values[np.newaxis, :], (n_lat, n_lon)).astype(np.float32)
    lat_grid = np.broadcast_to(ref_lat.values[:, np.newaxis], (n_lat, n_lon)).astype(np.float32)

    # Stack → (n_pixels, n_features) and mask nodata
    bands = np.stack(
        [
            elevation,
            twi_,
            dist_river_,
            vv_change_,
            rain7_,
            rain30_,
            mndwi_,
            landcover_,
            lon_grid,
            lat_grid,
        ],
        axis=0,
    )  # (10, n_lat, n_lon)
    X_full = bands.reshape(len(FEATURE_COLS), -1).T  # (n_pixels, 10)
    valid_mask = ~np.isnan(X_full).any(axis=1)

    transform = terrain["elevation"].rio.transform()
    crs = terrain["elevation"].rio.crs or "EPSG:4326"
    geometries = _aoi_geometries(aoi_geojson)
    if geometries:
        inside_aoi = geometry_mask(
            geometries,
            out_shape=(n_lat, n_lon),
            transform=transform,
            invert=True,
        ).reshape(-1)
        valid_mask = valid_mask & inside_aoi

    X_valid = X_full[valid_mask]

    # Inference — route by model_type
    risk_grid: np.ndarray = np.zeros(n_lat * n_lon, dtype=np.uint8)
    if X_valid.size:
        prob_rf = rf_model.predict_proba(X_valid)[:, 1]
        prob_xgb = xgb_model.predict_proba(X_valid)[:, 1]
        if model_type == "rf":
            selected_prob = prob_rf
        elif model_type == "xgboost":
            selected_prob = prob_xgb
        else:  # ensemble
            selected_prob = (prob_rf + prob_xgb) / 2.0

        # Classify and reconstruct spatial grid
        risk_labels = classify_flood_risk(selected_prob)
        risk_valid = np.array([RISK_INT[r] for r in risk_labels], dtype=np.uint8)
        risk_grid[valid_mask] = risk_valid
    risk_grid = risk_grid.reshape(n_lat, n_lon)

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
            RISK_CLASSES="0=nodata,1=Low,2=Medium,3=High,4=Very High",
            MODEL_TYPE=model_type,
        )

    uncompressed_mb = risk_grid.nbytes / 1e6
    _log.info(
        "COG exported: %s  (%d×%d px, %.1f MB uncompressed)",
        cog_path,
        n_lat,
        n_lon,
        uncompressed_mb,
    )
    return {"flood_risk": str(cog_path)}
