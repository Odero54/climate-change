from __future__ import annotations

from pathlib import Path

import lightgbm as lgb
import numpy as np
import rioxarray  # noqa: F401 — activates the .rio accessor
import xarray as xr
from rasterio.features import geometry_mask
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from .features import FEATURE_COLS, align_datasets


def export_degradation_cog(
    rf_model: RandomForestClassifier,
    lgbm_model: lgb.LGBMClassifier,
    scaler: StandardScaler,
    datasets: dict[str, xr.Dataset],
    output_dir: str = "outputs",
    prefix: str = "land_degradation",
    model_type: str = "lgbm",
    aoi_geojson: dict | None = None,
) -> dict[str, str]:
    """
    Apply the trained model to the full pixel grid and write a Cloud-Optimised GeoTIFF.

    Prediction values:
      0  = Not Degraded
      1  = Degraded
      -1 = NoData (pixels with missing feature values)

    Returns dict: {"degradation_risk": "<path>"}
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    aligned = align_datasets(datasets, ref_key="ndvi")
    ref_ds = aligned["ndvi"]
    lat = ref_ds.lat.values
    lon = ref_ds.lon.values
    nlat = len(lat)
    nlon = len(lon)
    n_px = nlat * nlon

    # Build feature matrix in FEATURE_COLS order
    feat_arrays: dict[str, np.ndarray] = {}
    for _key, ds in aligned.items():
        for var in ds.data_vars:
            col = "land_cover" if var == "Map" else str(var)
            if col in FEATURE_COLS:
                feat_arrays[col] = ds[var].values.ravel()

    X = np.column_stack([feat_arrays.get(c, np.full(n_px, np.nan)) for c in FEATURE_COLS])
    valid_mask = ~np.any(np.isnan(X), axis=1)
    X_valid = scaler.transform(X[valid_mask])

    if model_type == "rf":
        pred_valid = np.asarray(rf_model.predict(X_valid)).astype(np.int8)
    elif model_type == "lgbm":
        pred_valid = np.asarray(lgbm_model.predict(X_valid)).astype(np.int8)
    else:
        rf_pred = np.asarray(rf_model.predict(X_valid)).astype(int)
        lgbm_pred = np.asarray(lgbm_model.predict(X_valid)).astype(int)
        pred_valid = ((rf_pred + lgbm_pred) >= 1).astype(np.int8)

    prediction = np.full(n_px, -1, dtype=np.int8)
    prediction[valid_mask] = pred_valid
    prediction_2d = prediction.reshape(nlat, nlon)
    transform = ref_ds.rio.transform()
    if aoi_geojson:
        inside_aoi = geometry_mask(
            [aoi_geojson],
            out_shape=(nlat, nlon),
            transform=transform,
            invert=True,
        )
        prediction_2d[~inside_aoi] = -1

    da = xr.DataArray(
        prediction_2d,
        dims=["lat", "lon"],
        coords={"lat": lat, "lon": lon},
        name="degradation_class",
    )
    da.attrs.update({"long_name": "Degradation class (0=Not Degraded, 1=Degraded)", "nodata": -1})
    da = da.rio.set_spatial_dims(x_dim="lon", y_dim="lat")
    da = da.rio.write_crs("EPSG:4326")
    da = da.rio.set_nodata(-1)

    cog_path = out / f"{prefix}_degradation_risk.tif"
    da.rio.to_raster(str(cog_path), driver="COG", compress="LZW")

    return {"degradation_risk": str(cog_path)}
