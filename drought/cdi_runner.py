import logging
from pathlib import Path

import numpy as np
import rasterio
import xarray as xr
from drought_monitoring import compute_all
from drought_monitoring.gee import (
    fetch_era5_precip,
    fetch_era5_temp,
    fetch_modis_ndvi,
    yearly_drought_maps,
)
from drought_monitoring.plot import classify_cdi
from rasterio.features import geometry_mask

from climate_change.core.base_use_case import _aoi_geometries

_log = logging.getLogger(__name__)

#  CDI severity reference
CDI_SCALE = [
    (0.50, "Extreme drought"),
    (0.65, "Severe drought"),
    (0.80, "Moderate drought"),
    (0.90, "Mild drought"),
    (1.10, "Near normal"),
    (1.20, "Mild wet"),
    (1.30, "Moderately wet"),
    (float("inf"), "Very wet"),
]

DROUGHT_CLASS_COLORS = {
    "Extreme drought": "#990000",
    "Severe drought": "#E65100",
    "Moderate drought": "#FFB74D",
    "Mild drought": "#F57C00",
    "Near normal": "#E0E0E0",
    "Mild wet": "#80CBC4",
    "Moderately wet": "#00897B",
    "Very wet": "#4DB6AC",
}

DROUGHT_CLASS_ORDER = [
    "Extreme drought",
    "Severe drought",
    "Mild drought",
    "Moderate drought",
    "Near normal",
    "Mild wet",
    "Very wet",
    "Moderately wet",
]


def _normalize_drought_class(label):
    aliases = {
        "normal/wet": "Near normal",
        "near normal": "Near normal",
        "mild wet": "Mild wet",
        "moderately wet": "Moderately wet",
        "very wet": "Very wet",
        "mild drought": "Mild drought",
        "moderate drought": "Moderate drought",
        "severe drought": "Severe drought",
        "extreme drought": "Extreme drought",
    }
    return aliases.get(str(label).strip().lower(), str(label))


def _classify_cdi_value(value: float) -> str:
    if value < 0.50:
        return "Extreme drought"
    if value < 0.65:
        return "Severe drought"
    if value < 0.80:
        return "Moderate drought"
    if value < 0.90:
        return "Mild drought"
    if value < 1.10:
        return "Near normal"
    if value < 1.20:
        return "Mild wet"
    if value < 1.30:
        return "Moderately wet"
    return "Very wet"


def _temporally_fill_dataframe(df):
    value_cols = [col for col in ["PDI", "TDI", "VDI", "CDI"] if col in df.columns]
    if not value_cols:
        return df.bfill().ffill()

    filled = df.copy()
    filled[value_cols] = (
        filled[value_cols]
        .replace([np.inf, -np.inf], np.nan)
        .interpolate(method="time", limit_direction="both")
        .bfill()
        .ffill()
    )
    for col in value_cols:
        if filled[col].isna().any():
            fallback = filled[col].median()
            if not np.isfinite(fallback):
                fallback = 0.0
            filled[col] = filled[col].fillna(float(fallback))
    return filled


def _temporally_fill_dataset(ds):
    filled = ds.where(np.isfinite(ds))
    try:
        filled = filled.interpolate_na(
            dim="time",
            method="linear",
            fill_value="extrapolate",
        )
    except Exception:
        _log.debug("interpolate_na failed; falling back to ffill/bfill", exc_info=True)
    filled = filled.ffill(dim="time").bfill(dim="time")

    for name in filled.data_vars:
        arr = filled[name]
        if not bool(arr.isnull().any()):
            continue
        temporal_mean = arr.mean(dim="time", skipna=True)
        temporal_mean = temporal_mean.fillna(arr.mean(skipna=True))
        temporal_mean = temporal_mean.fillna(0)
        filled[name] = arr.fillna(temporal_mean)
    return filled


def run_cdi_pipeline(raw_data: dict) -> dict:
    """
    Full CDI pipeline.
    """
    bbox = raw_data["bbox"]  # [lon_min, lat_min, lon_max, lat_max]
    aoi = raw_data.get("aoi_geojson") or bbox
    start_year = raw_data["start_year"]
    end_year = raw_data["end_year"]

    # Monthly area-averaged time series
    precip = fetch_era5_precip(aoi, start_year=start_year, end_year=end_year)
    temp = fetch_era5_temp(aoi, start_year=start_year, end_year=end_year)
    ndvi = fetch_modis_ndvi(aoi, start_year=start_year, end_year=end_year)

    df = compute_all(precip, temp, ndvi, window=3, weights=(0.50, 0.25, 0.25))
    df = _temporally_fill_dataframe(df)
    df["severity"] = classify_cdi(df["CDI"]).map(_normalize_drought_class)

    # Annual pixel-wise spatial CDI maps
    # end_year - 1 because the current partial year has no complete annual map
    ds = yearly_drought_maps(aoi, start_year=start_year, end_year=end_year - 1)
    ds = _temporally_fill_dataset(ds)
    ds = _mask_dataset_to_aoi(ds, raw_data.get("aoi_geojson"))
    return {
        "cdi_maps": ds,
        "cdi_series": df,
        "precip": precip,
        "temp": temp,
        "ndvi": ndvi,
        "bbox": bbox,
        "aoi_geojson": raw_data.get("aoi_geojson"),
        "start_year": start_year,
        "end_year": end_year,
    }


def build_drought_charts(features: dict) -> dict:
    """
    Build chart data payloads for the frontend.
    """
    df = features["cdi_series"]

    timeseries = {
        "labels": df.index.strftime("%Y-%m").tolist(),
        "datasets": [
            {"label": "CDI", "data": df["CDI"].tolist(), "color": "#C0392B"},
            {"label": "PDI", "data": df["PDI"].tolist(), "color": "#2980B9"},
            {"label": "TDI", "data": df["TDI"].tolist(), "color": "#E67E22"},
            {"label": "VDI", "data": df["VDI"].tolist(), "color": "#27AE60"},
        ],
    }

    annual = df["CDI"].resample("YE").mean()
    anomaly = {
        "labels": annual.index.year.tolist(),
        "data": (annual - annual.mean()).round(4).tolist(),
        "mean": float(annual.mean()),
    }

    seasonal = df["CDI"].groupby(df.index.month).mean()
    seasonal_chart = {
        "labels": list(range(1, 13)),
        "data": seasonal.round(4).tolist(),
    }

    temporal_severity_dist = (
        df["severity"].value_counts(normalize=True).mul(100).round(1)
    )
    ds = features.get("cdi_maps")
    latest = ds["CDI"].isel(time=-1).values if ds is not None else np.array([])
    valid = latest[np.isfinite(latest)]
    if valid.size:
        classes = np.array([_classify_cdi_value(float(value)) for value in valid])
        counts = {label: int((classes == label).sum()) for label in DROUGHT_CLASS_ORDER}
        labels = [label for label in DROUGHT_CLASS_ORDER if counts[label] > 0]
        data = [round(counts[label] / valid.size * 100, 1) for label in labels]
    else:
        labels = temporal_severity_dist.index.tolist()
        data = temporal_severity_dist.tolist()
    severity = {
        "labels": labels,
        "data": data,
        "colors": [DROUGHT_CLASS_COLORS.get(str(label), "#95A5A6") for label in labels],
        "basis": "latest_spatial_aoi",
        "valid_pixel_count": int(valid.size),
    }
    temporal_severity = {
        "labels": temporal_severity_dist.index.tolist(),
        "data": temporal_severity_dist.tolist(),
        "colors": [
            DROUGHT_CLASS_COLORS.get(str(label), "#95A5A6")
            for label in temporal_severity_dist.index
        ],
        "basis": "monthly_aoi_series",
    }

    return {
        "timeseries": timeseries,
        "anomaly": anomaly,
        "seasonal": seasonal_chart,
        "severity_distribution": severity,
        "temporal_severity_distribution": temporal_severity,
    }


def compute_spatial_uncertainty(features: dict) -> dict:
    """
    Two spatial uncertainty metrics derived from the annual CDI dataset.
    """
    ds = features["cdi_maps"]
    cdi_std = ds["CDI"].std("time").values  # (lon, lat)
    stacked = np.stack(
        [ds["PDI"].values, ds["TDI"].values, ds["VDI"].values], axis=0
    )  # (3, time, lon, lat)
    component_spread = np.nanmean(np.nanstd(stacked, axis=0), axis=0)  # (lon, lat)
    finite_cdi_std = cdi_std[np.isfinite(cdi_std)]
    finite_component_spread = component_spread[np.isfinite(component_spread)]

    return {
        "lons": ds.lon.values.tolist(),
        "lats": ds.lat.values.tolist(),
        "temporal_std": cdi_std.T.tolist(),  # (lat, lon)
        "component_spread": component_spread.T.tolist(),  # (lat, lon)
        "temporal_std_stats": {
            "min": float(finite_cdi_std.min()) if finite_cdi_std.size else 0.0,
            "max": float(finite_cdi_std.max()) if finite_cdi_std.size else 0.0,
            "mean": float(finite_cdi_std.mean()) if finite_cdi_std.size else 0.0,
        },
        "component_spread_stats": {
            "min": float(finite_component_spread.min())
            if finite_component_spread.size
            else 0.0,
            "max": float(finite_component_spread.max())
            if finite_component_spread.size
            else 0.0,
            "mean": float(finite_component_spread.mean())
            if finite_component_spread.size
            else 0.0,
        },
    }


def _spatial_dims(ds) -> tuple[str, str]:
    lat_dim = next((dim for dim in ds.dims if dim in ("lat", "latitude", "y")), "lat")
    lon_dim = next((dim for dim in ds.dims if dim in ("lon", "longitude", "x")), "lon")
    return lat_dim, lon_dim


def _dataset_transform(ds):
    from rasterio.transform import from_bounds

    lat_dim, lon_dim = _spatial_dims(ds)
    lats = ds[lat_dim].values
    lons = ds[lon_dim].values
    return from_bounds(
        float(np.nanmin(lons)),
        float(np.nanmin(lats)),
        float(np.nanmax(lons)),
        float(np.nanmax(lats)),
        len(lons),
        len(lats),
    )


def _mask_dataset_to_aoi(ds, aoi_geojson: dict | None):
    geometries = _aoi_geometries(aoi_geojson)
    if not geometries:
        return ds

    lat_dim, lon_dim = _spatial_dims(ds)
    inside_aoi = geometry_mask(
        geometries,
        out_shape=(len(ds[lat_dim]), len(ds[lon_dim])),
        transform=_dataset_transform(ds),
        invert=True,
    )
    lats = ds[lat_dim].values
    if len(lats) > 1 and float(lats[0]) < float(lats[-1]):
        inside_aoi = inside_aoi[::-1, :]
    mask_da = xr.DataArray(
        inside_aoi,
        dims=(lat_dim, lon_dim),
        coords={lat_dim: ds[lat_dim], lon_dim: ds[lon_dim]},
        name="inside_aoi",
    )
    return ds.where(mask_da)


def _clip_cog_to_aoi(path: Path, aoi_geojson: dict | None) -> Path:
    geometries = _aoi_geometries(aoi_geojson)
    if not geometries:
        return path

    tmp_path = path.with_suffix(f".masked{path.suffix}")
    with rasterio.open(path) as src:
        data = src.read().astype("float32")
        inside_aoi = geometry_mask(
            geometries,
            out_shape=(src.height, src.width),
            transform=src.transform,
            invert=True,
        )
        data[:, ~inside_aoi] = np.nan

        profile = src.profile.copy()
        profile.update(
            driver="COG",
            dtype="float32",
            nodata=np.nan,
            compress="deflate",
            predictor=3,
        )
        descriptions = [src.descriptions[idx] for idx in range(src.count)]
        band_tags = [src.tags(idx) for idx in range(1, src.count + 1)]
        dataset_tags = src.tags()

    with rasterio.open(tmp_path, "w", **profile) as dst:
        dst.write(data)
        dst.update_tags(**dataset_tags)
        for idx, description in enumerate(descriptions, start=1):
            if description:
                dst.set_band_description(idx, description)
            if band_tags[idx - 1]:
                dst.update_tags(idx, **band_tags[idx - 1])

    tmp_path.replace(path)
    return path


def export_cdi_cog(
    features: dict,
    output_dir: str,
    aoi_geojson: dict | None = None,
) -> dict[str, Path]:
    """Export CDI maps to Cloud Optimised GeoTIFFs. Called only at the save step."""
    from drought_monitoring.io import cdi_stack_to_cog

    ds = features["cdi_maps"]
    prefix = f"drought_{features['start_year']}_{features['end_year']}"
    paths = cdi_stack_to_cog(ds, output_dir=output_dir, prefix=prefix)
    if aoi_geojson:
        paths = {
            key: _clip_cog_to_aoi(path, aoi_geojson) for key, path in paths.items()
        }
    return paths
