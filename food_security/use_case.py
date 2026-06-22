from __future__ import annotations

import logging
import os
import traceback

_log = logging.getLogger(__name__)

import ee
import numpy as np
from dask.distributed import as_completed as dask_as_completed

from climate_change.core.base_use_case import (
    AnalysisConfig,
    AnalysisOutput,
    BaseUseCase,
    _ee_geometry_from_geojson,
    _lons_lats,
)
from climate_change.core.dask_engine import DaskEngine
from climate_change.core.gee_auth import ensure_gee
from climate_change.core.runner import register_module

from .cog_export import export_food_security_cog, predict_food_security_grid
from .features import (
    build_feature_datasets,
    build_gee_feature_stack,
    fetch_monthly_rainfall,
    fetch_ndvi_monthly_timeseries,
    sample_training_data,
)
from .model import FoodSecurityModel


def _aoi_area_ha(aoi_geojson: dict | None) -> float | None:
    if not aoi_geojson:
        return None
    try:
        from pyproj import Geod
        from shapely.geometry import shape

        geom = shape(aoi_geojson)
        area_m2, _ = Geod(ellps="WGS84").geometry_area_perimeter(geom)
        return abs(float(area_m2)) / 10_000.0
    except Exception:
        _log.warning(
            "AOI area calculation failed; area stats will be omitted", exc_info=True
        )
        return None


def _attach_risk_area(charts: dict, total_area_ha: float | None) -> None:
    if not total_area_ha:
        return
    risk_dist = charts.get("riskDist")
    if not isinstance(risk_dist, dict):
        return
    percentages = risk_dist.get("data")
    if not isinstance(percentages, list):
        return
    areas = [round(total_area_ha * (float(pct) / 100.0), 2) for pct in percentages]
    risk_dist["area_ha"] = areas
    risk_dist["data_ha"] = areas


def _bbox_from_geojson(aoi_geojson: dict) -> list[float]:
    lons, lats = _lons_lats(aoi_geojson)
    return [min(lons), min(lats), max(lons), max(lats)]


class FoodSecurityUseCase(BaseUseCase):
    """
    Entry point for the food security assessment domain.

    Minimal config (all optional — defaults match Marsabit County, Kenya 2018–2023):

    {
      "aoi_geojson":        {"type": "Polygon", "coordinates": [...]},
      "gee_project":        "your-gee-project-id",
      "start_date":         "2018-01-01",
      "end_date":           "2023-12-31",
      "lt_baseline_start":  "2001-01-01",   # long-term baseline for VCI/TCI
      "s2_start":           "2020-01-01",   # Sentinel-2 MNDWI start
      "model_type":         "rf",           # "rf" | "xgboost" | "ensemble"
      "n_pixels":           3000,
      "scale":              1000,           # metres — MODIS / CHIRPS native
      "output_dir":         "outputs",
      "prefix":             "food_security",
    }
    """

    def __init__(self, dask_engine: DaskEngine) -> None:
        super().__init__(dask_engine)

    # ── BaseUseCase abstract methods (called by execute() via core runner) ──

    def fetch_data(self, config: AnalysisConfig) -> dict:
        flat: dict = {
            "aoi_geojson": config.aoi_geojson,
            "start_date": config.start_date,
            "end_date": config.end_date,
            "country": config.country,
            **config.extra_params,
        }
        return self._fetch_from_dict(flat)

    def preprocess(self, raw_data: dict, config: AnalysisConfig) -> dict:
        return self._preprocess_raw(raw_data)

    def run_model(self, features: dict, config: AnalysisConfig) -> AnalysisOutput:
        model_type = config.extra_params.get("model_type", "rf")
        dict_config = {"model_type": model_type, **config.extra_params}
        result = self._run_model_dict(features, dict_config)
        total_area_ha = _aoi_area_ha(config.aoi_geojson)
        spatial_grid = predict_food_security_grid(
            rf_model=features["_rf"],
            xgb_model=features["_xgb"],
            scaler=features["_scaler"],
            datasets=features["datasets"],
            model_type=model_type,
            aoi_geojson=config.aoi_geojson,
        )
        self._apply_spatial_risk_summary(result, spatial_grid, total_area_ha)
        if total_area_ha:
            result["stats"]["total_area_ha"] = round(total_area_ha, 1)
            _attach_risk_area(result.get("charts", {}), total_area_ha)
        raster_paths: dict[str, str] | None = None
        raster_error: str | None = None
        try:
            raster_paths = export_food_security_cog(
                rf_model=features["_rf"],
                xgb_model=features["_xgb"],
                scaler=features["_scaler"],
                datasets=features["datasets"],
                output_dir=dict_config.get("output_dir", "outputs"),
                prefix=dict_config.get("prefix", "food_security"),
                model_type=model_type,
                aoi_geojson=config.aoi_geojson,
            )
        except Exception as exc:
            raster_error = str(exc)

        geojson_features = self._risk_grid_to_geojson(spatial_grid)
        for p in result.pop("_sample_points", []):
            geojson_features.append(
                {
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [p["lon"], p["lat"]]},
                    "properties": {"risk_class": p["risk_class"]},
                }
            )
        metadata = {
            "model": model_type,
            "country": config.country,
            "start_date": config.start_date,
            "end_date": config.end_date,
            "raster": raster_paths or {},
            "spatial_resolution_m": dict_config.get("scale"),
            "n_pixels_sampled": dict_config.get("n_pixels"),
        }
        if raster_error:
            metadata["raster_error"] = raster_error

        return AnalysisOutput(
            module="food_security",
            geojson={"type": "FeatureCollection", "features": geojson_features},
            raster_path=raster_paths,
            stats={**result["stats"], "country": config.country},
            shap=result.get("charts", {}).get("shap"),
            charts=result.get("charts", {}),
            metadata=metadata,
        )

    # ── Private pipeline helpers ────────────────────────────────────────────

    def _fetch_from_dict(self, config: dict) -> dict:
        """Authenticate GEE and parse the AOI geometry."""
        project = config.get("gee_project", os.environ.get("GEE_PROJECT", ""))
        ensure_gee(project)
        bbox = _bbox_from_geojson(config["aoi_geojson"])
        aoi = _ee_geometry_from_geojson(config["aoi_geojson"])
        return {"aoi": aoi, "bbox": bbox, "config": config}

    def _preprocess_raw(self, raw_data: dict) -> dict:
        """
        Download GEE feature bands, monthly time series, and sample labelled pixels.

        The four independent GEE stages run concurrently:
          1. build_feature_datasets         — 6 band downloads (each internally parallel)
          2. build_gee_feature_stack         — server-side image assembly for sampling
          3. fetch_ndvi_monthly_timeseries   — monthly NDVI area means
          4. fetch_monthly_rainfall          — monthly rainfall area means

        Sampling depends on the feature stack and runs after stage 2 completes.
        All I/O runs in threads sharing the authenticated GEE session.
        """
        from concurrent.futures import ThreadPoolExecutor

        aoi = raw_data["aoi"]
        cfg = raw_data["config"]
        start = cfg.get("start_date", "2018-01-01")
        end = cfg.get("end_date", "2023-12-31")
        scale = cfg.get("scale", 1000)
        n_pix = cfg.get("n_pixels", 3000)

        with ThreadPoolExecutor(max_workers=4) as pool:
            f_datasets = pool.submit(build_feature_datasets, aoi, cfg)
            f_stack = pool.submit(build_gee_feature_stack, aoi, cfg)
            f_ndvi = pool.submit(fetch_ndvi_monthly_timeseries, aoi, start, end)
            f_rain = pool.submit(fetch_monthly_rainfall, aoi, start, end)
            datasets = f_datasets.result()
            feature_stack = f_stack.result()
            ndvi_df = f_ndvi.result()
            rain_df = f_rain.result()

        df = sample_training_data(feature_stack, aoi, n_pixels=n_pix, scale=scale)
        return {
            "datasets": datasets,
            "training_df": df,
            "ndvi_df": ndvi_df,
            "rain_df": rain_df,
            "aoi": aoi,
            "bbox": raw_data["bbox"],
        }

    def _run_model_dict(self, features: dict, config: dict | None = None) -> dict:
        """Train RF + XGBoost, evaluate, compute SHAP and VHI summary statistics."""
        model = FoodSecurityModel()
        result = model.predict(
            features["training_df"],
            ndvi_df=features.get("ndvi_df"),
            rain_df=features.get("rain_df"),
            config=config,
        )
        features["_rf"] = model.rf
        features["_xgb"] = model.xgb
        features["_scaler"] = model.scaler
        return result

    # ── Standalone pipeline (direct use without core runner) ───────────────

    def run(self, config: dict) -> dict:
        """Single AOI analysis — full pipeline in one call."""
        raw_data = self._fetch_from_dict(config)
        features = self._preprocess_raw(raw_data)
        result = self._run_model_dict(features, config)
        total_area_ha = _aoi_area_ha(config.get("aoi_geojson"))
        spatial_grid = predict_food_security_grid(
            rf_model=features["_rf"],
            xgb_model=features["_xgb"],
            scaler=features["_scaler"],
            datasets=features["datasets"],
            model_type=config.get("model_type", "rf"),
            aoi_geojson=config.get("aoi_geojson"),
        )
        self._apply_spatial_risk_summary(result, spatial_grid, total_area_ha)
        if total_area_ha:
            result["stats"]["total_area_ha"] = round(total_area_ha, 1)
            _attach_risk_area(result.get("charts", {}), total_area_ha)

        cog_paths: dict[str, str] | None = None
        try:
            cog_paths = export_food_security_cog(
                rf_model=features["_rf"],
                xgb_model=features["_xgb"],
                scaler=features["_scaler"],
                datasets=features["datasets"],
                output_dir=config.get("output_dir", "outputs"),
                prefix=config.get("prefix", "food_security"),
                model_type=config.get("model_type", "rf"),
                aoi_geojson=config.get("aoi_geojson"),
            )
        except Exception as exc:
            result["raster_error"] = str(exc)
        result["raster"] = cog_paths or {}
        result["geojson"] = {
            "type": "FeatureCollection",
            "features": self._risk_grid_to_geojson(spatial_grid),
        }
        result["stats"].update(
            {
                "bbox": features["bbox"],
                "prefix": config.get("prefix", "food_security"),
            }
        )
        return result

    @staticmethod
    def _apply_spatial_risk_summary(
        result: dict,
        spatial_grid: dict,
        total_area_ha: float | None,
    ) -> None:
        labels = ["Low Risk", "Medium Risk", "High Risk"]
        colors = ["#184c09", "#ffcc36", "#f22d06"]
        percentages = [round(float(pct), 1) for pct in spatial_grid["percentages"]]
        counts = [int(count) for count in spatial_grid["counts"]]

        risk_dist = {
            "labels": labels,
            "data": percentages,
            "colors": colors,
            "pixel_count": counts,
            "basis": "full_spatial_aoi",
            "valid_pixel_count": int(spatial_grid["valid_pixel_count"]),
        }
        if total_area_ha:
            areas = [
                round(total_area_ha * (float(pct) / 100.0), 2) for pct in percentages
            ]
            risk_dist["area_ha"] = areas
            risk_dist["data_ha"] = areas

        charts = result.setdefault("charts", {})
        charts["riskDist"] = risk_dist
        result.setdefault("stats", {}).update(
            {
                "analysed_pixels": int(spatial_grid["valid_pixel_count"]),
                "low_risk_pct": percentages[0],
                "medium_risk_pct": percentages[1],
                "high_risk_pct": percentages[2],
            }
        )

    @staticmethod
    def _risk_grid_to_geojson(spatial_grid: dict) -> list[dict]:
        try:
            from rasterio.features import shapes
        except ImportError:
            _log.warning(
                "rasterio not available; food-security GeoJSON features will be empty"
            )
            return []

        risk_grid = np.asarray(spatial_grid["risk_grid"], dtype=np.uint8)
        mask = risk_grid > 0
        labels = {1: "Low Risk", 2: "Medium Risk", 3: "High Risk"}
        colors = {1: "#184c09", 2: "#ffcc36", 3: "#f22d06"}
        features = []
        for geom, value in shapes(
            risk_grid, mask=mask, transform=spatial_grid["transform"]
        ):
            code = int(value)
            if code == 0:
                continue
            features.append(
                {
                    "type": "Feature",
                    "geometry": geom,
                    "properties": {
                        "risk_class": labels.get(code, "Unknown"),
                        "risk_code": code,
                        "color": colors.get(code, "#95A5A6"),
                    },
                }
            )
        return features

    def run_date_ranges(self, config: dict, date_ranges: list[dict]) -> list[dict]:
        """Run the same AOI over multiple date-range configurations in parallel."""
        merged = [{**config, **dr} for dr in date_ranges]
        return self._run_parallel(merged)

    def run_multi_regions(self, configs: list[dict]) -> list[dict]:
        """Run multiple AOI configs (same timeframe, different regions) in parallel."""
        return self._run_parallel(configs)

    def _run_parallel(self, configs: list[dict]) -> list[dict]:
        """
        Run each config on a separate Dask worker for true distributed execution.
        Each worker calls self.run() which re-initialises GEE via ensure_gee()
        (skipping the interactive auth step — credentials must be pre-configured).
        """
        from climate_change.core.dask_engine import DaskEngine

        client = DaskEngine.get_client()
        n = len(configs)
        results: list[dict | None] = [None] * n
        idx_map = {
            client.submit(self.run, cfg, pure=False): i for i, cfg in enumerate(configs)
        }
        for future in dask_as_completed(idx_map):
            idx = idx_map[future]
            try:
                results[idx] = future.result()
            except Exception as exc:
                results[idx] = {
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                    "config": configs[idx],
                }
        return results  # type: ignore[return-value]


register_module("food_security", FoodSecurityUseCase)
