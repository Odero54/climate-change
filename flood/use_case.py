from __future__ import annotations

import os
import traceback

import ee
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

from .cog_export import export_flood_cog, flood_raster_distribution
from .features import (
    build_feature_datasets,
    sample_training_data,
)
from .model import FloodModel


class FloodRiskUseCase(BaseUseCase):
    """
    Entry point for the flood risk domain.

    Minimal config (all optional — defaults match the Niger flood notebook example):

    {
      "aoi_geojson":      {"type": "Polygon", "coordinates": [...]},
      "gee_project":      "your-gee-project-id",
      "model_type":       "ensemble",        # "rf" | "xgboost" | "ensemble"
      "output_dir":       "outputs",
      "prefix":           "flood_2022_2023",
      "flood_event":      "Niger River flood pulse Oct 2022-Jan 2023",
      # Main analysis period
      "start_date":       "2022-01-01",  "end_date":       "2023-01-31",
      # Flood event windows
      "pre_flood_start":  "2022-05-01",  "pre_flood_end":  "2022-07-31",
      "post_flood_start": "2022-10-01",  "post_flood_end": "2023-01-31",
      # Rainfall windows
      "rain_7d_start":    "2022-08-18",  "rain_7d_end":    "2022-08-25",
      "rain_30d_start":   "2022-08-01",  "rain_30d_end":   "2022-08-31",
      # Sentinel-2 MNDWI window
      "mndwi_start":      "2022-10-01",  "mndwi_end":      "2023-01-31",
      # JRC flood label window
      "flood_label_start":"2021-01-01",  "flood_label_end":"2021-12-31",
      # Sampling
      "n_pixels": 3000,
      "scale":    90,
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
        model_type = config.extra_params.get("model_type", "ensemble")
        dict_config = {"model_type": model_type, **config.extra_params}
        result = self._run_model_dict(features, dict_config)
        raster_paths: dict[str, str] | None = None
        raster_error: str | None = None
        try:
            raster_paths = export_flood_cog(
                rf_model=features["_rf"],
                xgb_model=features["_xgb"],
                datasets=features["datasets"],
                output_dir=dict_config.get("output_dir", "outputs"),
                prefix=dict_config.get("prefix", "flood"),
                model_type=model_type,
                aoi_geojson=config.aoi_geojson,
            )
        except Exception as exc:
            raster_error = str(exc)

        if raster_paths and raster_paths.get("flood_risk"):
            distribution = flood_raster_distribution(raster_paths["flood_risk"])
            labels = distribution["labels"]
            percentages = distribution["percentages"]
            counts = distribution["counts"]
            result["charts"]["risk_distribution"] = {
                "labels": labels,
                "data": percentages,
                "counts": counts,
                "colors": [
                    "#E74C3C",
                    "#E67E22",
                    "#F1C40F",
                    "#2ECC71",
                ],
            }
            pct_by_label = dict(zip(labels, percentages, strict=False))
            count_by_label = dict(zip(labels, counts, strict=False))
            result["stats"].update(
                {
                    "very_high_risk_pct": pct_by_label.get("Very High", 0.0),
                    "high_risk_pct": pct_by_label.get("High", 0.0),
                    "medium_risk_pct": pct_by_label.get("Medium", 0.0),
                    "low_risk_pct": pct_by_label.get("Low", 0.0),
                    "very_high_risk_pixels": count_by_label.get("Very High", 0),
                    "high_risk_pixels": count_by_label.get("High", 0),
                    "medium_risk_pixels": count_by_label.get("Medium", 0),
                    "low_risk_pixels": count_by_label.get("Low", 0),
                    "mapped_pixel_count": distribution["valid_pixel_count"],
                }
            )

        geojson_features = []
        if config.aoi_geojson:
            geojson_features.append(
                {
                    "type": "Feature",
                    "geometry": config.aoi_geojson,
                    "properties": {"type": "boundary"},
                }
            )
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
            "windows": {
                "pre_flood": {
                    "start": dict_config.get("pre_flood_start")
                    or dict_config.get("pre_sar_start"),
                    "end": dict_config.get("pre_flood_end")
                    or dict_config.get("pre_sar_end"),
                },
                "post_flood": {
                    "start": dict_config.get("post_flood_start")
                    or dict_config.get("flood_sar_start"),
                    "end": dict_config.get("post_flood_end")
                    or dict_config.get("flood_sar_end"),
                },
                "rainfall_7d": {
                    "start": dict_config.get("rain_7d_start"),
                    "end": dict_config.get("rain_7d_end"),
                },
                "rainfall_30d": {
                    "start": dict_config.get("rain_30d_start"),
                    "end": dict_config.get("rain_30d_end"),
                },
                "mndwi": {
                    "start": dict_config.get("mndwi_start"),
                    "end": dict_config.get("mndwi_end"),
                },
            },
        }
        if raster_error:
            metadata["raster_error"] = raster_error

        return AnalysisOutput(
            module="flood",
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
        lons, lats = _lons_lats(config["aoi_geojson"])
        bbox = [min(lons), min(lats), max(lons), max(lats)]
        aoi = _ee_geometry_from_geojson(config["aoi_geojson"])
        return {"aoi": aoi, "bbox": bbox, "config": config}

    def _preprocess_raw(self, raw_data: dict) -> dict:
        """
        Download GEE feature bands and sample labelled training pixels.

        Two independent GEE stages run concurrently:
          1. build_feature_datasets — 7 band downloads for COG export
          2. sample_training_data  — builds static/dynamic/label images in
             background threads internally, then samples pixels
        """
        from concurrent.futures import ThreadPoolExecutor

        aoi = raw_data["aoi"]
        cfg = raw_data["config"]
        n_pix = cfg.get("n_pixels", 5000)
        scale = cfg.get("scale", 90)

        with ThreadPoolExecutor(max_workers=2) as pool:
            f_datasets = pool.submit(build_feature_datasets, aoi, cfg)
            f_df = pool.submit(sample_training_data, aoi, cfg, n_pix, scale)
            datasets = f_datasets.result()
            df = f_df.result()

        return {
            "datasets": datasets,
            "training_df": df,
            "aoi": aoi,
            "bbox": raw_data["bbox"],
        }

    def _run_model_dict(self, features: dict, config: dict | None = None) -> dict:
        """Train RF + XGBoost, evaluate, compute SHAP and uncertainty."""
        model = FloodModel()
        result = model.predict(features["training_df"], config)
        features["_rf"] = model.rf
        features["_xgb"] = model.xgb
        return result

    # ── Standalone pipeline (direct use without core runner) ───────────────

    def run(self, config: dict) -> dict:
        """Single AOI analysis — full pipeline in one call."""
        raw_data = self._fetch_from_dict(config)
        features = self._preprocess_raw(raw_data)
        result = self._run_model_dict(features, config)

        model_type = config.get("model_type", "ensemble")
        raster_paths: dict[str, str] | None = None
        raster_error: str | None = None
        try:
            raster_paths = export_flood_cog(
                rf_model=features["_rf"],
                xgb_model=features["_xgb"],
                datasets=features["datasets"],
                output_dir=config.get("output_dir", "outputs"),
                prefix=config.get("prefix", "flood"),
                model_type=model_type,
                aoi_geojson=config.get("aoi_geojson"),
            )
        except Exception as exc:
            raster_error = str(exc)

        result["raster"] = raster_paths or {}
        result["stats"].update(
            {
                "bbox": features["bbox"],
                "flood_event": config.get("flood_event", ""),
                "prefix": config.get("prefix", "flood"),
            }
        )
        if raster_error:
            result["raster_error"] = raster_error

        if raster_paths and raster_paths.get("flood_risk"):
            distribution = flood_raster_distribution(raster_paths["flood_risk"])
            labels = distribution["labels"]
            percentages = distribution["percentages"]
            counts = distribution["counts"]
            result["charts"]["risk_distribution"] = {
                "labels": labels,
                "data": percentages,
                "counts": counts,
                "colors": [
                    "#E74C3C",
                    "#E67E22",
                    "#F1C40F",
                    "#2ECC71",
                ],
            }
            pct_by_label = dict(zip(labels, percentages, strict=False))
            count_by_label = dict(zip(labels, counts, strict=False))
            result["stats"].update(
                {
                    "very_high_risk_pct": pct_by_label.get("Very High", 0.0),
                    "high_risk_pct": pct_by_label.get("High", 0.0),
                    "medium_risk_pct": pct_by_label.get("Medium", 0.0),
                    "low_risk_pct": pct_by_label.get("Low", 0.0),
                    "very_high_risk_pixels": count_by_label.get("Very High", 0),
                    "high_risk_pixels": count_by_label.get("High", 0),
                    "medium_risk_pixels": count_by_label.get("Medium", 0),
                    "low_risk_pixels": count_by_label.get("Low", 0),
                    "mapped_pixel_count": distribution["valid_pixel_count"],
                }
            )
        return result

    def run_date_ranges(self, config: dict, date_ranges: list[dict]) -> list[dict]:
        """Run the same AOI over two (or more) date-range configurations in parallel."""
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


register_module("flood", FloodRiskUseCase)
