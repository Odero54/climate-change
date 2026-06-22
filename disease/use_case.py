from __future__ import annotations

import logging
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

_log = logging.getLogger(__name__)
from climate_change.core.dask_engine import DaskEngine
from climate_change.core.gee_auth import ensure_gee
from climate_change.core.runner import register_module

from .cog_export import export_disease_cog
from .features import (
    _normalise_date_window,
    build_feature_datasets,
    build_gee_feature_stack,
    fetch_monthly_timeseries,
    sample_training_data,
)
from .model import DiseaseModel


class DiseaseRiskUseCase(BaseUseCase):
    """
    Entry point for the climate-driven disease surveillance domain.

    Minimal config (all optional — defaults match Kisumu County, Kenya 2021–2023):

    {
      "aoi_geojson":  {"type": "Polygon", "coordinates": [...]},
      "gee_project":  "your-gee-project-id",
      "start_date":   "2021-01-01",
      "end_date":     "2023-12-31",
      "model_type":   "gbm",         # "gbm" | "xgboost" | "ensemble"
      "n_pixels":     3000,
      "scale":        1000,          # metres — MODIS / CHIRPS native
      "output_dir":   "outputs",
      "prefix":       "disease",
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
        model_type = config.extra_params.get("model_type", "gbm")
        dict_config = {"model_type": model_type, **config.extra_params}
        result = self._run_model_dict(features, dict_config)

        raster_paths: dict[str, str] | None = None
        raster_error: str | None = None
        try:
            raster_paths = export_disease_cog(
                gbm_model=features["_gbm"],
                xgb_model=features["_xgb"],
                scaler=features["_scaler"],
                datasets=features["datasets"],
                output_dir=dict_config.get("output_dir", "outputs"),
                prefix=dict_config.get("prefix", "disease"),
                model_type=model_type,
                aoi_geojson=config.aoi_geojson,
            )
        except Exception as exc:
            raster_error = str(exc)

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
        }
        if raster_error:
            metadata["raster_error"] = raster_error

        return AnalysisOutput(
            module="disease",
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
        Download GEE feature bands, monthly time series, and sample labelled pixels.

        The three independent GEE stages run concurrently:
          1. build_feature_datasets  — 7 band downloads (each internally parallel)
          2. build_gee_feature_stack — server-side image assembly for sampling
          3. fetch_monthly_timeseries — 3 monthly queries (each internally parallel)

        Sampling depends on the feature stack and runs after stage 2 completes.
        All I/O runs in threads sharing the authenticated GEE session.
        """
        from concurrent.futures import ThreadPoolExecutor

        aoi = raw_data["aoi"]
        cfg = raw_data["config"]
        start = cfg.get("start_date", "2021-01-01")
        end = cfg.get("end_date", "2023-12-31")
        start, end = _normalise_date_window(start, end)
        scale = cfg.get("scale", 1000)
        n_pix = cfg.get("n_pixels", 3000)

        with ThreadPoolExecutor(max_workers=3) as pool:
            f_datasets = pool.submit(build_feature_datasets, aoi, cfg)
            f_stack = pool.submit(build_gee_feature_stack, aoi, cfg)
            f_ts = pool.submit(fetch_monthly_timeseries, aoi, start, end)
            datasets = f_datasets.result()
            feature_stack = f_stack.result()
            timeseries = f_ts.result()

        df = sample_training_data(feature_stack, aoi, n_pixels=n_pix, scale=scale)
        return {
            "datasets": datasets,
            "training_df": df,
            "timeseries": timeseries,
            "aoi": aoi,
            "bbox": raw_data["bbox"],
        }

    def _run_model_dict(self, features: dict, config: dict | None = None) -> dict:
        """Train GBM + XGBoost, evaluate, compute SHAP and DBSCAN hotspots."""
        model = DiseaseModel()
        result = model.predict(
            features["training_df"],
            timeseries=features.get("timeseries", {}),
            config=config,
        )
        features["_gbm"] = model.gbm
        features["_xgb"] = model.xgb
        features["_scaler"] = model.scaler
        return result

    # ── Standalone pipeline (direct use without core runner) ───────────────

    def run(self, config: dict) -> dict:
        """Single AOI analysis — full pipeline in one call."""
        raw_data = self._fetch_from_dict(config)
        features = self._preprocess_raw(raw_data)
        result = self._run_model_dict(features, config)

        cog_paths: dict[str, str] | None = None
        cog_error: str | None = None
        try:
            cog_paths = export_disease_cog(
                gbm_model=features["_gbm"],
                xgb_model=features["_xgb"],
                scaler=features["_scaler"],
                datasets=features["datasets"],
                output_dir=config.get("output_dir", "outputs"),
                prefix=config.get("prefix", "disease"),
                model_type=config.get("model_type", "gbm"),
                aoi_geojson=config.get("aoi_geojson"),
            )
        except Exception as exc:
            _log.warning("Disease COG export failed: %s", exc, exc_info=True)
            cog_error = str(exc)
        result["raster"] = cog_paths or {}
        if cog_error:
            result["raster_error"] = cog_error
        result["stats"].update(
            {
                "bbox": features["bbox"],
                "prefix": config.get("prefix", "disease"),
            }
        )
        return result

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


register_module("disease", DiseaseRiskUseCase)
