from __future__ import annotations

import logging
import os

import numpy as np

_log = logging.getLogger(__name__)

from climate_change.core.base_use_case import (
    AnalysisConfig,
    AnalysisOutput,
    BaseUseCase,
    _aoi_geometries,
    _lons_lats,
    _round_floats,
)
from climate_change.core.cache import feature_cache
from climate_change.core.dask_engine import DaskEngine
from climate_change.core.gee_auth import ensure_gee
from climate_change.core.runner import register_module

from .cdi_runner import export_cdi_cog, run_cdi_pipeline
from .model import DroughtModel


def _run_cdi_pipeline_local(raw_data: dict) -> dict:
    # xee (xarray-Earth Engine) creates lazy Dask arrays that call the GEE
    # Python client when .compute() is triggered. Keep the computation in the
    # main process and avoid distributed P2P rechunk shuffle, which requires
    # worker context and can fail inside the API request thread.
    import dask

    with dask.config.set(
        scheduler="synchronous",
        array__rechunk__method="tasks",
        dataframe__shuffle__method="tasks",
    ):
        return run_cdi_pipeline(raw_data)


# Module-level cached function avoids joblib's broken `ignore=["self"]`
# handling on bound methods in joblib >= 1.4.
@feature_cache.cache
def _run_cdi_pipeline_cached(raw_data: dict) -> dict:
    return _run_cdi_pipeline_local(raw_data)


class DroughtUseCase(BaseUseCase):
    """
    Drought risk module — wraps the drought-monitoring package via cdi_runner.py.
    Implements BaseUseCase and registers itself in core.runner.MODULE_MAP.
    """

    def __init__(self, dask_engine: DaskEngine) -> None:
        super().__init__(dask_engine)

    # ── BaseUseCase abstract methods ────────────────────────────────────────

    def fetch_data(self, config: AnalysisConfig) -> dict:
        """
        Extract bbox and year range from AnalysisConfig.
        GEE authentication and data fetching happen inside preprocess().
        """
        lons, lats = _lons_lats(config.aoi_geojson)
        return {
            "bbox": [min(lons), min(lats), max(lons), max(lats)],
            "aoi_geojson": config.aoi_geojson,
            "start_year": int(config.start_date[:4]),
            "end_year": int(config.end_date[:4]),
            "gee_project": config.extra_params.get("gee_project", ""),
        }

    def preprocess(self, raw_data: dict, config: AnalysisConfig) -> dict:
        """
        Fetch ERA5-Land + MODIS from GEE and compute CDI time-series / annual maps.
        Delegates to a module-level joblib-cached function so `self` is never
        part of the cache key (joblib >= 1.4 broke ignore=["self"] on methods).
        Coordinates are rounded to 6 d.p. before hashing to prevent float-jitter
        cache misses for semantically identical AOIs.
        """
        ensure_gee(
            raw_data.get("gee_project") or config.extra_params.get("gee_project", "")
        )
        return _run_cdi_pipeline_cached(_round_floats(raw_data))

    def run_model(self, features: dict, config: AnalysisConfig) -> AnalysisOutput:
        """
        Run DroughtModel (LSTM or drought_monitoring statistical forecast),
        build GeoJSON from the spatial CDI map, and package as AnalysisOutput.
        """
        model_type = config.extra_params.get("model_type", "lstm")
        result = DroughtModel().predict(features, {"model_type": model_type})
        geojson = self._cdi_to_geojson(features, config.aoi_geojson)
        raster_paths: dict[str, str] | None = None
        raster_error: str | None = None
        try:
            cog_paths = export_cdi_cog(
                features,
                output_dir=config.extra_params.get("output_dir", "outputs"),
                aoi_geojson=config.aoi_geojson,
            )
            raster_paths = {key: str(path) for key, path in cog_paths.items()}
        except Exception as exc:
            raster_error = str(exc)

        metadata = {
            "model": f"drought-monitoring CDI + {model_type}",
            "package_version": "0.1.7",
            "country": config.country,
            "start_date": config.start_date,
            "end_date": config.end_date,
            "raster": raster_paths or {},
            "spatial_resolution_source": "drought_monitoring.yearly_drought_maps",
        }
        if raster_error:
            metadata["raster_error"] = raster_error

        return AnalysisOutput(
            module="drought",
            geojson=geojson,
            raster_path=raster_paths,
            stats={**result["stats"], "country": config.country},
            shap=None,  # CDI is an ensemble index — no single SHAP decomposition
            charts=result["charts"],
            metadata=metadata,
        )

    # ── Analysis mode helpers ───────────────────────────────────────────────

    def run(self, config: dict) -> dict:
        """Single AOI pipeline — dict config for internal parallel use."""
        raw_data = self._fetch_from_dict(config)
        features = _run_cdi_pipeline_local(raw_data)
        result = DroughtModel().predict(features, config)

        output_dir = config.get("output_dir", "outputs")
        cog_paths = export_cdi_cog(
            features,
            output_dir,
            aoi_geojson=config.get("aoi_geojson"),
        )
        result["raster"] = {k: str(v) for k, v in cog_paths.items()}
        return result

    def run_date_ranges(self, config: dict, date_ranges: list[dict]) -> list[dict]:
        """Run the same AOI over multiple date ranges in parallel (TWO_DATE mode)."""
        merged = [{**config, **dr} for dr in date_ranges]
        return self._run_parallel(merged)

    def run_multi_regions(self, configs: list[dict]) -> list[dict]:
        """Run multiple AOI configs in parallel (MULTI_REGION mode)."""
        return self._run_parallel(configs)

    # ── Internal helpers ────────────────────────────────────────────────────

    def _run_parallel(self, configs: list[dict]) -> list[dict]:
        """Run each config on a separate Dask worker for distributed execution."""
        import traceback

        from dask.distributed import as_completed as dask_as_completed

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

    @staticmethod
    def _fetch_from_dict(config: dict) -> dict:
        """Extract raw_data dict from a plain dict config (used by run())."""
        ensure_gee(config.get("gee_project", os.environ.get("GEE_PROJECT", "")))
        lons, lats = _lons_lats(config["aoi_geojson"])
        return {
            "bbox": [min(lons), min(lats), max(lons), max(lats)],
            "aoi_geojson": config.get("aoi_geojson"),
            "start_year": int(config["start_date"][:4]),
            "end_year": int(config["end_date"][:4]),
            "gee_project": config.get("gee_project", ""),
        }

    @staticmethod
    def _cdi_to_geojson(features: dict, aoi_geojson: dict | None = None) -> dict:
        """
        Vectorise the latest annual CDI spatial map into a GeoJSON FeatureCollection.
        Each polygon represents a contiguous area of the same drought severity class.
        Falls back to a bbox feature if rasterio is unavailable.
        """
        ds = features["cdi_maps"]
        try:
            from rasterio.features import geometry_mask, shapes
            from rasterio.transform import from_bounds

            lat_dim = next(
                (dim for dim in ds["CDI"].dims if dim in ("lat", "latitude", "y")),
                "lat",
            )
            lon_dim = next(
                (dim for dim in ds["CDI"].dims if dim in ("lon", "longitude", "x")),
                "lon",
            )
            latest = ds["CDI"].isel(time=-1).transpose(lat_dim, lon_dim)
            cdi_vals = latest.values.astype(np.float32)
            lons = ds[lon_dim].values
            lats = ds[lat_dim].values

            # Classify to uint8 severity codes
            sev = np.zeros_like(cdi_vals, dtype=np.uint8)
            sev[(cdi_vals >= 0.80) & (cdi_vals < 0.90)] = 1  # Mild Drought
            sev[(cdi_vals >= 0.65) & (cdi_vals < 0.80)] = 2  # Moderate Drought
            sev[(cdi_vals >= 0.50) & (cdi_vals < 0.65)] = 3  # Severe Drought
            sev[cdi_vals < 0.50] = 4  # Extreme Drought

            transform = from_bounds(
                float(lons.min()),
                float(lats.min()),
                float(lons.max()),
                float(lats.max()),
                sev.shape[1],
                sev.shape[0],
            )
            mask = None
            if aoi_geojson:
                geometries = _aoi_geometries(aoi_geojson)
                if geometries:
                    mask = geometry_mask(
                        geometries,
                        out_shape=sev.shape,
                        transform=transform,
                        invert=True,
                    )
                    if len(lats) > 1 and float(lats[0]) < float(lats[-1]):
                        mask = mask[::-1, :]
                    sev = np.where(mask, sev, 255).astype(np.uint8)

            _labels = {
                0: "Near normal",
                1: "Mild drought",
                2: "Moderate drought",
                3: "Severe drought",
                4: "Extreme drought",
            }
            _colors = {
                0: "#E0E0E0",
                1: "#F57C00",
                2: "#FFB74D",
                3: "#E65100",
                4: "#990000",
            }

            feat_list = []
            for geom, value in shapes(sev, mask=mask, transform=transform):
                code = int(value)
                if code == 255:
                    continue
                feat_list.append(
                    {
                        "type": "Feature",
                        "geometry": geom,
                        "properties": {
                            "severity": _labels.get(code, "Unknown"),
                            "severity_code": code,
                            "color": _colors.get(code, "#CCCCCC"),
                        },
                    }
                )
            return {"type": "FeatureCollection", "features": feat_list}

        except Exception:
            _log.warning(
                "CDI rasterio vectorisation failed; falling back to bbox polygon",
                exc_info=True,
            )
            # Fallback: single bbox polygon annotated with mean CDI
            lon_name = "lon" if "lon" in ds.coords else "longitude"
            lat_name = "lat" if "lat" in ds.coords else "latitude"
            lons = ds[lon_name].values
            lats = ds[lat_name].values
            return {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [
                                [
                                    [float(lons.min()), float(lats.min())],
                                    [float(lons.max()), float(lats.min())],
                                    [float(lons.max()), float(lats.max())],
                                    [float(lons.min()), float(lats.max())],
                                    [float(lons.min()), float(lats.min())],
                                ]
                            ],
                        },
                        "properties": {"note": "CDI raster available via raster_path"},
                    }
                ],
            }


register_module("drought", DroughtUseCase)
