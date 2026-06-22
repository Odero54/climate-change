from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from climate_change.core.cache import analysis_cache

if TYPE_CHECKING:
    import ee

    from climate_change.core.dask_engine import DaskEngine


def _round_floats(obj: object, decimals: int = 6) -> object:
    """Recursively round all floats in a JSON-like structure to a fixed precision."""
    if isinstance(obj, float):
        return round(obj, decimals)
    if isinstance(obj, list):
        return [_round_floats(v, decimals) for v in obj]
    if isinstance(obj, dict):
        return {k: _round_floats(v, decimals) for k, v in obj.items()}
    return obj


def _lons_lats(aoi_geojson: dict) -> tuple[list[float], list[float]]:
    """Extract lon/lat coordinate lists from any GeoJSON type.

    Handles Polygon, MultiPolygon, Feature, and FeatureCollection inputs.
    Raises ValueError for unrecognised or empty geometries.
    """
    geo_type = aoi_geojson.get("type")
    if geo_type == "FeatureCollection":
        features = aoi_geojson.get("features", [])
        if not features or not features[0].get("geometry"):
            raise ValueError("FeatureCollection has no usable geometry")
        aoi_geojson = features[0]["geometry"]
        geo_type = aoi_geojson.get("type")
    elif geo_type == "Feature":
        aoi_geojson = aoi_geojson.get("geometry") or {}
        geo_type = aoi_geojson.get("type")

    coords = aoi_geojson.get("coordinates")
    if not coords:
        raise ValueError(f"Cannot extract coordinates from GeoJSON type '{geo_type}'")

    ring = coords[0][0] if geo_type == "MultiPolygon" else coords[0]
    return [p[0] for p in ring], [p[1] for p in ring]


def _ee_geometry_from_geojson(aoi_geojson: dict) -> "ee.Geometry":
    """Convert any GeoJSON type to an ee.Geometry preserving the exact polygon shape.

    Handles Feature, FeatureCollection, Polygon, and MultiPolygon inputs.
    Called after ensure_gee() so the lazy `import ee` is always safe.
    """
    import ee  # local import — ee must be initialised before calling this

    geometry = (
        aoi_geojson.get("geometry")
        if aoi_geojson.get("type") == "Feature"
        else aoi_geojson
    )
    if geometry and geometry.get("type") == "FeatureCollection":
        features = geometry.get("features", [])
        if features:
            geometry = features[0].get("geometry")
    return ee.Geometry(geometry)


def _aoi_geometries(aoi_geojson: dict | None) -> list[dict]:
    """Return a list of bare geometry dicts usable by rasterio.features.geometry_mask.

    Handles Polygon, MultiPolygon, Feature, and FeatureCollection inputs.
    Returns [] when aoi_geojson is None or has no usable geometry.
    """
    if not aoi_geojson:
        return []
    geo_type = aoi_geojson.get("type")
    if geo_type == "FeatureCollection":
        return [
            feature["geometry"]
            for feature in aoi_geojson.get("features", [])
            if feature.get("geometry")
        ]
    if geo_type == "Feature":
        geometry = aoi_geojson.get("geometry")
        return [geometry] if geometry else []
    if geo_type in {"Polygon", "MultiPolygon"}:
        return [aoi_geojson]
    return []


@dataclass
class AnalysisConfig:
    module: str
    aoi_geojson: dict  # GeoJSON Polygon
    start_date: str  # ISO: YYYY-MM-DD
    end_date: str
    country: str
    extra_params: dict = field(default_factory=dict)  # module-specific overrides


@dataclass
class AnalysisOutput:
    module: str
    geojson: dict  # GeoJSON FeatureCollection with risk/severity scores
    raster_path: str | dict[str, str] | None  # Local/exported COG GeoTIFF path(s)
    stats: dict  # summary statistics for the frontend
    shap: dict | None  # SHAP feature importances (None for drought)
    charts: dict  # chart data payloads (time-series, distribution, etc.)
    metadata: dict  # model name, feature list, run duration, country


class BaseUseCase(ABC):
    """
    Abstract base for all five climate risk use-case modules.
    Subclasses must implement fetch_data, preprocess, and run_model.
    execute() orchestrates the full pipeline with caching.
    """

    def __init__(self, dask_engine: DaskEngine) -> None:
        self.dask = dask_engine

    @abstractmethod
    def fetch_data(self, config: AnalysisConfig) -> dict:
        """Fetch raw EO data from GEE / CHIRPS / ERA5 as xr.Dataset."""

    @abstractmethod
    def preprocess(self, raw_data: dict, config: AnalysisConfig) -> dict:
        """
        Clip to AOI, compute indices, build Dask feature arrays.
        Must be decorated with @feature_cache.cache in subclasses.
        Must return in-memory feature dict — no disk writes.
        """

    @abstractmethod
    def run_model(self, features: dict, config: AnalysisConfig) -> AnalysisOutput:
        """Run ML model and return AnalysisOutput with GeoJSON + stats + charts."""

    def explain(self, features: dict) -> dict:
        """Override in subclasses that support SHAP explanations."""
        return {}

    async def execute(self, config: AnalysisConfig) -> AnalysisOutput:
        """
        Main entry point. Called by core.runner.run_analysis().
        Checks in-process cache first; runs full pipeline on miss.
        Each sync stage runs in a thread pool so the event loop stays free.
        """
        import asyncio

        cache_key = self._cache_key(config)
        cached = analysis_cache.get(cache_key)
        if cached:
            return cached

        raw = await asyncio.to_thread(self.fetch_data, config)
        features = await asyncio.to_thread(self.preprocess, raw, config)
        output = await asyncio.to_thread(self.run_model, features, config)
        explain_payload = await asyncio.to_thread(self.explain, features)
        output.shap = explain_payload or output.shap or output.charts.get("shap")

        analysis_cache.set(cache_key, output, expire=3600)
        return output

    @staticmethod
    def _cache_key(config: AnalysisConfig) -> str:
        payload = json.dumps(
            {
                "module": config.module,
                "start": config.start_date,
                "end": config.end_date,
                "aoi": _round_floats(config.aoi_geojson),
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]
