"""
climate_change — ARIN Climate Resilience Decision Support System
Core computation package. All analysis logic lives here.
The FastAPI backend calls this package; it does no computation itself.

Usage:
    from climate_change import run_analysis, USE_CASE_REGISTRY

    result = await run_analysis(
        module="drought",
        aoi_geojson={"type": "Polygon", "coordinates": [...]},
        start_date="2010-01-01",
        end_date="2023-12-31",
        country="Kenya",
    )
"""

from __future__ import annotations

# Use absolute imports so this file is safe to load without a package context
# (pytest traverses parent __init__.py files during test discovery).
from climate_change.ai_interpreter import (
    AIInterpreter,
    build_interpretation_prompt,
)
from climate_change.core import (
    MODULE_MAP,
    AnalysisConfig,
    AnalysisOutput,
    BaseUseCase,
    analysis_cache,
    ensure_gee,
    feature_cache,
    register_module,
    validate_gee_project,
)
from climate_change.registry import (
    USE_CASE_REGISTRY,
    ModelOption,
    UseCaseInfo,
    get_use_case_info,
)
from climate_change.reporting import ReportBuilder


def __getattr__(name: str):
    if name == "DaskEngine":
        from climate_change.core.dask_engine import DaskEngine

        return DaskEngine
    if name == "DroughtUseCase":
        from climate_change.drought import DroughtUseCase

        return DroughtUseCase
    if name == "FloodRiskUseCase":
        from climate_change.flood import FloodRiskUseCase

        return FloodRiskUseCase
    if name == "FoodSecurityUseCase":
        from climate_change.food_security import FoodSecurityUseCase

        return FoodSecurityUseCase
    if name == "DiseaseRiskUseCase":
        from climate_change.disease import DiseaseRiskUseCase

        return DiseaseRiskUseCase
    if name == "LandDegradationUseCase":
        from climate_change.land_degradation import LandDegradationUseCase

        return LandDegradationUseCase
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


async def run_analysis(
    module: str,
    aoi_geojson: dict,
    start_date: str,
    end_date: str,
    country: str,
    gee_project: str = "",
    extra_params: dict | None = None,
    openai_api_key: str | None = None,
    report_output_dir: str | None = None,
    map_png_bytes: bytes | None = None,
):
    from climate_change.core.runner import run_analysis as _run

    return await _run(
        module=module,
        aoi_geojson=aoi_geojson,
        start_date=start_date,
        end_date=end_date,
        country=country,
        gee_project=gee_project,
        extra_params=extra_params,
        openai_api_key=openai_api_key,
        report_output_dir=report_output_dir,
        map_png_bytes=map_png_bytes,
    )


__all__ = [
    # entry point
    "run_analysis",
    # core
    "AnalysisConfig",
    "AnalysisOutput",
    "BaseUseCase",
    "DaskEngine",
    "MODULE_MAP",
    "analysis_cache",
    "feature_cache",
    "register_module",
    "ensure_gee",
    "validate_gee_project",
    # registry
    "USE_CASE_REGISTRY",
    "UseCaseInfo",
    "ModelOption",
    "get_use_case_info",
    # reporting
    "ReportBuilder",
    # ai interpreter
    "AIInterpreter",
    "build_interpretation_prompt",
    # domain use-cases
    "DroughtUseCase",
    "FloodRiskUseCase",
    "FoodSecurityUseCase",
    "DiseaseRiskUseCase",
    "LandDegradationUseCase",
]
__version__ = "1.0.0"
