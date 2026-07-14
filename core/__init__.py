from typing import TYPE_CHECKING

from climate_change.core.base_use_case import (
    AnalysisConfig,
    AnalysisOutput,
    BaseUseCase,
)
from climate_change.core.cache import analysis_cache, feature_cache
from climate_change.core.gee_auth import ensure_gee, validate_gee_project
from climate_change.core.runner import MODULE_MAP, register_module, run_analysis

if TYPE_CHECKING:
    # Satisfies static analysis of __all__ below without eagerly importing
    # this heavy module at runtime — __getattr__ below does the actual lazy
    # import when the name is accessed.
    from climate_change.core.dask_engine import DaskEngine


def __getattr__(name: str):
    if name == "DaskEngine":
        from climate_change.core.dask_engine import DaskEngine

        return DaskEngine
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "AnalysisConfig",
    "AnalysisOutput",
    "BaseUseCase",
    "DaskEngine",
    "analysis_cache",
    "feature_cache",
    "register_module",
    "MODULE_MAP",
    "run_analysis",
    "ensure_gee",
    "validate_gee_project",
]
