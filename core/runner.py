from __future__ import annotations

import importlib
import os

from climate_change.core.base_use_case import (
    AnalysisConfig,
    AnalysisOutput,
    BaseUseCase,
)
from climate_change.core.gee_auth import ensure_gee

# Populated by each module's __init__.py via register_module()
MODULE_MAP: dict[str, type[BaseUseCase]] = {}
_MODULE_IMPORTS = {
    "drought": "climate_change.drought",
    "food_security": "climate_change.food_security",
    "flood": "climate_change.flood",
    "disease": "climate_change.disease",
    "land_degradation": "climate_change.land_degradation",
}


def register_module(name: str, cls: type[BaseUseCase]) -> None:
    """Register a use-case class under the given module name."""
    MODULE_MAP[name] = cls


def _ensure_module_registered(module: str) -> None:
    module_path = _MODULE_IMPORTS.get(module)
    if module_path:
        importlib.import_module(module_path)


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
) -> AnalysisOutput:
    """
    Main public API of the climate_change package.
    The Celery task calls only this function — no other file in the
    infrastructure layer imports climate_change directly.

    Parameters
    ----------
    gee_project:
        Google Earth Engine Cloud project ID.  Falls back to the
        ``GEE_PROJECT`` environment variable when not supplied.
        Authentication runs via ``drought_monitoring.gee.authenticate``
        before any use-case logic executes.

    Optional:
        openai_api_key: if provided, GPT-4o interprets the results and the
            text is stored in output.metadata["ai_interpretation"].
        report_output_dir: if provided, a PDF report is written to this
            directory and the path stored in output.metadata["report_path"].
        map_png_bytes: optional PNG screenshot embedded in the PDF report.
    """
    _ensure_module_registered(module)
    if module not in MODULE_MAP:
        raise ValueError(f"Unknown module: '{module}'. Available: {sorted(MODULE_MAP)}")

    # ── GEE authentication — happens once per process before any use-case ──────
    project = gee_project or os.environ.get("GEE_PROJECT", "")
    ensure_gee(project)

    # Inject resolved project into extra_params so each use-case run() can
    # access it without re-deriving from the environment.
    params = dict(extra_params or {})
    params.setdefault("gee_project", project)

    from climate_change.core.dask_engine import DaskEngine

    dask_engine = DaskEngine()
    dask_engine.get_client()  # start cluster (idempotent)

    config = AnalysisConfig(
        module=module,
        aoi_geojson=aoi_geojson,
        start_date=start_date,
        end_date=end_date,
        country=country,
        extra_params=params,
    )

    use_case: BaseUseCase = MODULE_MAP[module](dask_engine)
    output = await use_case.execute(config)

    ai_text: str | None = None
    if openai_api_key:
        from climate_change.ai_interpreter.interpreter import AIInterpreter

        try:
            ai_text = AIInterpreter(openai_api_key).interpret(output)
            output.metadata["ai_interpretation"] = ai_text
        except Exception as exc:
            output.metadata["ai_interpretation_error"] = str(exc)

    if report_output_dir:
        from datetime import datetime, timezone
        from pathlib import Path

        from climate_change.reporting.report_builder import ReportBuilder

        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        report_path = Path(report_output_dir) / f"{module}_{ts}.pdf"
        ReportBuilder(report_path).build(output, ai_text, map_png_bytes)
        output.metadata["report_path"] = str(report_path)

    return output
