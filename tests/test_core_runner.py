"""Tests for core/runner.py — register_module, _ensure_module_registered, run_analysis."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from climate_change.core.base_use_case import AnalysisOutput
from climate_change.core.runner import MODULE_MAP, _ensure_module_registered, register_module


class TestRegisterModule:
    def test_registers_class_in_module_map(self):
        class FakeUseCase:
            pass

        register_module("_test_fake", FakeUseCase)
        assert MODULE_MAP.get("_test_fake") is FakeUseCase

    def test_overwrite_existing_entry(self):
        class A:
            pass

        class B:
            pass

        register_module("_test_overwrite", A)
        register_module("_test_overwrite", B)
        assert MODULE_MAP["_test_overwrite"] is B


class TestEnsureModuleRegistered:
    def test_known_module_triggers_import(self):
        with patch("climate_change.core.runner.importlib.import_module") as mock_import:
            _ensure_module_registered("drought")
        mock_import.assert_called_once_with("climate_change.drought")

    def test_unknown_module_does_nothing(self):
        with patch("climate_change.core.runner.importlib.import_module") as mock_import:
            _ensure_module_registered("nonexistent_module")
        mock_import.assert_not_called()


class TestRunAnalysis:
    """run_analysis requires GEE + Dask; we mock all external calls."""

    def _make_output(self):
        return AnalysisOutput(
            module="drought",
            geojson={},
            raster_path=None,
            stats={},
            shap=None,
            charts={},
            metadata={},
        )

    def test_unknown_module_raises_value_error(self):
        from climate_change.core.runner import run_analysis

        with (
            patch("climate_change.core.runner.ensure_gee"),
            patch("climate_change.core.runner._ensure_module_registered"),
            pytest.raises(ValueError, match="Unknown module"),
        ):
            asyncio.run(
                run_analysis(
                    module="__nonexistent__",
                    aoi_geojson={},
                    start_date="2020-01-01",
                    end_date="2021-01-01",
                    country="Kenya",
                )
            )

    def test_run_analysis_calls_use_case_execute(self):
        output = self._make_output()
        mock_uc_instance = MagicMock()
        mock_uc_instance.execute = AsyncMock(return_value=output)
        mock_uc_class = MagicMock(return_value=mock_uc_instance)

        with (
            patch("climate_change.core.runner.ensure_gee"),
            patch("climate_change.core.runner._ensure_module_registered"),
            patch.dict("climate_change.core.runner.MODULE_MAP", {"drought": mock_uc_class}),
        ):
            from climate_change.core.dask_engine import DaskEngine
            from climate_change.core.runner import run_analysis

            with patch.object(DaskEngine, "get_client", return_value=MagicMock()):
                result = asyncio.run(
                    run_analysis(
                        module="drought",
                        aoi_geojson={"type": "Polygon", "coordinates": [[]]},
                        start_date="2020-01-01",
                        end_date="2021-01-01",
                        country="Kenya",
                    )
                )
        assert result is output
