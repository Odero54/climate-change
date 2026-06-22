"""Tests for reporting/report_builder.py — _styles and ReportBuilder."""
import io
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from core.base_use_case import AnalysisOutput
from reporting.report_builder import ReportBuilder, _styles


def _make_output(module="drought") -> AnalysisOutput:
    return AnalysisOutput(
        module=module,
        geojson={},
        raster_path=None,
        stats={"mean_cdi": 0.9, "extreme_pct": 5.0},
        shap=None,
        charts={
            "timeseries": {
                "labels": ["2020-01"],
                "datasets": [{"label": "CDI", "data": [0.9], "color": "#C0392B"}],
            },
            "severity_distribution": {"labels": ["Near normal"], "data": [95.0], "colors": ["#E0E0E0"]},
        },
        metadata={
            "country": "Kenya",
            "start_date": "2020-01-01",
            "end_date": "2023-12-31",
            "model": "lstm",
        },
    )


class TestStyles:
    def test_returns_dict(self):
        result = _styles()
        assert isinstance(result, dict)

    def test_expected_style_keys(self):
        result = _styles()
        for key in ("cover_title", "section_heading", "body"):
            assert key in result


class TestReportBuilderBuild:
    def test_build_creates_pdf_bytes(self, tmp_path):
        output_path = tmp_path / "report.pdf"
        builder = ReportBuilder(output_path)
        builder.build(_make_output("drought"), ai_text=None, map_png_bytes=None)
        assert output_path.exists()
        assert output_path.stat().st_size > 0

    def test_build_with_ai_text(self, tmp_path):
        output_path = tmp_path / "report_ai.pdf"
        builder = ReportBuilder(output_path)
        builder.build(
            _make_output("flood"),
            ai_text="Flood risk is moderate in the region.",
            map_png_bytes=None,
        )
        assert output_path.exists()

    def test_build_with_map_png(self, tmp_path):
        import numpy as np
        from PIL import Image as PILImage

        img = PILImage.fromarray(np.zeros((100, 100, 3), dtype=np.uint8))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        map_bytes = buf.getvalue()

        output_path = tmp_path / "report_map.pdf"
        builder = ReportBuilder(output_path)
        builder.build(_make_output("food_security"), ai_text=None, map_png_bytes=map_bytes)
        assert output_path.exists()

    @pytest.mark.parametrize("module_id", [
        "drought", "flood", "food_security", "disease", "land_degradation"
    ])
    def test_build_all_modules(self, tmp_path, module_id):
        output_path = tmp_path / f"{module_id}.pdf"
        builder = ReportBuilder(output_path)
        builder.build(_make_output(module_id), ai_text=None, map_png_bytes=None)
        assert output_path.exists()
        assert output_path.stat().st_size > 0
