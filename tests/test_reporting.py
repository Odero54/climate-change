"""Tests for reporting/report_builder.py — _styles and ReportBuilder."""

import io

import pytest

from climate_change.core.base_use_case import AnalysisOutput
from climate_change.reporting.report_builder import ReportBuilder, _styles

# module -> (risk chart key, at-risk labels) mirroring each domain's
# use_case.py AT_RISK_LABELS constants and the chart key core.population's
# data_population is attached to.
_POPULATION_CHART_KEY = {
    "disease": "riskDist",
    "food_security": "riskDist",
    "flood": "risk_distribution",
    "land_degradation": "riskDist",
    "drought": "severity_distribution",
}


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
            "severity_distribution": {
                "labels": ["Near normal"],
                "data": [95.0],
                "colors": ["#E0E0E0"],
            },
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
        builder.build(_make_output("drought"), ai_interpretation=None, map_png_bytes=None)
        assert output_path.exists()
        assert output_path.stat().st_size > 0

    def test_build_with_ai_text(self, tmp_path):
        output_path = tmp_path / "report_ai.pdf"
        builder = ReportBuilder(output_path)
        builder.build(
            _make_output("flood"),
            ai_interpretation="Flood risk is moderate in the region.",
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
        builder.build(
            _make_output("food_security"), ai_interpretation=None, map_png_bytes=map_bytes
        )
        assert output_path.exists()

    @pytest.mark.parametrize(
        "module_id", ["drought", "flood", "food_security", "disease", "land_degradation"]
    )
    def test_build_all_modules(self, tmp_path, module_id):
        output_path = tmp_path / f"{module_id}.pdf"
        builder = ReportBuilder(output_path)
        builder.build(_make_output(module_id), ai_interpretation=None, map_png_bytes=None)
        assert output_path.exists()
        assert output_path.stat().st_size > 0


class TestPopulationExposureSection:
    @pytest.mark.parametrize(
        "module_id", ["drought", "flood", "food_security", "disease", "land_degradation"]
    )
    def test_build_with_population_data(self, tmp_path, module_id):
        """Population fields, best-effort attached by core.population, must
        render into the PDF without crashing regardless of which domain's
        risk chart key (riskDist / risk_distribution / severity_distribution)
        they land on."""
        output = _make_output(module_id)
        output.stats["total_population"] = 100_000.0
        output.stats["population_affected"] = 42_000.0
        chart_key = _POPULATION_CHART_KEY[module_id]
        chart = output.charts.setdefault(
            chart_key, {"labels": ["A"], "data": [100.0], "colors": ["#2ECC71"]}
        )
        chart["data_population"] = [42_000.0] if len(chart["labels"]) == 1 else chart["data"]

        output_path = tmp_path / f"{module_id}_population.pdf"
        builder = ReportBuilder(output_path)
        builder.build(output, ai_interpretation=None, map_png_bytes=None)
        assert output_path.exists()
        assert output_path.stat().st_size > 0

    @pytest.mark.parametrize(
        "module_id", ["drought", "flood", "food_security", "disease", "land_degradation"]
    )
    def test_build_without_population_data_omits_section(self, tmp_path, module_id):
        """total_population/population_affected are best-effort (core.population
        fetch may fail) — their absence must not crash report generation."""
        output = _make_output(module_id)
        assert "total_population" not in output.stats

        output_path = tmp_path / f"{module_id}_no_population.pdf"
        builder = ReportBuilder(output_path)
        builder.build(output, ai_interpretation=None, map_png_bytes=None)
        assert output_path.exists()
        assert output_path.stat().st_size > 0

    def test_chart_population_dist_returns_none_without_data(self):
        builder = ReportBuilder.__new__(ReportBuilder)
        builder._styles = _styles()
        assert builder._chart_population_dist([], [], [], "title") is None

    def test_chart_population_dist_returns_image_with_data(self):
        builder = ReportBuilder.__new__(ReportBuilder)
        builder._styles = _styles()
        img = builder._chart_population_dist(
            ["Low", "High"], [1000.0, 5000.0], ["#2ECC71", "#E74C3C"], "title"
        )
        assert img is not None

    def test_add_population_section_noop_when_total_population_missing(self):
        builder = ReportBuilder.__new__(ReportBuilder)
        builder._styles = _styles()
        story: list = []
        builder._add_population_section(
            story,
            section_num=5,
            figure_num=3,
            classification_label="disease-risk",
            stats={},
            chart={"labels": ["Low"], "data": [100.0]},
            at_risk_labels=["Medium Risk", "High Risk"],
            chart_title="title",
        )
        assert story == []

    def test_add_population_section_appends_when_present(self):
        builder = ReportBuilder.__new__(ReportBuilder)
        builder._styles = _styles()
        story: list = []
        builder._add_population_section(
            story,
            section_num=5,
            figure_num=3,
            classification_label="disease-risk",
            stats={"total_population": 100_000.0, "population_affected": 42_000.0},
            chart={
                "labels": ["Low Risk", "Medium Risk", "High Risk"],
                "data": [30.0, 30.0, 40.0],
                "colors": ["#2ECC71", "#F1C40F", "#E74C3C"],
                "data_population": [10_000.0, 48_000.0, 42_000.0],
            },
            at_risk_labels=["Medium Risk", "High Risk"],
            chart_title="title",
        )
        assert len(story) > 0
