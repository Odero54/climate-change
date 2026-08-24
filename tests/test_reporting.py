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


class TestStatsRows:
    """The "Key Statistics" section shows a curated, per-module, plain-
    language subset of output.stats — not every raw key (model-internal
    F1/AUC/CV/threshold fields are covered by the separate "Model
    Performance Comparison" section instead)."""

    def _builder(self) -> ReportBuilder:
        builder = ReportBuilder.__new__(ReportBuilder)
        builder._styles = _styles()
        return builder

    def test_food_security_shows_plain_labels_not_raw_keys(self):
        output = AnalysisOutput(
            module="food_security",
            geojson={},
            raster_path=None,
            stats={
                "model_type": "rf",
                "n_pixels_sampled": 598,
                "rf_cv_f1": 0.8745,
                "rf_f1": 0.4979,
                "rf_accuracy": 0.9917,
                "xgb_cv_f1": 0.7489,
                "ensemble_f1": 0.4979,
                "selected_f1": 0.4979,
                "high_risk_pct": 0.0,
                "medium_risk_pct": 0.7,
                "low_risk_pct": 99.3,
                "top_driver": "ndvi_slope",
                "vci_mean": 76.5,
                "tci_mean": 86.3,
                "vhi_mean": 81.4,
                "total_population": 200498.7,
                "population_affected": 428.2,
                "total_area_ha": 60787.1,
                "country": "Kenya",
                "run_duration_s": 19.5,
            },
            shap=None,
            charts={},
            metadata={},
        )
        rows = self._builder()._stats_rows(output)
        labels = [r[0] for r in rows]
        by_label = dict(rows)

        # Model-internal and generic technical fields must not appear.
        for excluded in (
            "Model Type",
            "N Pixels Sampled",
            "Rf Cv F1",
            "Rf F1",
            "Rf Accuracy",
            "Xgb Cv F1",
            "Ensemble F1",
            "Selected F1",
            "Country",
            "Run Duration S",
        ):
            assert excluded not in labels

        # Curated fields render with plain labels and formatted values.
        assert by_label["Overall vegetation health (0-100)"] == "81.4"
        assert by_label["Area at high risk"] == "0%"
        assert by_label["Area at low risk"] == "99.3%"
        assert by_label["Main driver of food insecurity"] == "ndvi_slope"
        assert by_label["People living in the area"] == "200,499"
        assert by_label["People at risk"] == "428"
        assert by_label["Area covered"] == "60,787.1 ha"

    def test_flood_curated_fields(self):
        output = AnalysisOutput(
            module="flood",
            geojson={},
            raster_path=None,
            stats={
                "flooded_pct": 12.3,
                "very_high_risk_pct": 4.0,
                "high_risk_pct": 8.0,
                "top_flood_driver": "vv_change",
                "mean_spread": 0.05,
                "spread_stats": {"min": 0.0, "max": 0.3},
            },
            shap=None,
            charts={},
            metadata={},
        )
        rows = self._builder()._stats_rows(output)
        by_label = dict(rows)
        assert by_label["Area currently flooded"] == "12.3%"
        assert by_label["Main driver of flood risk"] == "vv_change"
        # Nested/non-curated fields (spread_stats, mean_spread) are omitted.
        assert "Spread Stats" not in by_label
        assert "Mean Spread" not in by_label

    def test_missing_curated_keys_falls_back_to_raw_dump(self):
        """A module with none of its curated keys present (e.g. an old
        cached result, or an unrecognised module) still shows something
        rather than an empty table."""
        output = AnalysisOutput(
            module="food_security",
            geojson={},
            raster_path=None,
            stats={"some_future_field": 1.0},
            shap=None,
            charts={},
            metadata={},
        )
        rows = self._builder()._stats_rows(output)
        assert rows == [["Some Future Field", "1.0"]]

    def test_none_value_renders_as_not_available(self):
        output = AnalysisOutput(
            module="land_degradation",
            geojson={},
            raster_path=None,
            stats={"ndvi_trend_per_year": None, "degraded_label_pct": 10.0},
            shap=None,
            charts={},
            metadata={},
        )
        rows = self._builder()._stats_rows(output)
        by_label = dict(rows)
        assert by_label["Vegetation trend (per year)"] == "Not available"

    def test_boolean_value_renders_as_yes_no(self):
        output = AnalysisOutput(
            module="land_degradation",
            geojson={},
            raster_path=None,
            stats={"mk_significant": True},
            shap=None,
            charts={},
            metadata={},
        )
        rows = self._builder()._stats_rows(output)
        assert dict(rows)["Trend is statistically significant"] == "Yes"
