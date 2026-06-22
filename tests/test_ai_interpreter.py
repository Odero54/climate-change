"""Tests for ai_interpreter/interpreter.py — prompt builders and AIInterpreter."""

from unittest.mock import MagicMock, patch

import pytest

from climate_change.ai_interpreter.interpreter import (
    AIInterpreter,
    build_disease_prompt,
    build_drought_prompt,
    build_flood_prompt,
    build_food_security_prompt,
    build_interpretation_prompt,
    build_land_degradation_prompt,
    build_prompt,
)
from climate_change.core.base_use_case import AnalysisOutput


def _make_output(
    module: str, charts: dict | None = None, stats: dict | None = None
) -> AnalysisOutput:
    return AnalysisOutput(
        module=module,
        geojson={},
        raster_path=None,
        stats=stats or {"mean_cdi": 0.9},
        shap=None,
        charts=charts or {},
        metadata={"country": "Kenya", "start_date": "2020-01-01", "end_date": "2023-12-31"},
    )


class TestBuildInterpretationPrompt:
    def test_contains_module_name(self):
        output = _make_output("drought")
        prompt = build_interpretation_prompt(output)
        assert "Drought" in prompt

    def test_contains_country(self):
        output = _make_output("flood")
        prompt = build_interpretation_prompt(output)
        assert "Kenya" in prompt

    def test_contains_analysis_period(self):
        output = _make_output("food_security")
        prompt = build_interpretation_prompt(output)
        assert "2020-01-01" in prompt

    def test_contains_required_sections(self):
        output = _make_output("disease")
        prompt = build_interpretation_prompt(output)
        assert "SUMMARY" in prompt
        assert "KEY DRIVERS" in prompt
        assert "RECOMMENDATIONS" in prompt
        assert "CAVEATS" in prompt

    def test_includes_stats(self):
        output = _make_output("drought", stats={"my_stat": 42})
        prompt = build_interpretation_prompt(output)
        assert "my_stat" in prompt


class TestBuildDroughtPrompt:
    def test_includes_base_prompt(self):
        output = _make_output("drought")
        prompt = build_drought_prompt(output)
        assert "SUMMARY" in prompt

    def test_forecast_appended_when_present(self):
        charts = {"forecast": {"mean": [0.8, 0.9], "dates": ["2024-01", "2024-02"]}}
        output = _make_output("drought", charts=charts)
        prompt = build_drought_prompt(output)
        assert "FORECAST" in prompt

    def test_severity_appended_when_present(self):
        charts = {"severity_distribution": {"labels": ["Extreme drought"], "data": [30.0]}}
        output = _make_output("drought", charts=charts)
        prompt = build_drought_prompt(output)
        assert "SEVERITY DISTRIBUTION" in prompt


class TestBuildFloodPrompt:
    def test_includes_base_prompt(self):
        output = _make_output("flood")
        prompt = build_flood_prompt(output)
        assert "SUMMARY" in prompt

    def test_risk_distribution_appended(self):
        charts = {"risk_distribution": {"labels": ["High"], "data": [60.0]}}
        output = _make_output("flood", charts=charts)
        prompt = build_flood_prompt(output)
        assert "RISK CLASS DISTRIBUTION" in prompt

    def test_shap_driver_appended(self):
        charts = {"shap": {"features": ["elevation"], "mean_abs_shap": [0.3]}}
        output = _make_output("flood", charts=charts)
        prompt = build_flood_prompt(output)
        assert "TOP FLOOD DRIVER" in prompt


class TestBuildLandDegradationPrompt:
    def test_includes_base_prompt(self):
        output = _make_output("land_degradation")
        prompt = build_land_degradation_prompt(output)
        assert "SUMMARY" in prompt

    def test_ndvi_trend_appended(self):
        charts = {"trend": {"ndvi_trend_per_year": -0.005, "mk_significant": True}}
        output = _make_output("land_degradation", charts=charts)
        prompt = build_land_degradation_prompt(output)
        assert "NDVI TREND" in prompt


class TestBuildDiseasePrompt:
    def test_includes_base_prompt(self):
        output = _make_output("disease")
        prompt = build_disease_prompt(output)
        assert "SUMMARY" in prompt

    def test_hotspot_clusters_appended(self):
        stats = {"n_hotspot_clusters": 3, "hotspot_population": 12000}
        output = _make_output("disease", stats=stats)
        prompt = build_disease_prompt(output)
        assert "HOTSPOT CLUSTERS" in prompt


class TestBuildFoodSecurityPrompt:
    def test_includes_base_prompt(self):
        output = _make_output("food_security")
        prompt = build_food_security_prompt(output)
        assert "SUMMARY" in prompt

    def test_vci_tci_vhi_appended(self):
        stats = {"vci_mean": 55.0, "tci_mean": 60.0, "vhi_mean": 57.5}
        output = _make_output("food_security", stats=stats)
        prompt = build_food_security_prompt(output)
        assert "VEGETATION HEALTH INDICES" in prompt


class TestBuildPromptDispatch:
    @pytest.mark.parametrize(
        "module_id", ["drought", "flood", "land_degradation", "disease", "food_security"]
    )
    def test_returns_non_empty_string(self, module_id):
        output = _make_output(module_id)
        result = build_prompt(output)
        assert isinstance(result, str)
        assert len(result) > 100


class TestAIInterpreter:
    def _mock_openai_response(self, text="Test interpretation"):
        mock_resp = MagicMock()
        mock_resp.choices[0].message.content = text
        return mock_resp

    def test_interpret_returns_string(self):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = self._mock_openai_response()
        with patch("climate_change.ai_interpreter.interpreter.OpenAI", return_value=mock_client):
            ai = AIInterpreter(api_key="test-key")
            result = ai.interpret(_make_output("drought"))
        assert result == "Test interpretation"

    def test_interpret_calls_gpt4o(self):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = self._mock_openai_response()
        with patch("climate_change.ai_interpreter.interpreter.OpenAI", return_value=mock_client):
            ai = AIInterpreter(api_key="test-key")
            ai.interpret(_make_output("flood"))
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["model"] == "gpt-4o"

    def test_openai_error_raises_runtime_error(self):
        from openai import OpenAIError

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = OpenAIError("api down")
        with patch("climate_change.ai_interpreter.interpreter.OpenAI", return_value=mock_client):
            ai = AIInterpreter(api_key="test-key")
            with pytest.raises(RuntimeError, match="OpenAI interpretation failed"):
                ai.interpret(_make_output("drought"))

    def test_chat_returns_string(self):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = self._mock_openai_response("Follow-up")
        with patch("climate_change.ai_interpreter.interpreter.OpenAI", return_value=mock_client):
            ai = AIInterpreter(api_key="test-key")
            result = ai.chat(
                _make_output("drought"), history=[], user_message="What does this mean?"
            )
        assert result == "Follow-up"

    def test_repr_redacts_key(self):
        with patch("climate_change.ai_interpreter.interpreter.OpenAI"):
            ai = AIInterpreter(api_key="super-secret")
        assert "super-secret" not in repr(ai)
        assert "redacted" in repr(ai)
