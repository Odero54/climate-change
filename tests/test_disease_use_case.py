"""Tests for disease/use_case.py — _apply_raster_risk_pct."""

from climate_change.disease.use_case import _apply_raster_risk_pct


class TestApplyRasterRiskPct:
    def test_overwrites_stats_high_risk_pct(self):
        result = {"stats": {"high_risk_pct": 64.6}, "charts": {}}
        _apply_raster_risk_pct(result, [34.8, 65.2, 0.0])
        assert result["stats"]["high_risk_pct"] == 0.0

    def test_overwrites_chart_risk_dist_data(self):
        result = {
            "stats": {"high_risk_pct": 64.6},
            "charts": {
                "riskDist": {
                    "labels": ["Low Risk", "Medium Risk", "High Risk"],
                    "data": [0.0, 35.4, 64.6],
                }
            },
        }
        _apply_raster_risk_pct(result, [34.8, 65.2, 0.0])
        assert result["charts"]["riskDist"]["data"] == [34.8, 65.2, 0.0]
        # labels/order untouched
        assert result["charts"]["riskDist"]["labels"] == ["Low Risk", "Medium Risk", "High Risk"]

    def test_missing_risk_dist_is_a_noop(self):
        result = {"stats": {"high_risk_pct": 64.6}, "charts": {}}
        _apply_raster_risk_pct(result, [34.8, 65.2, 0.0])
        assert "riskDist" not in result["charts"]
