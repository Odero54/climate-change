"""Tests for land_degradation/use_case.py — _apply_raster_risk_pct."""

from climate_change.land_degradation.use_case import _apply_raster_risk_pct


class TestApplyRasterRiskPct:
    def test_overwrites_chart_risk_dist_data_and_ha(self):
        result = {
            "charts": {
                "riskDist": {
                    "labels": ["Not Degraded", "Degraded"],
                    "data": [70.0, 30.0],
                    "data_ha": [700.0, 300.0],
                }
            }
        }
        _apply_raster_risk_pct(result, [45.2, 54.8], [4520.0, 5480.0])
        assert result["charts"]["riskDist"]["data"] == [45.2, 54.8]
        assert result["charts"]["riskDist"]["data_ha"] == [4520.0, 5480.0]
        # labels/order untouched
        assert result["charts"]["riskDist"]["labels"] == ["Not Degraded", "Degraded"]

    def test_missing_risk_dist_is_a_noop(self):
        result = {"charts": {}}
        _apply_raster_risk_pct(result, [45.2, 54.8], [4520.0, 5480.0])
        assert "riskDist" not in result["charts"]
