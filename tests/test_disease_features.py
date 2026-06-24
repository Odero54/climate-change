"""Tests for disease/features.py — constants and _normalise_date_window."""

from datetime import datetime, timedelta, timezone

from climate_change.disease.features import (
    DISEASE_CLASSES,
    DISEASE_COLORS,
    FEATURE_COLS,
    RISK_PERCENTILES,
    SCORE_WEIGHTS,
    _normalise_date_window,
)


class TestFeatureCols:
    def test_seven_features(self):
        assert len(FEATURE_COLS) == 7

    def test_expected_features_present(self):
        for col in ("rainfall_4w", "temp_mean", "ndwi", "elevation", "pop_density"):
            assert col in FEATURE_COLS


class TestDiseaseClasses:
    def test_three_classes(self):
        assert len(DISEASE_CLASSES) == 3

    def test_risk_order(self):
        assert DISEASE_CLASSES[0] == "Low Risk"
        assert DISEASE_CLASSES[-1] == "High Risk"


class TestDiseaseColors:
    def test_one_color_per_class(self):
        assert len(DISEASE_COLORS) == len(DISEASE_CLASSES)

    def test_colors_are_hex(self):
        for c in DISEASE_COLORS:
            assert c.startswith("#")


class TestScoreWeights:
    def test_weights_sum_to_one(self):
        total = sum(SCORE_WEIGHTS.values())
        assert abs(total - 1.0) < 1e-9

    def test_all_positive(self):
        for w in SCORE_WEIGHTS.values():
            assert w > 0

    def test_expected_keys(self):
        for k in ("temp_suit", "rain_suit", "ndwi_score"):
            assert k in SCORE_WEIGHTS


class TestRiskPercentiles:
    def test_two_values(self):
        assert len(RISK_PERCENTILES) == 2

    def test_ordered(self):
        assert RISK_PERCENTILES[0] < RISK_PERCENTILES[1]


class TestNormaliseDateWindow:
    def test_future_end_capped_to_safe_date(self):
        far_future = (datetime.now(timezone.utc).date() + timedelta(days=30)).isoformat()
        start, end = _normalise_date_window("2024-01-01", far_future)
        safe = (datetime.now(timezone.utc).date() - timedelta(days=7)).isoformat()
        assert end <= safe

    def test_start_after_end_pushed_back(self):
        # start > latest_safe triggers automatic adjustment
        today = datetime.now(timezone.utc).date()
        start = today.isoformat()
        end = today.isoformat()
        s, e = _normalise_date_window(start, end, minimum_days=90)
        assert s < e

    def test_normal_dates_unchanged(self):
        start, end = _normalise_date_window("2022-01-01", "2022-06-01")
        assert start == "2022-01-01"
        assert end == "2022-06-01"

    def test_end_date_before_start_after_capping(self):
        # Even if end is capped, start must be before end
        far_future = (datetime.now(timezone.utc).date() + timedelta(days=1)).isoformat()
        s, e = _normalise_date_window(far_future, far_future, minimum_days=30)
        assert s < e
