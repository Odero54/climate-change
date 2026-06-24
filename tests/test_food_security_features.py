"""Tests for food_security/features.py — constants and pure utilities."""

from climate_change.food_security.features import (
    FEATURE_COLS,
    FOOD_CLASSES,
    FOOD_COLORS,
    RISK_PERCENTILES,
    SCORE_WEIGHTS,
)


class TestFeatureCols:
    def test_is_list(self):
        assert isinstance(FEATURE_COLS, list)

    def test_expected_features_present(self):
        for col in ("vci", "tci", "rainfall_anom_pct", "ndvi_slope", "mndwi"):
            assert col in FEATURE_COLS

    def test_seven_features(self):
        assert len(FEATURE_COLS) == 7


class TestFoodClasses:
    def test_three_classes(self):
        assert len(FOOD_CLASSES) == 3

    def test_classes_ordered_by_severity(self):
        assert FOOD_CLASSES[0] == "Low Risk"
        assert FOOD_CLASSES[-1] == "High Risk"


class TestFoodColors:
    def test_one_color_per_class(self):
        assert len(FOOD_COLORS) == len(FOOD_CLASSES)

    def test_colors_are_hex(self):
        for color in FOOD_COLORS:
            assert color.startswith("#"), f"Not a hex color: {color}"


class TestScoreWeights:
    def test_weights_sum_to_one(self):
        total = sum(SCORE_WEIGHTS.values())
        assert abs(total - 1.0) < 1e-9

    def test_all_positive(self):
        for key, w in SCORE_WEIGHTS.items():
            assert w > 0, f"Weight for {key} is not positive"

    def test_expected_keys(self):
        for key in ("vci_stress", "tci_stress", "rain_deficit", "slope_inv"):
            assert key in SCORE_WEIGHTS


class TestRiskPercentiles:
    def test_two_thresholds(self):
        assert len(RISK_PERCENTILES) == 2

    def test_lower_less_than_upper(self):
        assert RISK_PERCENTILES[0] < RISK_PERCENTILES[1]

    def test_both_between_0_and_1(self):
        for p in RISK_PERCENTILES:
            assert 0.0 < p < 1.0
