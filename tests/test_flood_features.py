"""Tests for flood/features.py — pure utility functions (no GEE)."""

from climate_change.flood.features import (
    FEATURE_COLS,
    RISK_INT,
    _is_gee_download_too_large,
    _next_download_scale,
)


class TestFeatureCols:
    def test_feature_cols_is_list(self):
        assert isinstance(FEATURE_COLS, list)

    def test_expected_features_present(self):
        for col in ("elevation", "twi", "dist_river", "mndwi"):
            assert col in FEATURE_COLS

    def test_ten_features(self):
        assert len(FEATURE_COLS) == 10


class TestRiskInt:
    def test_four_classes(self):
        assert len(RISK_INT) == 4

    def test_labels_present(self):
        for label in ("Low", "Medium", "High", "Very High"):
            assert label in RISK_INT

    def test_ordering(self):
        assert RISK_INT["Low"] < RISK_INT["Medium"] < RISK_INT["High"] < RISK_INT["Very High"]


class TestIsGeeDownloadTooLarge:
    def test_matching_markers_returns_true(self):
        exc = Exception("Total request size must be less than or equal to 50MB")
        assert _is_gee_download_too_large(exc) is True

    def test_irrelevant_exception_returns_false(self):
        exc = Exception("Some other error")
        assert _is_gee_download_too_large(exc) is False

    def test_getpixels_url_with_400_returns_true(self):
        exc = Exception("400 error")
        mock_response = type(
            "R",
            (),
            {
                "status_code": 400,
                "url": "https://earthengine.googleapis.com/v1/projects/x/image:getPixels",
                "text": "",
            },
        )()
        exc.response = mock_response
        assert _is_gee_download_too_large(exc) is True

    def test_request_payload_marker_returns_true(self):
        exc = Exception("Request payload size exceeds the limit")
        assert _is_gee_download_too_large(exc) is True


class TestNextDownloadScale:
    def test_first_attempt_uses_1_6_multiplier(self):
        result = _next_download_scale(100, attempt=0)
        assert result >= 160  # 100 * 1.6

    def test_subsequent_attempts_use_2_0_multiplier(self):
        result = _next_download_scale(100, attempt=1)
        assert result >= 200

    def test_always_at_least_scale_plus_one(self):
        result = _next_download_scale(1, attempt=0)
        assert result >= 2

    def test_returns_integer(self):
        result = _next_download_scale(50, attempt=0)
        assert isinstance(result, int)
