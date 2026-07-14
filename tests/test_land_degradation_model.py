"""Tests for land_degradation/model.py — train_rf, train_lgbm, evaluate_models,
compute_ndvi_trend, build_degradation_charts."""

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from climate_change.land_degradation.features import DEGRADATION_CLASSES, FEATURE_COLS
from climate_change.land_degradation.model import (
    VALID_MODEL_TYPES,
    _safe_cv_folds,
    _safe_stratify,
    build_degradation_charts,
    compute_ndvi_trend,
    evaluate_models,
    train_lgbm,
    train_rf,
)

# ── _safe_cv_folds ────────────────────────────────────────────────────────────


class TestSafeCvFolds:
    def test_balanced_classes_clamps_to_cv_folds(self):
        y = np.array([0] * 50 + [1] * 50)
        assert _safe_cv_folds(y, cv_folds=5) == 5

    def test_scarce_class_clamps_down_to_its_count(self):
        y = np.array([0] * 27 + [1] * 3)
        assert _safe_cv_folds(y, cv_folds=5) == 3

    def test_single_member_class_returns_none(self):
        y = np.array([0] * 20 + [1] * 1)
        assert _safe_cv_folds(y, cv_folds=5) is None

    def test_single_well_populated_class_is_not_blocked(self):
        """Every sampled pixel being the same class (e.g. none degraded) is a
        legitimate data state that sklearn's StratifiedKFold handles fine as
        long as that one class has >= cv_folds members — must not be treated
        as 'scarce' just because only one class is present."""
        y = np.array([1] * 80)
        assert _safe_cv_folds(y, cv_folds=5) == 5


# ── _safe_stratify ────────────────────────────────────────────────────────────


class TestSafeStratify:
    def test_balanced_classes_returns_y_unchanged(self):
        y = np.array([0] * 50 + [1] * 50)
        result = _safe_stratify(y)
        assert result is not None
        np.testing.assert_array_equal(result, y)

    def test_single_member_class_returns_none(self):
        """Regression test: train_test_split(stratify=y) raises 'The least
        populated class in y has only 1 member' when a class has exactly 1
        sampled pixel (e.g. an AOI with a single degraded pixel) — must fall
        back to a non-stratified split instead of crashing."""
        y = np.array([0] * 49 + [1] * 1)
        assert _safe_stratify(y) is None

    def test_actually_prevents_train_test_split_crash(self):
        rng = np.random.default_rng(7)
        X = rng.standard_normal((50, 8))
        y = np.array([0] * 49 + [1] * 1)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=_safe_stratify(y)
        )
        assert len(X_train) + len(X_test) == 50


@pytest.fixture()
def trained_land_models(tiny_binary_xy):
    X, y = tiny_binary_xy
    X_tr, X_te, y_tr, y_te = train_test_split(
        X[:, :8], y, test_size=0.3, random_state=42, stratify=y
    )
    rf, _ = train_rf(X_tr, y_tr, cv_folds=2)
    lgbm, _ = train_lgbm(X_tr, y_tr, cv_folds=2)
    return rf, lgbm, X_te, y_te


class TestTrainRf:
    def test_returns_rf_and_metadata(self, tiny_binary_xy):
        X, y = tiny_binary_xy
        rf, meta = train_rf(X[:40, :8], y[:40], cv_folds=2)
        assert isinstance(rf, RandomForestClassifier)
        assert "cv_f1_mean" in meta
        assert "cv_f1_std" in meta

    def test_model_predicts_binary(self, tiny_binary_xy):
        X, y = tiny_binary_xy
        rf, _ = train_rf(X[:40, :8], y[:40], cv_folds=2)
        preds = rf.predict(X[40:, :8])
        assert set(preds).issubset({0, 1})


class TestTrainLgbm:
    def test_returns_lgbm_and_metadata(self, tiny_binary_xy):
        X, y = tiny_binary_xy
        lgbm_model, meta = train_lgbm(X[:40, :8], y[:40], cv_folds=2)
        assert isinstance(lgbm_model, lgb.LGBMClassifier)
        assert "cv_f1_mean" in meta

    def test_model_predicts_binary(self, tiny_binary_xy):
        X, y = tiny_binary_xy
        lgbm_model, _ = train_lgbm(X[:40, :8], y[:40], cv_folds=2)
        preds = lgbm_model.predict(X[40:, :8])
        assert set(np.asarray(preds)).issubset({0, 1})


class TestEvaluateModels:
    def test_all_keys_present(self, trained_land_models):
        rf, lgbm, X_te, y_te = trained_land_models
        result = evaluate_models(rf, lgbm, X_te, y_te)
        for key in ("rf", "lgbm", "ensemble", "actuals"):
            assert key in result

    def test_f1_between_0_and_1(self, trained_land_models):
        rf, lgbm, X_te, y_te = trained_land_models
        result = evaluate_models(rf, lgbm, X_te, y_te)
        for key in ("rf", "lgbm", "ensemble"):
            assert 0.0 <= result[key]["f1"] <= 1.0

    def test_accuracy_between_0_and_1(self, trained_land_models):
        rf, lgbm, X_te, y_te = trained_land_models
        result = evaluate_models(rf, lgbm, X_te, y_te)
        for key in ("rf", "lgbm", "ensemble"):
            assert 0.0 <= result[key]["accuracy"] <= 1.0


class TestComputeNdviTrend:
    def test_returns_expected_keys(self):
        years = list(range(2010, 2025))
        values = [0.4 + i * 0.01 for i in range(len(years))]  # upward trend
        ndvi = pd.Series(values, index=years)
        result = compute_ndvi_trend(ndvi)
        for key in ("ndvi_trend_per_year", "ndvi_trend_r2", "mk_significant", "breakpoint_years"):
            assert key in result

    def test_upward_trend_positive_slope(self):
        years = list(range(2000, 2020))
        values = [0.3 + i * 0.02 for i in range(len(years))]
        ndvi = pd.Series(values, index=years)
        result = compute_ndvi_trend(ndvi)
        assert result["ndvi_trend_per_year"] > 0

    def test_downward_trend_negative_slope(self):
        years = list(range(2000, 2020))
        values = [0.8 - i * 0.02 for i in range(len(years))]
        ndvi = pd.Series(values, index=years)
        result = compute_ndvi_trend(ndvi)
        assert result["ndvi_trend_per_year"] < 0

    def test_r2_between_0_and_1(self):
        years = list(range(2000, 2015))
        values = [0.5 + i * 0.01 for i in range(len(years))]
        ndvi = pd.Series(values, index=years)
        result = compute_ndvi_trend(ndvi)
        assert 0.0 <= result["ndvi_trend_r2"] <= 1.0

    def test_breakpoint_years_is_list(self):
        years = list(range(2000, 2020))
        values = [0.5] * 10 + [0.3] * 10  # structural break at 2010
        ndvi = pd.Series(values, index=years)
        result = compute_ndvi_trend(ndvi)
        assert isinstance(result["breakpoint_years"], list)

    def test_nan_values_dropped(self):
        years = list(range(2000, 2015))
        values = [0.5 if i % 3 != 0 else np.nan for i in range(len(years))]
        ndvi = pd.Series(values, index=years)
        result = compute_ndvi_trend(ndvi)
        assert "ndvi_trend_per_year" in result

    def test_empty_series_returns_none_not_nan(self):
        """Regression test: a date range so short (or so cloud-affected) that
        zero annual NDVI points survive dropna() must return None for the
        undefined trend stats, not NaN — NaN isn't valid JSON and would
        corrupt the analysis response."""
        ndvi = pd.Series([], dtype=float, index=pd.Index([], dtype=int))
        result = compute_ndvi_trend(ndvi)
        assert result["ndvi_trend_per_year"] is None
        assert result["ndvi_trend_r2"] is None
        assert result["mk_tau"] is None
        assert result["mk_significant"] is False
        assert result["breakpoint_years"] == []
        assert result["breakpoint_year"] is None

    def test_single_point_returns_none_not_nan(self):
        ndvi = pd.Series([0.45], index=[2022])
        result = compute_ndvi_trend(ndvi)
        assert result["ndvi_trend_per_year"] is None
        assert result["breakpoint_years"] == []

    def test_few_points_does_not_crash_breakpoint_detection(self):
        """Regression test: with 2-3 annual points, linregress/kendalltau
        compute fine, but the old code unconditionally fell back to
        n_bkps=1 even when ruptures.utils.sanity_check said no breakpoint
        count was feasible — Binseg.predict() then raised
        BadSegmentationParameters. Must degrade to an empty breakpoint list
        instead of crashing the whole analysis."""
        for n in (2, 3):
            years = list(range(2020, 2020 + n))
            values = [0.4 + i * 0.01 for i in range(n)]
            ndvi = pd.Series(values, index=years)
            result = compute_ndvi_trend(ndvi)
            assert result["ndvi_trend_per_year"] is not None  # linregress still works
            assert result["breakpoint_years"] == []  # but too few points to segment

    def test_four_points_breakpoint_detection_still_works(self):
        """4 points is the minimum where Binseg can actually place a
        breakpoint — confirms the guard didn't disable the feature entirely."""
        years = list(range(2020, 2024))
        values = [0.3, 0.3, 0.6, 0.6]
        ndvi = pd.Series(values, index=years)
        result = compute_ndvi_trend(ndvi)
        assert isinstance(result["breakpoint_years"], list)


class TestBuildDegradationCharts:
    def _make_eval_result(self, n=30):
        preds = (np.arange(n) % 2).tolist()
        entry = {"label": "test", "f1": 0.75, "accuracy": 0.75, "predictions": preds}
        return {"rf": entry, "lgbm": entry, "ensemble": entry, "actuals": preds}

    def test_keys_present(self):
        ndvi = pd.Series([0.4, 0.5, 0.45], index=[2020, 2021, 2022])
        trend = {"ndvi_trend_per_year": 0.01, "mk_significant": False}
        result = build_degradation_charts(
            eval_result=self._make_eval_result(),
            shap_payload={"features": FEATURE_COLS, "mean_abs_shap": [0.1] * 8},
            ndvi_annual=ndvi,
            trend_stats=trend,
        )
        for key in ("riskDist", "timeSeries", "shap", "trend", "model_performance"):
            assert key in result

    def test_risk_dist_has_degradation_classes(self):
        ndvi = pd.Series([0.4, 0.5], index=[2020, 2021])
        trend = {}
        result = build_degradation_charts(
            eval_result=self._make_eval_result(),
            shap_payload={},
            ndvi_annual=ndvi,
            trend_stats=trend,
        )
        assert result["riskDist"]["labels"] == DEGRADATION_CLASSES

    def test_valid_model_types(self):
        for mt in ("rf", "lgbm", "ensemble"):
            assert mt in VALID_MODEL_TYPES
