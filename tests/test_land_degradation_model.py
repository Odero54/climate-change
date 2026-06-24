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
    build_degradation_charts,
    compute_ndvi_trend,
    evaluate_models,
    train_lgbm,
    train_rf,
)


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
