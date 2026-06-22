"""Tests for flood/model.py — classify_flood_risk, train_rf, train_xgb, evaluate_models,
find_best_threshold, compute_uncertainty, build_flood_charts."""

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from xgboost.sklearn import XGBClassifier

from climate_change.flood.features import FEATURE_COLS
from climate_change.flood.model import (
    VALID_MODEL_TYPES,
    build_flood_charts,
    classify_flood_risk,
    compute_uncertainty,
    evaluate_models,
    find_best_threshold,
    train_rf,
    train_xgb,
)


@pytest.fixture()
def trained_models(tiny_binary_xy):
    X, y = tiny_binary_xy
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    rf, _ = train_rf(X_tr, y_tr, cv_folds=2)
    xgb, _ = train_xgb(X_tr, y_tr, X_te, y_te, cv_folds=2)
    return rf, xgb, X_te, y_te


# ── classify_flood_risk ───────────────────────────────────────────────────────


class TestClassifyFloodRisk:
    def test_low_below_25(self):
        prob = np.array([0.1, 0.2])
        result = classify_flood_risk(prob)
        assert list(result) == ["Low", "Low"]

    def test_medium_25_to_50(self):
        prob = np.array([0.25, 0.49])
        result = classify_flood_risk(prob)
        assert list(result) == ["Medium", "Medium"]

    def test_high_50_to_75(self):
        prob = np.array([0.50, 0.74])
        result = classify_flood_risk(prob)
        assert list(result) == ["High", "High"]

    def test_very_high_above_75(self):
        prob = np.array([0.75, 0.99])
        result = classify_flood_risk(prob)
        assert list(result) == ["Very High", "Very High"]

    def test_mixed(self):
        prob = np.array([0.1, 0.3, 0.6, 0.9])
        result = classify_flood_risk(prob)
        assert list(result) == ["Low", "Medium", "High", "Very High"]

    def test_output_dtype_object(self):
        result = classify_flood_risk(np.array([0.5]))
        assert result.dtype == object


# ── train_rf ──────────────────────────────────────────────────────────────────


class TestTrainRf:
    def test_returns_rf_and_metadata(self, tiny_binary_xy):
        X, y = tiny_binary_xy
        rf, meta = train_rf(X[:40], y[:40], cv_folds=2)
        assert isinstance(rf, RandomForestClassifier)
        assert "cv_f1_mean" in meta
        assert "cv_f1_std" in meta

    def test_model_fitted(self, tiny_binary_xy):
        X, y = tiny_binary_xy
        rf, _ = train_rf(X[:40], y[:40], cv_folds=2)
        preds = rf.predict(X[40:])
        assert len(preds) == len(X[40:])


# ── train_xgb ─────────────────────────────────────────────────────────────────


class TestTrainXgb:
    def test_returns_xgb_and_metadata(self, tiny_binary_xy):
        X, y = tiny_binary_xy
        xgb, meta = train_xgb(X[:40], y[:40], X[40:], y[40:], cv_folds=2)
        assert isinstance(xgb, XGBClassifier)
        assert "cv_f1_mean" in meta


# ── find_best_threshold ───────────────────────────────────────────────────────


class TestFindBestThreshold:
    def test_returns_threshold_and_f1(self, tiny_binary_xy):
        X, y = tiny_binary_xy
        rf, _ = train_rf(X[:40], y[:40], cv_folds=2)
        probs = rf.predict_proba(X[40:])[:, 1]
        thresh, f1 = find_best_threshold(probs, y[40:])
        assert 0.0 <= thresh <= 1.0
        assert 0.0 <= f1 <= 1.0


# ── evaluate_models ───────────────────────────────────────────────────────────


class TestEvaluateModels:
    def test_all_keys_present(self, trained_models):
        rf, xgb, X_te, y_te = trained_models
        result = evaluate_models(rf, xgb, X_te, y_te)
        assert "rf" in result
        assert "xgb" in result
        assert "ensemble" in result
        assert "actuals" in result

    def test_f1_scores_between_0_and_1(self, trained_models):
        rf, xgb, X_te, y_te = trained_models
        result = evaluate_models(rf, xgb, X_te, y_te)
        for key in ("rf", "xgb", "ensemble"):
            assert 0.0 <= result[key]["f1"] <= 1.0


# ── compute_uncertainty ───────────────────────────────────────────────────────


class TestComputeUncertainty:
    def test_keys_present(self):
        rf_prob = np.array([0.2, 0.8, 0.5])
        xgb_prob = np.array([0.3, 0.7, 0.6])
        result = compute_uncertainty(rf_prob, xgb_prob)
        assert "mean_spread" in result
        assert "high_uncertainty_pct" in result
        assert "spread_stats" in result

    def test_identical_probs_zero_spread(self):
        prob = np.array([0.3, 0.6, 0.9])
        result = compute_uncertainty(prob, prob)
        assert result["mean_spread"] == 0.0

    def test_high_uncertainty_pct_between_0_and_100(self):
        rf_prob = np.zeros(10)
        xgb_prob = np.ones(10)
        result = compute_uncertainty(rf_prob, xgb_prob)
        assert 0.0 <= result["high_uncertainty_pct"] <= 100.0


# ── build_flood_charts ────────────────────────────────────────────────────────


class TestBuildFloodCharts:
    def _make_eval_result(self):
        n = 20
        probs = np.linspace(0, 1, n)
        actuals = (np.arange(n) % 2).tolist()
        pred = (probs > 0.5).astype(int).tolist()
        model_entry = {
            "label": "test",
            "f1": 0.8,
            "auc": 0.85,
            "threshold": 0.5,
            "predictions": pred,
            "probabilities": probs.tolist(),
        }
        return {
            "rf": model_entry,
            "xgb": model_entry,
            "ensemble": model_entry,
            "actuals": actuals,
        }

    def test_keys_present(self):
        eval_result = self._make_eval_result()
        shap_payload = {"features": FEATURE_COLS, "mean_abs_shap": [0.1] * len(FEATURE_COLS)}
        uncertainty = {"mean_spread": 0.05, "high_uncertainty_pct": 10.0, "spread_stats": {}}
        result = build_flood_charts(eval_result, shap_payload, uncertainty)
        assert "risk_distribution" in result
        assert "shap" in result
        assert "uncertainty" in result
        assert "model_performance" in result

    def test_valid_model_types(self):
        assert "rf" in VALID_MODEL_TYPES
        assert "xgboost" in VALID_MODEL_TYPES
        assert "ensemble" in VALID_MODEL_TYPES
