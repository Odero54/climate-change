"""Tests for core/shap_utils.py — compute_shap_importance.

Regression coverage for a real bug found while wiring SHAP to follow the
selected model type: shap 0.52.0's TreeExplainer returns
(n_samples, n_features, n_classes) for RandomForestClassifier — features in
the *middle* axis — while XGBoost/LightGBM/GradientBoostingClassifier all
return a plain (n_samples, n_features) array. A fixed-axis assumption
silently produced 2-element, class-indexed "feature" lists for RF instead
of real per-feature importances.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from climate_change.core.shap_utils import compute_shap_importance

FEATURE_COLS = [f"feat_{i}" for i in range(10)]


def _fit_rf(n=60, n_features=10, seed=1):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, n_features))
    y = np.array([0] * (n // 2) + [1] * (n // 2))
    rf = RandomForestClassifier(n_estimators=20, random_state=42).fit(X, y)
    return rf, X


def test_rf_returns_one_entry_per_feature_not_per_class():
    """RandomForest's shap_values is (n_samples, n_features, n_classes) in
    this shap version — must not be mistaken for (n_classes, ...) and
    collapsed down to a 2-element (per-class) result."""
    rf, X = _fit_rf()
    result = compute_shap_importance(rf, X, FEATURE_COLS)
    assert len(result["features"]) == len(FEATURE_COLS)
    assert len(result["mean_abs_shap"]) == len(FEATURE_COLS)
    assert sorted(result["features"]) == sorted(FEATURE_COLS)


def test_features_sorted_by_descending_importance():
    rf, X = _fit_rf()
    result = compute_shap_importance(rf, X, FEATURE_COLS)
    assert result["mean_abs_shap"] == sorted(result["mean_abs_shap"], reverse=True)


def test_xgboost_2d_shap_values_also_work():
    """XGBoost's Booster returns a plain (n_samples, n_features) array in
    this shap version — must still work correctly, not just RF's 3D case."""
    from xgboost import DMatrix
    from xgboost import train as xgb_train

    rng = np.random.default_rng(2)
    X = rng.standard_normal((60, 6))
    y = np.array([0] * 30 + [1] * 30)
    booster = xgb_train(
        {"objective": "binary:logistic", "verbosity": 0},
        DMatrix(X, label=y),
        num_boost_round=20,
    )
    cols = [f"feat_{i}" for i in range(6)]
    result = compute_shap_importance(booster, X, cols)
    assert sorted(result["features"]) == sorted(cols)
