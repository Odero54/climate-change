"""Tests for core/shap_utils.py — compute_shap_importance.

Regression coverage for a real bug found while wiring SHAP to follow the
selected model type: shap 0.52.0's TreeExplainer returns
(n_samples, n_features, n_classes) for RandomForestClassifier — features in
the *middle* axis — while XGBoost/LightGBM/GradientBoostingClassifier all
return a plain (n_samples, n_features) array. A fixed-axis assumption
silently produced 2-element, class-indexed "feature" lists for RF instead
of real per-feature importances.
"""

from unittest.mock import patch

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


def test_multiclass_gradient_boosting_falls_back_to_model_agnostic_explainer():
    """shap.TreeExplainer explicitly rejects sklearn's GradientBoostingClassifier
    for 3+-class problems ('only supported for binary classification right
    now') — this only surfaced once SHAP started following the actually-
    selected model (disease's "gbm" option) instead of always using
    XGBoost. Must fall back to a model-agnostic explainer rather than crash."""
    from sklearn.ensemble import GradientBoostingClassifier

    rng = np.random.default_rng(3)
    n = 40
    X = rng.standard_normal((n, 5))
    y = np.array([0] * 14 + [1] * 13 + [2] * 13)
    gbm = GradientBoostingClassifier(n_estimators=20, random_state=42).fit(X, y)

    cols = [f"feat_{i}" for i in range(5)]
    result = compute_shap_importance(gbm, X[:15], cols)
    assert sorted(result["features"]) == sorted(cols)
    assert len(result["mean_abs_shap"]) == 5


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


def test_xgboost_booster_falls_back_when_tree_explainer_cant_parse_the_model():
    """On Python 3.10/3.11, the newest resolvable shap for that Python
    version can't parse the newest resolvable XGBoost's base_score
    serialization ('could not convert string to float' on a JSON-array-
    as-string base_score) — confirmed live in CI. Booster has no
    predict_proba (unlike RF/LightGBM/GBM), so the fallback must wrap
    Booster.predict(DMatrix(...)) itself rather than relying on
    predict_proba. Simulates the failure directly since reproducing the
    exact broken shap+xgboost version pair isn't possible from a single
    environment."""
    from xgboost import DMatrix
    from xgboost import train as xgb_train

    rng = np.random.default_rng(2)
    X = rng.standard_normal((60, 6))
    y = np.array([0] * 20 + [1] * 20 + [2] * 20)
    booster = xgb_train(
        {"objective": "multi:softprob", "num_class": 3, "verbosity": 0},
        DMatrix(X, label=y),
        num_boost_round=20,
    )
    cols = [f"feat_{i}" for i in range(6)]

    class _FakeTreeExplainer:
        def __init__(self, model):
            raise ValueError("could not convert string to float: '[0E0,0E0,0E0]'")

    with patch("shap.TreeExplainer", _FakeTreeExplainer):
        result = compute_shap_importance(booster, X[:15], cols)

    assert sorted(result["features"]) == sorted(cols)
    assert len(result["mean_abs_shap"]) == 6
