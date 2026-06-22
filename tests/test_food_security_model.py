"""Tests for food_security/model.py — train_rf, train_xgb, evaluate_models, build_food_security_charts."""
import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from food_security.features import FEATURE_COLS, FOOD_CLASSES
from food_security.model import (
    VALID_MODEL_TYPES,
    build_food_security_charts,
    evaluate_models,
    train_rf,
    train_xgb,
)


@pytest.fixture()
def trained_food_models(tiny_multiclass_xy):
    X, y = tiny_multiclass_xy
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    rf, _ = train_rf(X_tr, y_tr, cv_folds=2)
    xgb, _ = train_xgb(X_tr, y_tr, cv_folds=2)
    return rf, xgb, X_te, y_te


class TestTrainRf:
    def test_returns_model_and_metadata(self, tiny_multiclass_xy):
        X, y = tiny_multiclass_xy
        rf, meta = train_rf(X[:60], y[:60], cv_folds=2)
        assert "cv_f1_mean" in meta
        assert "cv_f1_std" in meta

    def test_model_can_predict(self, tiny_multiclass_xy):
        X, y = tiny_multiclass_xy
        rf, _ = train_rf(X[:60], y[:60], cv_folds=2)
        preds = rf.predict(X[60:])
        assert len(preds) == len(X[60:])


class TestTrainXgb:
    def test_returns_model_and_metadata(self, tiny_multiclass_xy):
        X, y = tiny_multiclass_xy
        xgb, meta = train_xgb(X[:60], y[:60], cv_folds=2)
        assert "cv_f1_mean" in meta

    def test_model_predicts_valid_classes(self, tiny_multiclass_xy):
        X, y = tiny_multiclass_xy
        xgb, _ = train_xgb(X[:60], y[:60], cv_folds=2)
        preds = xgb.predict(X[60:])
        assert set(preds).issubset({0, 1, 2})


class TestEvaluateModels:
    def test_all_keys_present(self, trained_food_models):
        rf, xgb, X_te, y_te = trained_food_models
        result = evaluate_models(rf, xgb, X_te, y_te)
        assert "rf" in result
        assert "xgb" in result
        assert "ensemble" in result
        assert "actuals" in result

    def test_f1_between_0_and_1(self, trained_food_models):
        rf, xgb, X_te, y_te = trained_food_models
        result = evaluate_models(rf, xgb, X_te, y_te)
        for key in ("rf", "xgb", "ensemble"):
            assert 0.0 <= result[key]["f1"] <= 1.0

    def test_accuracy_between_0_and_1(self, trained_food_models):
        rf, xgb, X_te, y_te = trained_food_models
        result = evaluate_models(rf, xgb, X_te, y_te)
        for key in ("rf", "xgb", "ensemble"):
            assert 0.0 <= result[key]["accuracy"] <= 1.0


class TestBuildFoodSecurityCharts:
    def _make_eval_result(self, n=30):
        preds = (np.arange(n) % 3).tolist()
        entry = {"label": "test", "f1": 0.7, "accuracy": 0.7, "predictions": preds}
        return {
            "rf": entry, "xgb": entry, "ensemble": entry,
            "actuals": preds,
        }

    def test_keys_present(self):
        result = build_food_security_charts(
            eval_result=self._make_eval_result(),
            shap_payload={"features": FEATURE_COLS, "mean_abs_shap": [0.1] * 7},
            ndvi_df=pd.DataFrame({"ndvi": [0.4, 0.5]}, index=[0, 1]),
            rain_df=pd.DataFrame({"rain_mm": [80, 90]}, index=[0, 1]),
            vci_mean=55.0, tci_mean=60.0, vhi_mean=57.5,
        )
        for key in ("riskDist", "timeSeries", "shap", "indices"):
            assert key in result

    def test_risk_dist_has_three_classes(self):
        result = build_food_security_charts(
            eval_result=self._make_eval_result(),
            shap_payload={},
            ndvi_df=pd.DataFrame(),
            rain_df=pd.DataFrame(),
            vci_mean=50.0, tci_mean=50.0, vhi_mean=50.0,
        )
        assert result["riskDist"]["labels"] == FOOD_CLASSES

    def test_valid_model_types(self):
        for mt in ("rf", "xgboost", "ensemble"):
            assert mt in VALID_MODEL_TYPES
