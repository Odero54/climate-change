"""Tests for food_security/model.py — train_rf, train_xgb, evaluate_models, build_food_security_charts."""

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split
from xgboost import Booster, DMatrix

from climate_change.food_security.features import FEATURE_COLS, FOOD_CLASSES
from climate_change.food_security.model import (
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
        assert isinstance(xgb, Booster)
        assert "cv_f1_mean" in meta

    def test_model_predicts_valid_classes(self, tiny_multiclass_xy):
        X, y = tiny_multiclass_xy
        xgb, _ = train_xgb(X[:60], y[:60], cv_folds=2)
        preds = np.argmax(xgb.predict(DMatrix(X[60:])), axis=1)
        assert set(preds).issubset({0, 1, 2})

    def test_handles_missing_class_in_training_data(self):
        """Regression test: an AOI/time window with zero samples of one risk
        class must train successfully instead of raising the XGBClassifier
        'Invalid classes inferred from unique values of `y`' error."""
        rng = np.random.default_rng(1)
        X = rng.standard_normal((40, 7))
        y = np.repeat([1, 2], 20)  # class 0 ("Low Risk") entirely absent

        xgb, meta = train_xgb(X, y, cv_folds=2)

        assert isinstance(xgb, Booster)
        proba = xgb.predict(DMatrix(X))
        assert proba.shape == (40, 3)
        assert set(np.argmax(proba, axis=1)).issubset({0, 1, 2})


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

    def test_ensemble_handles_missing_class_in_training_data(self):
        """Regression test: when an AOI's data is missing one risk class, RF's
        predict_proba only has columns for the classes it saw while train_xgb's
        Booster always outputs 3 columns (num_class is fixed). Combining them
        for the ensemble must not raise a broadcast shape mismatch."""
        rng = np.random.default_rng(3)
        X_train = rng.standard_normal((40, 7))
        y_train = np.repeat([1, 2], 20)  # class 0 ("Low Risk") entirely absent
        X_test = rng.standard_normal((10, 7))
        y_test = np.array([1, 2] * 5)

        rf, _ = train_rf(X_train, y_train, cv_folds=2)
        xgb, _ = train_xgb(X_train, y_train, cv_folds=2)

        result = evaluate_models(rf, xgb, X_test, y_test)

        assert len(result["ensemble"]["predictions"]) == 10
        assert set(result["ensemble"]["predictions"]).issubset({0, 1, 2})


class TestBuildFoodSecurityCharts:
    def _make_eval_result(self, n=30):
        preds = (np.arange(n) % 3).tolist()
        entry = {"label": "test", "f1": 0.7, "accuracy": 0.7, "predictions": preds}
        return {
            "rf": entry,
            "xgb": entry,
            "ensemble": entry,
            "actuals": preds,
        }

    def test_keys_present(self):
        result = build_food_security_charts(
            eval_result=self._make_eval_result(),
            shap_payload={"features": FEATURE_COLS, "mean_abs_shap": [0.1] * 7},
            ndvi_df=pd.DataFrame({"ndvi": [0.4, 0.5]}, index=[0, 1]),
            rain_df=pd.DataFrame({"rain_mm": [80, 90]}, index=[0, 1]),
            vci_mean=55.0,
            tci_mean=60.0,
            vhi_mean=57.5,
        )
        for key in ("riskDist", "timeSeries", "shap", "indices"):
            assert key in result

    def test_risk_dist_has_three_classes(self):
        result = build_food_security_charts(
            eval_result=self._make_eval_result(),
            shap_payload={},
            ndvi_df=pd.DataFrame(),
            rain_df=pd.DataFrame(),
            vci_mean=50.0,
            tci_mean=50.0,
            vhi_mean=50.0,
        )
        assert result["riskDist"]["labels"] == FOOD_CLASSES

    def test_valid_model_types(self):
        for mt in ("rf", "xgboost", "ensemble"):
            assert mt in VALID_MODEL_TYPES
