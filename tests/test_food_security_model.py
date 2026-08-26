"""Tests for food_security/model.py — train_rf, train_xgb, evaluate_models, build_food_security_charts."""

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split
from xgboost import Booster, DMatrix

from climate_change.food_security.features import FEATURE_COLS, FOOD_CLASSES
from climate_change.food_security.model import (
    VALID_MODEL_TYPES,
    FoodSecurityModel,
    _safe_cv_folds,
    _safe_stratify,
    build_food_security_charts,
    evaluate_models,
    train_rf,
    train_xgb,
)

# ── _safe_cv_folds ────────────────────────────────────────────────────────────


class TestSafeCvFolds:
    def test_balanced_classes_clamps_to_cv_folds(self):
        y = np.array([0] * 30 + [1] * 30 + [2] * 30)
        assert _safe_cv_folds(y, cv_folds=5) == 5

    def test_scarce_class_clamps_down_to_its_count(self):
        y = np.array([0] * 30 + [1] * 30 + [2] * 3)
        assert _safe_cv_folds(y, cv_folds=5) == 3

    def test_single_member_class_returns_none(self):
        y = np.array([0] * 20 + [1] * 20 + [2] * 1)
        assert _safe_cv_folds(y, cv_folds=5) is None

    def test_entirely_absent_class_is_not_mistaken_for_scarce(self):
        y = np.array([1] * 40 + [2] * 40)  # class 0 ("Low Risk") never appears
        assert _safe_cv_folds(y, cv_folds=5) == 5


# ── _safe_stratify ────────────────────────────────────────────────────────────


class TestSafeStratify:
    def test_balanced_classes_returns_y_unchanged(self):
        y = np.array([0] * 30 + [1] * 30 + [2] * 30)
        result = _safe_stratify(y)
        assert result is not None
        np.testing.assert_array_equal(result, y)

    def test_single_member_class_returns_none(self):
        """Regression test: train_test_split(stratify=y) raises 'The least
        populated class in y has only 1 member' when a risk class has exactly
        1 sampled pixel — must fall back to a non-stratified split instead of
        crashing."""
        y = np.array([0] * 30 + [1] * 30 + [2] * 1)
        assert _safe_stratify(y) is None

    def test_actually_prevents_train_test_split_crash(self):
        rng = np.random.default_rng(8)
        X = rng.standard_normal((61, 7))
        y = np.array([0] * 30 + [1] * 30 + [2] * 1)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=_safe_stratify(y)
        )
        assert len(X_train) + len(X_test) == 61


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


# ── evaluate_models: single-model selection ────────────────────────────────────


class TestEvaluateModelsSingleModel:
    def test_rf_none_omits_rf_and_ensemble_keys(self, trained_food_models):
        _, xgb, X_te, y_te = trained_food_models
        result = evaluate_models(None, xgb, X_te, y_te)
        assert "rf" not in result
        assert "ensemble" not in result
        assert "xgb" in result

    def test_xgb_none_omits_xgb_and_ensemble_keys(self, trained_food_models):
        rf, _, X_te, y_te = trained_food_models
        result = evaluate_models(rf, None, X_te, y_te)
        assert "xgb" not in result
        assert "ensemble" not in result
        assert "rf" in result


# ── FoodSecurityModel.predict: single-model selection trains only that model ───


@pytest.fixture()
def food_security_training_df():
    rng = np.random.default_rng(5)
    n = 60
    df = pd.DataFrame(
        {
            "vci": rng.uniform(0, 100, n),
            "tci": rng.uniform(0, 100, n),
            "rainfall_anom_pct": rng.standard_normal(n),
            "ndvi_slope": rng.standard_normal(n),
            "mndwi": rng.standard_normal(n),
            "slope_terrain": rng.uniform(0, 30, n),
            "land_cover": rng.standard_normal(n),
            "lon": rng.uniform(30, 31, n),
            "lat": rng.uniform(-1, 0, n),
        }
    )
    label = np.zeros(n, dtype=int)
    label[20:40] = 1
    label[40:] = 2
    df["label"] = label
    df["food_score"] = rng.standard_normal(n)
    return df


class TestFoodSecurityModelPredictSingleModel:
    def test_rf_only_never_trains_xgb(self, food_security_training_df):
        model = FoodSecurityModel()
        result = model.predict(food_security_training_df, config={"model_type": "rf"})
        assert model.rf is not None
        assert model.xgb is None
        assert result["stats"]["xgb_f1"] is None
        assert result["stats"]["xgb_accuracy"] is None
        assert result["stats"]["ensemble_f1"] is None

    def test_xgboost_only_never_trains_rf(self, food_security_training_df):
        model = FoodSecurityModel()
        result = model.predict(food_security_training_df, config={"model_type": "xgboost"})
        assert model.xgb is not None
        assert model.rf is None
        assert result["stats"]["rf_f1"] is None
        assert result["stats"]["rf_accuracy"] is None
        assert result["stats"]["ensemble_f1"] is None

    def test_ensemble_trains_both(self, food_security_training_df):
        model = FoodSecurityModel()
        result = model.predict(food_security_training_df, config={"model_type": "ensemble"})
        assert model.rf is not None
        assert model.xgb is not None
        assert result["stats"]["rf_f1"] is not None
        assert result["stats"]["xgb_f1"] is not None
        assert result["stats"]["ensemble_f1"] is not None
