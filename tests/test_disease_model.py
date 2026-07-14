"""Tests for disease/model.py — train_gbm, train_xgb, evaluate_models, detect_hotspots."""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from xgboost import Booster, DMatrix

from climate_change.disease.model import (
    VALID_MODEL_TYPES,
    _safe_cv_folds,
    _safe_stratify,
    detect_hotspots,
    evaluate_models,
    train_gbm,
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
        rng = np.random.default_rng(6)
        X = rng.standard_normal((61, 7))
        y = np.array([0] * 30 + [1] * 30 + [2] * 1)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=_safe_stratify(y)
        )
        assert len(X_train) + len(X_test) == 61


@pytest.fixture()
def trained_disease_models(tiny_multiclass_xy):
    X, y = tiny_multiclass_xy
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    gbm, _ = train_gbm(X_tr, y_tr, cv_folds=2)
    xgb, _ = train_xgb(X_tr, y_tr, cv_folds=2)
    return gbm, xgb, X_te, y_te


class TestTrainGbm:
    def test_returns_model_and_metadata(self, tiny_multiclass_xy):
        X, y = tiny_multiclass_xy
        gbm, meta = train_gbm(X[:60], y[:60], cv_folds=2)
        assert isinstance(gbm, GradientBoostingClassifier)
        assert "cv_f1_mean" in meta
        assert "cv_f1_std" in meta

    def test_model_predicts_valid_classes(self, tiny_multiclass_xy):
        X, y = tiny_multiclass_xy
        gbm, _ = train_gbm(X[:60], y[:60], cv_folds=2)
        preds = gbm.predict(X[60:])
        assert set(preds).issubset({0, 1, 2})


class TestTrainXgb:
    def test_returns_model_and_metadata(self, tiny_multiclass_xy):
        X, y = tiny_multiclass_xy
        xgb, meta = train_xgb(X[:60], y[:60], cv_folds=2)
        assert isinstance(xgb, Booster)
        assert "cv_f1_mean" in meta

    def test_predictions_three_classes(self, tiny_multiclass_xy):
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
    def test_all_keys_present(self, trained_disease_models):
        gbm, xgb, X_te, y_te = trained_disease_models
        result = evaluate_models(gbm, xgb, X_te, y_te)
        for key in ("gbm", "xgb", "ensemble", "actuals"):
            assert key in result

    def test_f1_between_0_and_1(self, trained_disease_models):
        gbm, xgb, X_te, y_te = trained_disease_models
        result = evaluate_models(gbm, xgb, X_te, y_te)
        for key in ("gbm", "xgb", "ensemble"):
            assert 0.0 <= result[key]["f1"] <= 1.0

    def test_accuracy_between_0_and_1(self, trained_disease_models):
        gbm, xgb, X_te, y_te = trained_disease_models
        result = evaluate_models(gbm, xgb, X_te, y_te)
        for key in ("gbm", "xgb", "ensemble"):
            assert 0.0 <= result[key]["accuracy"] <= 1.0

    def test_predictions_length_matches_test_set(self, trained_disease_models):
        gbm, xgb, X_te, y_te = trained_disease_models
        result = evaluate_models(gbm, xgb, X_te, y_te)
        assert len(result["actuals"]) == len(y_te)

    def test_ensemble_handles_missing_class_in_training_data(self):
        """Regression test: when an AOI's data is missing one risk class, GBM's
        predict_proba only has columns for the classes it saw while train_xgb's
        Booster always outputs 3 columns (num_class is fixed). Combining them
        for the ensemble must not raise a broadcast shape mismatch."""
        rng = np.random.default_rng(3)
        X_train = rng.standard_normal((40, 7))
        y_train = np.repeat([1, 2], 20)  # class 0 ("Low Risk") entirely absent
        X_test = rng.standard_normal((10, 7))
        y_test = np.array([1, 2] * 5)

        gbm, _ = train_gbm(X_train, y_train, cv_folds=2)
        xgb, _ = train_xgb(X_train, y_train, cv_folds=2)

        result = evaluate_models(gbm, xgb, X_test, y_test)

        assert len(result["ensemble"]["predictions"]) == 10
        assert set(result["ensemble"]["predictions"]).issubset({0, 1, 2})


class TestDetectHotspots:
    def test_no_lon_lat_returns_empty(self):
        df = pd.DataFrame({"feature": [1, 2, 3]})
        labels = np.array([2, 2, 2])
        result = detect_hotspots(df, labels)
        assert result == []

    def test_few_high_risk_returns_empty(self):
        df = pd.DataFrame({"lon": [36.0, 36.1], "lat": [-1.0, -1.1]})
        labels = np.array([2, 2])  # only 2 < DBSCAN_MIN_SAMPLES=3
        result = detect_hotspots(df, labels)
        assert result == []

    def test_sufficient_cluster_returns_list(self):
        lons = [36.0, 36.01, 36.02, 36.03, 36.04]
        lats = [-1.0, -1.01, -1.02, -1.03, -1.04]
        df = pd.DataFrame({"lon": lons, "lat": lats})
        labels = np.array([2, 2, 2, 2, 2])
        result = detect_hotspots(df, labels, eps=0.09, min_samples=3)
        assert isinstance(result, list)

    def test_no_high_risk_pixels_returns_empty(self):
        df = pd.DataFrame({"lon": [36.0, 36.1, 36.2], "lat": [-1.0, -1.1, -1.2]})
        labels = np.array([0, 1, 1])  # no class-2 pixels
        result = detect_hotspots(df, labels)
        assert result == []

    def test_valid_model_types(self):
        for mt in ("gbm", "xgboost", "ensemble"):
            assert mt in VALID_MODEL_TYPES
