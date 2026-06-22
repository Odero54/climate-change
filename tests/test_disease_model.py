"""Tests for disease/model.py — train_gbm, train_xgb, evaluate_models, detect_hotspots."""
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from disease.features import FEATURE_COLS
from disease.model import (
    VALID_MODEL_TYPES,
    detect_hotspots,
    evaluate_models,
    train_gbm,
    train_xgb,
)


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
        assert isinstance(xgb, XGBClassifier)
        assert "cv_f1_mean" in meta

    def test_predictions_three_classes(self, tiny_multiclass_xy):
        X, y = tiny_multiclass_xy
        xgb, _ = train_xgb(X[:60], y[:60], cv_folds=2)
        preds = xgb.predict(X[60:])
        assert set(preds).issubset({0, 1, 2})


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
