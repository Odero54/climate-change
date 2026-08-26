"""Tests for flood/model.py — classify_flood_risk, train_rf, train_xgb, evaluate_models,
find_best_threshold, compute_uncertainty, build_flood_charts."""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from xgboost import Booster, DMatrix

from climate_change.flood.features import FEATURE_COLS
from climate_change.flood.model import (
    VALID_MODEL_TYPES,
    FloodModel,
    _safe_cv_folds,
    _safe_stratify,
    build_flood_charts,
    classify_flood_risk,
    compute_shap_importance,
    compute_uncertainty,
    evaluate_models,
    find_best_threshold,
    positive_class_proba,
    train_rf,
    train_xgb,
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
        """Every sampled pixel being the same class (e.g. an AOI/window that's
        entirely flooded or entirely dry) is a legitimate data state that
        sklearn's StratifiedKFold handles fine as long as that one class has
        >= cv_folds members — must not be treated as 'scarce' just because
        only one class is present."""
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
        sample (e.g. a small AOI/window with a single flooded pixel) — must
        fall back to a non-stratified split instead of crashing."""
        y = np.array([0] * 49 + [1] * 1)
        assert _safe_stratify(y) is None

    def test_actually_prevents_train_test_split_crash(self):
        rng = np.random.default_rng(5)
        X = rng.standard_normal((50, 10))
        y = np.array([0] * 49 + [1] * 1)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=_safe_stratify(y)
        )
        assert len(X_train) + len(X_test) == 50


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
        assert isinstance(xgb, Booster)
        assert "cv_f1_mean" in meta

    def test_handles_single_class_training_data(self):
        """Regression test: an AOI/time window where every sampled pixel is
        flooded (or none are) must train successfully instead of raising the
        XGBClassifier 'Invalid classes inferred from unique values of `y`'
        error, which fires whenever y's lone class isn't 0."""
        rng = np.random.default_rng(2)
        X_train = rng.standard_normal((30, 10))
        y_train = np.ones(30, dtype=int)  # every sampled pixel flooded
        X_val = rng.standard_normal((10, 10))
        y_val = np.ones(10, dtype=int)

        xgb, meta = train_xgb(X_train, y_train, X_val, y_val, cv_folds=2)

        assert isinstance(xgb, Booster)
        prob = xgb.predict(DMatrix(X_train))
        assert prob.shape == (30,)
        assert "cv_f1_mean" in meta

    def test_scarce_minority_class_skips_cv_instead_of_crashing(self):
        """Regression test: a positive class with fewer members than cv_folds
        (e.g. a small/urban AOI with very few flooded pixels sampled) used to
        raise sklearn's 'n_splits=5 cannot be greater than the number of
        members in each class' ValueError and crash the whole analysis."""
        rng = np.random.default_rng(3)
        X_train = rng.standard_normal((30, 10))
        y_train = np.array([0] * 27 + [1] * 3)  # minority class has only 3 members
        X_val = rng.standard_normal((10, 10))
        y_val = np.array([0] * 8 + [1] * 2)

        rf, rf_meta = train_rf(X_train, y_train, cv_folds=5)
        xgb, xgb_meta = train_xgb(X_train, y_train, X_val, y_val, cv_folds=5)

        assert isinstance(rf, RandomForestClassifier)
        assert isinstance(xgb, Booster)
        assert rf_meta["cv_f1_mean"] is not None  # 3 members still supports 3-fold CV
        assert xgb_meta["cv_f1_mean"] is not None

    def test_single_member_minority_class_returns_none_cv_score(self):
        """When a class has only 1 member, even 2-fold CV is impossible —
        cv_f1_mean must be None (not a crash, not a fabricated number)."""
        rng = np.random.default_rng(4)
        X_train = rng.standard_normal((21, 10))
        y_train = np.array([0] * 20 + [1] * 1)
        X_val = rng.standard_normal((10, 10))
        y_val = np.array([0] * 9 + [1] * 1)

        rf, rf_meta = train_rf(X_train, y_train, cv_folds=5)
        xgb, xgb_meta = train_xgb(X_train, y_train, X_val, y_val, cv_folds=5)

        assert isinstance(rf, RandomForestClassifier)
        assert isinstance(xgb, Booster)
        assert rf_meta["cv_f1_mean"] is None
        assert rf_meta["cv_f1_std"] is None
        assert xgb_meta["cv_f1_mean"] is None
        assert xgb_meta["cv_f1_std"] is None


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

    def test_single_class_test_set_returns_none_auc_not_nan(self, tiny_binary_xy):
        """Regression test: roc_auc_score is undefined when y_test has only
        one class present (e.g. the scarce class's lone member ended up in
        train via the non-stratified split fallback) — it returns NaN rather
        than raising, but NaN isn't valid JSON, so it must surface as None."""
        X, y = tiny_binary_xy
        rf, _ = train_rf(X[:40], y[:40], cv_folds=2)
        xgb, _ = train_xgb(X[:40], y[:40], X[40:], y[40:], cv_folds=2)
        y_test_single_class = np.zeros(10, dtype=int)

        result = evaluate_models(rf, xgb, X[40:50], y_test_single_class)

        for key in ("rf", "xgb", "ensemble"):
            assert result[key]["auc"] is None
            assert 0.0 <= result[key]["f1"] <= 1.0  # f1 still computes fine


class TestPositiveClassProba:
    def test_two_columns_returns_second_column(self):
        proba = np.array([[0.3, 0.7], [0.9, 0.1]])

        class _FakeClf:
            classes_ = np.array([0, 1])

        result = positive_class_proba(_FakeClf(), proba)
        assert list(result) == [0.7, 0.1]

    def test_single_column_class_one_returns_that_column(self):
        """When every training sample was flooded, predict_proba has a single
        column for class 1 — that column already is P(y=1)."""
        proba = np.array([[1.0], [1.0]])

        class _FakeClf:
            classes_ = np.array([1])

        result = positive_class_proba(_FakeClf(), proba)
        assert list(result) == [1.0, 1.0]

    def test_single_column_class_zero_returns_zeros(self):
        """When every training sample was dry, predict_proba has a single
        column for class 0 — P(y=1) is 0 everywhere."""
        proba = np.array([[1.0], [1.0]])

        class _FakeClf:
            classes_ = np.array([0])

        result = positive_class_proba(_FakeClf(), proba)
        assert list(result) == [0.0, 0.0]


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

    def test_single_model_selection_omits_uncertainty_and_comparison(self):
        """Single-model selections don't compute the other model at all, so
        eval_result only has the selected model's key — model_performance
        should reflect just that, and uncertainty should be omitted rather
        than crashing on a missing 'rf'/'xgb'/'ensemble' key."""
        n = 20
        probs = np.linspace(0, 1, n)
        actuals = (np.arange(n) % 2).tolist()
        pred = (probs > 0.5).astype(int).tolist()
        xgb_entry = {
            "label": "XGBoost",
            "f1": 0.8,
            "auc": 0.85,
            "threshold": 0.5,
            "predictions": pred,
            "probabilities": probs.tolist(),
        }
        eval_result = {"xgb": xgb_entry, "actuals": actuals}  # no "rf", no "ensemble"
        shap_payload = {"features": FEATURE_COLS, "mean_abs_shap": [0.1] * len(FEATURE_COLS)}

        result = build_flood_charts(eval_result, shap_payload, None, model_type="xgboost")

        assert "uncertainty" not in result
        assert result["model_performance"] == {
            "xgb": {"f1": 0.8, "auc": 0.85},
            "selected": "xgboost",
        }


# ── evaluate_models: single-model selection ────────────────────────────────────


class TestEvaluateModelsSingleModel:
    def test_rf_none_omits_rf_and_ensemble_keys(self, trained_models):
        _, xgb, X_te, y_te = trained_models
        result = evaluate_models(None, xgb, X_te, y_te)
        assert "rf" not in result
        assert "ensemble" not in result
        assert "xgb" in result

    def test_xgb_none_omits_xgb_and_ensemble_keys(self, trained_models):
        rf, _, X_te, y_te = trained_models
        result = evaluate_models(rf, None, X_te, y_te)
        assert "xgb" not in result
        assert "ensemble" not in result
        assert "rf" in result

    def test_both_none_returns_only_actuals(self, trained_models):
        _, _, X_te, y_te = trained_models
        result = evaluate_models(None, None, X_te, y_te)
        assert set(result.keys()) == {"actuals"}


# ── compute_shap_importance: works for either model type ───────────────────────


class TestComputeShapImportanceEitherModel:
    def test_rf_shap_returns_all_feature_cols(self, trained_models):
        rf, _, X_te, _ = trained_models
        result = compute_shap_importance(rf, X_te)
        assert sorted(result["features"]) == sorted(FEATURE_COLS)
        assert len(result["mean_abs_shap"]) == len(FEATURE_COLS)

    def test_xgb_shap_returns_all_feature_cols(self, trained_models):
        _, xgb, X_te, _ = trained_models
        result = compute_shap_importance(xgb, X_te)
        assert sorted(result["features"]) == sorted(FEATURE_COLS)
        assert len(result["mean_abs_shap"]) == len(FEATURE_COLS)


# ── FloodModel.predict: single-model selection trains only that model ──────────


@pytest.fixture()
def flood_training_df():
    rng = np.random.default_rng(3)
    n = 60
    df = pd.DataFrame(
        {
            "elevation": rng.standard_normal(n),
            "twi": rng.standard_normal(n),
            "dist_river": rng.standard_normal(n),
            "vv_change": rng.standard_normal(n),
            "rainfall_7d": rng.standard_normal(n),
            "rainfall_30d": rng.standard_normal(n),
            "mndwi": rng.standard_normal(n),
            "landcover": rng.standard_normal(n),
            "longitude": rng.uniform(30, 31, n),
            "latitude": rng.uniform(-1, 0, n),
        }
    )
    is_flooded = np.zeros(n, dtype=int)
    is_flooded[:30] = 1
    df["is_flooded"] = is_flooded
    return df


class TestFloodModelPredictSingleModel:
    def test_rf_only_never_trains_xgb(self, flood_training_df):
        model = FloodModel()
        result = model.predict(flood_training_df, {"model_type": "rf"})
        assert model.rf is not None
        assert model.xgb is None
        assert result["stats"]["xgb_f1"] is None
        assert result["stats"]["xgb_auc"] is None
        assert result["stats"]["ensemble_f1"] is None
        assert "uncertainty" not in result["charts"]
        assert "mean_spread" not in result["stats"]

    def test_xgboost_only_never_trains_rf(self, flood_training_df):
        model = FloodModel()
        result = model.predict(flood_training_df, {"model_type": "xgboost"})
        assert model.xgb is not None
        assert model.rf is None
        assert result["stats"]["rf_f1"] is None
        assert result["stats"]["rf_auc"] is None
        assert result["stats"]["ensemble_f1"] is None
        assert "uncertainty" not in result["charts"]

    def test_ensemble_trains_both(self, flood_training_df):
        model = FloodModel()
        result = model.predict(flood_training_df, {"model_type": "ensemble"})
        assert model.rf is not None
        assert model.xgb is not None
        assert result["stats"]["rf_f1"] is not None
        assert result["stats"]["xgb_f1"] is not None
        assert result["stats"]["ensemble_f1"] is not None
        assert "uncertainty" in result["charts"]

    def test_default_model_type_is_ensemble(self, flood_training_df):
        model = FloodModel()
        result = model.predict(flood_training_df, config=None)
        assert model.rf is not None
        assert model.xgb is not None
        assert result["stats"]["model_type"] == "ensemble"
