from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from xgboost import Booster, DMatrix
from xgboost import train as xgb_train

from .features import FEATURE_COLS

# Hyperparameters
RF_N_ESTIMATORS = 200
RF_MAX_DEPTH = 12
RF_MIN_SAMPLES_LEAF = 5

XGB_N_ESTIMATORS = 300
XGB_MAX_DEPTH = 6
XGB_LR = 0.05
XGB_SUBSAMPLE = 0.8
XGB_COLSAMPLE = 0.8

XGB_PARAMS = {
    "objective": "binary:logistic",
    "learning_rate": XGB_LR,
    "max_depth": XGB_MAX_DEPTH,
    "subsample": XGB_SUBSAMPLE,
    "colsample_bytree": XGB_COLSAMPLE,
    "eval_metric": "logloss",
    "seed": 42,
    "verbosity": 0,
}

VALID_MODEL_TYPES = ("rf", "xgboost", "ensemble")

RISK_COLORS: dict[str, str] = {
    "Low": "#2ECC71",
    "Medium": "#F1C40F",
    "High": "#E67E22",
    "Very High": "#E74C3C",
}


def _safe_cv_folds(y_train: np.ndarray, cv_folds: int) -> int | None:
    """Clamp cv_folds to the rarest class's sample count; None if CV isn't meaningful.

    StratifiedKFold/cross_val_score require every class to have at least
    n_splits members. A small or skewed AOI/date-range can easily sample
    fewer positive-class pixels than the default 5 folds, which otherwise
    crashes the whole analysis instead of just skipping CV. Uses np.unique
    (not np.bincount) so a class that's entirely absent — a legitimate,
    separately-handled data state — isn't mistaken for a scarce one.
    """
    _, counts = np.unique(np.asarray(y_train).astype(int), return_counts=True)
    min_class_count = int(counts.min()) if counts.size else 0
    if min_class_count < 2:
        return None
    return min(cv_folds, min_class_count)


def _safe_stratify(y: np.ndarray) -> np.ndarray | None:
    """Return y for a stratified train_test_split, or None to fall back to a
    plain random split.

    train_test_split(stratify=y) requires every class to have >= 2 members
    (one for train, one for test) — the same small/skewed-AOI scarcity that
    _safe_cv_folds guards against, but one step earlier in the pipeline.
    """
    _, counts = np.unique(np.asarray(y).astype(int), return_counts=True)
    if counts.size and counts.min() < 2:
        return None
    return y


# Risk classification
def classify_flood_risk(prob: np.ndarray) -> np.ndarray:
    """Map flood probability array to 4-class string labels."""
    risk = np.full(prob.shape, "Low", dtype=object)
    risk[(prob >= 0.25) & (prob < 0.50)] = "Medium"
    risk[(prob >= 0.50) & (prob < 0.75)] = "High"
    risk[prob >= 0.75] = "Very High"
    return risk


def train_rf(
    X_train: np.ndarray,
    y_train: np.ndarray,
    cv_folds: int = 5,
) -> tuple[RandomForestClassifier, dict]:
    """Fit a balanced Random Forest and report CV F1. Returns (model, metadata)."""
    rf = RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS,
        max_depth=RF_MAX_DEPTH,
        min_samples_leaf=RF_MIN_SAMPLES_LEAF,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    folds = _safe_cv_folds(y_train, cv_folds)
    if folds is None:
        return rf, {"cv_f1_mean": None, "cv_f1_std": None}
    cv_f1 = cross_val_score(rf, X_train, y_train, cv=folds, scoring="f1")
    return rf, {"cv_f1_mean": float(cv_f1.mean()), "cv_f1_std": float(cv_f1.std())}


def train_xgb(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    cv_folds: int = 5,
) -> tuple[Booster, dict]:
    """
    Fit XGBoost binary flood classifier via the low-level Booster API with
    scale_pos_weight and eval-set logging. Returns (model, metadata).

    Uses xgboost.train() rather than the XGBClassifier sklearn wrapper because
    the wrapper's fit() rejects a y whose sorted unique values aren't exactly
    [0, 1] — an AOI/time window where every sampled pixel is flooded (or none
    are) is a legitimate data state, not invalid input, and would otherwise
    crash training.
    """
    scale_pos_weight = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
    params = {**XGB_PARAMS, "scale_pos_weight": scale_pos_weight}
    dtrain = DMatrix(X_train, label=y_train)
    dval = DMatrix(X_val, label=y_val)
    booster = xgb_train(
        params,
        dtrain,
        num_boost_round=XGB_N_ESTIMATORS,
        evals=[(dval, "validation")],
        verbose_eval=False,
    )

    folds = _safe_cv_folds(y_train, cv_folds)
    if folds is None:
        return booster, {"cv_f1_mean": None, "cv_f1_std": None}
    cv = StratifiedKFold(n_splits=folds)
    cv_f1_scores = []
    for train_idx, val_idx in cv.split(X_train, y_train):
        fold_scale_pos_weight = float(
            (y_train[train_idx] == 0).sum() / max((y_train[train_idx] == 1).sum(), 1)
        )
        fold_booster = xgb_train(
            {**XGB_PARAMS, "scale_pos_weight": fold_scale_pos_weight},
            DMatrix(X_train[train_idx], label=y_train[train_idx]),
            num_boost_round=XGB_N_ESTIMATORS,
        )
        fold_pred = (fold_booster.predict(DMatrix(X_train[val_idx])) >= 0.5).astype(int)
        cv_f1_scores.append(f1_score(y_train[val_idx], fold_pred))

    cv_f1 = np.array(cv_f1_scores)
    return booster, {"cv_f1_mean": float(cv_f1.mean()), "cv_f1_std": float(cv_f1.std())}


def positive_class_proba(clf: RandomForestClassifier, proba: np.ndarray) -> np.ndarray:
    """
    Return P(y=1) from a binary sklearn classifier's predict_proba output.

    predict_proba only has one column when the classifier saw a single class
    during training (e.g. every sampled pixel in this AOI/time window was
    flooded, or none were) — proba[:, 1] would then raise IndexError. Falls
    back to reading clf.classes_ to know whether that lone column represents
    class 0 or class 1.
    """
    if proba.shape[1] == 2:
        return proba[:, 1]
    return proba[:, 0] if clf.classes_[0] == 1 else np.zeros(proba.shape[0])


# Threshold tuning
def find_best_threshold(probs: np.ndarray, y_true: np.ndarray) -> tuple[float, float]:
    """Return (threshold, best_f1) that maximises F1 on the precision-recall curve."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, probs)
    f1s = 2 * precisions * recalls / (precisions + recalls + 1e-9)
    best_idx = int(f1s.argmax())
    return float(thresholds[best_idx]), float(f1s[best_idx])


# Evaluation
def evaluate_models(
    rf: RandomForestClassifier | None,
    xgb: Booster | None,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    """
    Evaluate whichever of RF/XGBoost were actually trained (both, for an
    "ensemble" selection; exactly one otherwise) on the held-out test set,
    plus their ensemble when both are present. Thresholds are maximised
    per-model via the precision-recall curve.
    """

    def _metrics(prob: np.ndarray, label: str) -> dict:
        thresh, _ = find_best_threshold(prob, y_test)
        pred = (prob >= thresh).astype(int)
        # roc_auc_score is undefined (returns NaN, which isn't valid JSON) when
        # y_test ends up with only one class — a real possibility for a scarce
        # class routed through the non-stratified split fallback in predict().
        has_both_classes = len(np.unique(y_test)) > 1
        auc = round(float(roc_auc_score(y_test, prob)), 4) if has_both_classes else None
        return {
            "label": label,
            "f1": round(float(f1_score(y_test, pred)), 4),
            "auc": auc,
            "threshold": round(thresh, 4),
            "predictions": pred.tolist(),
            "probabilities": prob.round(4).tolist(),
        }

    rf_prob = positive_class_proba(rf, rf.predict_proba(X_test)) if rf is not None else None
    xgb_prob = xgb.predict(DMatrix(X_test)) if xgb is not None else None

    result: dict = {"actuals": y_test.tolist()}
    if rf_prob is not None:
        result["rf"] = _metrics(rf_prob, "Random Forest")
    if xgb_prob is not None:
        result["xgb"] = _metrics(xgb_prob, "XGBoost")
    if rf_prob is not None and xgb_prob is not None:
        result["ensemble"] = _metrics((rf_prob + xgb_prob) / 2.0, "Ensemble (mean prob)")
    return result


# SHAP
def compute_shap_importance(model: RandomForestClassifier | Booster, X_test: np.ndarray) -> dict:
    """TreeExplainer SHAP values for whichever model was actually selected/trained — see core.shap_utils for the axis-handling details."""
    from climate_change.core.shap_utils import compute_shap_importance as _compute_shap_importance

    return _compute_shap_importance(model, X_test, FEATURE_COLS)


# Uncertainty
def compute_uncertainty(rf_prob: np.ndarray, xgb_prob: np.ndarray) -> dict:
    """
    Epistemic uncertainty from RF–XGBoost probability spread.
    Pixels with spread > 0.20 are flagged for field validation.
    Only meaningful — and only computed — when both models were trained,
    i.e. an "ensemble" selection; single-model selections skip this
    entirely since there's no second model to compare against.
    """
    spread = np.abs(rf_prob - xgb_prob)
    return {
        "mean_spread": round(float(spread.mean()), 4),
        "high_uncertainty_pct": round(float((spread > 0.20).mean() * 100), 1),
        "spread_stats": {
            "min": round(float(spread.min()), 4),
            "p25": round(float(np.percentile(spread, 25)), 4),
            "p75": round(float(np.percentile(spread, 75)), 4),
            "max": round(float(spread.max()), 4),
            "mean": round(float(spread.mean()), 4),
        },
    }


# Chart payloads
# Maps the config key to the eval_result sub-key
_MODEL_TYPE_KEY: dict[str, str] = {
    "rf": "rf",
    "xgboost": "xgb",
    "ensemble": "ensemble",
}


def build_flood_charts(
    eval_result: dict,
    shap_payload: dict,
    uncertainty_payload: dict | None,
    model_type: str = "ensemble",
) -> dict:
    """
    Assemble frontend-ready chart payloads.

    Risk distribution is derived from the selected model_type's probabilities.
    model_performance includes metrics only for models actually present in
    eval_result — the full RF/XGBoost/Ensemble three-way comparison only
    when model_type == "ensemble" (both trained); otherwise just the
    selected model's own metrics. uncertainty is included only when
    uncertainty_payload is provided (ensemble selections only).
    """
    result_key = _MODEL_TYPE_KEY.get(model_type, "ensemble")
    selected_probs = np.array(eval_result[result_key]["probabilities"])

    _RISK_ORDER = ["Very High", "High", "Medium", "Low"]

    risk_labels = classify_flood_risk(selected_probs)
    counts = (
        pd.Series(risk_labels)
        .value_counts()
        .reindex(_RISK_ORDER)
        .fillna(0.0)
        .to_numpy(dtype=np.float64)
    )
    risk_pct = (counts / counts.sum() * 100).round(1)

    risk_chart = {
        "labels": _RISK_ORDER,
        "data": risk_pct.tolist(),
        "colors": [RISK_COLORS[c] for c in _RISK_ORDER],
    }

    model_performance: dict = {
        key: {"f1": eval_result[key]["f1"], "auc": eval_result[key]["auc"]}
        for key in ("rf", "xgb", "ensemble")
        if key in eval_result
    }
    model_performance["selected"] = model_type

    charts = {
        "risk_distribution": risk_chart,
        "shap": shap_payload,
        "model_performance": model_performance,
    }
    if uncertainty_payload is not None:
        charts["uncertainty"] = uncertainty_payload
    return charts


# FloodModel orchestrator
class FloodModel:
    """
    Orchestrates the full ML pipeline for a single flood event.
    config['model_type'] selects which model(s) are trained and which
    probabilities drive the primary stats, risk map, and COG export:
      "rf"       — Random Forest only
      "xgboost"  — XGBoost only
      "ensemble" — both, combined as the mean of RF + XGBoost (default)
    Only the selected model type is trained — an "rf"/"xgboost" selection
    never pays XGBoost's/RF's training cost. Trained models are stored on
    self.rf / self.xgb (whichever weren't needed stay None) so the use case
    can pass them to cog_export.export_flood_cog.
    """

    def __init__(self) -> None:
        self.rf: RandomForestClassifier | None = None
        self.xgb: Booster | None = None

    def predict(self, df: pd.DataFrame, config: dict | None = None) -> dict:
        """
        Parameters
        ----------
        df     : DataFrame with columns = FEATURE_COLS + ['is_flooded']
        config : optional dict; reads config['model_type'] (default 'ensemble')

        Returns
        -------
        dict with keys: stats, charts
        """
        model_type = (config or {}).get("model_type", "ensemble")
        if model_type not in VALID_MODEL_TYPES:
            raise ValueError(f"model_type must be one of {VALID_MODEL_TYPES}, got '{model_type}'")

        X = df[FEATURE_COLS].to_numpy(dtype=np.float64)
        y = df["is_flooded"].to_numpy(dtype=np.intp)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=_safe_stratify(y)
        )

        # Only train what this selection needs — "ensemble" needs both, "rf"
        # and "xgboost" need only the one selected. When both are needed,
        # train them concurrently on the Dask cluster if it's running,
        # otherwise fall back to sequential execution (e.g. during unit tests).
        from climate_change.core.dask_engine import DaskEngine

        need_rf = model_type in ("rf", "ensemble")
        need_xgb = model_type in ("xgboost", "ensemble")
        rf_meta: dict = {"cv_f1_mean": None, "cv_f1_std": None}
        xgb_meta: dict = {"cv_f1_mean": None, "cv_f1_std": None}

        client = DaskEngine.get_client_if_running() if (need_rf and need_xgb) else None
        if client is not None:
            f_rf = client.submit(train_rf, X_train, y_train, pure=False)
            f_xgb = client.submit(train_xgb, X_train, y_train, X_test, y_test, pure=False)
            (self.rf, rf_meta), (self.xgb, xgb_meta) = cast(list, client.gather([f_rf, f_xgb]))
        else:
            if need_rf:
                self.rf, rf_meta = train_rf(X_train, y_train)
            if need_xgb:
                self.xgb, xgb_meta = train_xgb(X_train, y_train, X_test, y_test)

        eval_result = evaluate_models(self.rf, self.xgb, X_test, y_test)

        if model_type == "rf":
            assert self.rf is not None
            shap_payload = compute_shap_importance(self.rf, X_test)
        else:
            assert self.xgb is not None
            shap_payload = compute_shap_importance(self.xgb, X_test)

        uncertainty = None
        if model_type == "ensemble":
            uncertainty = compute_uncertainty(
                np.array(eval_result["rf"]["probabilities"]),
                np.array(eval_result["xgb"]["probabilities"]),
            )
        charts = build_flood_charts(eval_result, shap_payload, uncertainty, model_type)
        # Risk percentages from the selected model
        result_key = _MODEL_TYPE_KEY.get(model_type, "ensemble")
        primary_probs = np.array(eval_result[result_key]["probabilities"])
        risk_labels = classify_flood_risk(primary_probs)

        _ORDER = ["Very High", "High", "Medium", "Low"]
        pct_arr = (
            pd.Series(risk_labels)
            .value_counts(normalize=True)
            .mul(100)
            .round(1)
            .reindex(_ORDER)
            .fillna(0.0)
            .to_numpy(dtype=np.float64)
        )
        very_high_pct, high_pct, medium_pct, low_pct = (float(v) for v in pct_arr)

        stats = {
            "model_type": model_type,
            "n_pixels_sampled": int(len(df)),
            "flooded_pct": round(float(y.mean() * 100), 1),
            "rf_cv_f1": round(rf_meta["cv_f1_mean"], 4)
            if rf_meta["cv_f1_mean"] is not None
            else None,
            "rf_f1": eval_result["rf"]["f1"] if "rf" in eval_result else None,
            "rf_auc": eval_result["rf"]["auc"] if "rf" in eval_result else None,
            "xgb_cv_f1": round(xgb_meta["cv_f1_mean"], 4)
            if xgb_meta["cv_f1_mean"] is not None
            else None,
            "xgb_f1": eval_result["xgb"]["f1"] if "xgb" in eval_result else None,
            "xgb_auc": eval_result["xgb"]["auc"] if "xgb" in eval_result else None,
            "ensemble_f1": eval_result["ensemble"]["f1"] if "ensemble" in eval_result else None,
            "ensemble_auc": eval_result["ensemble"]["auc"] if "ensemble" in eval_result else None,
            "selected_f1": eval_result[result_key]["f1"],
            "selected_auc": eval_result[result_key]["auc"],
            "selected_threshold": eval_result[result_key]["threshold"],
            "top_flood_driver": shap_payload["features"][0],
            "very_high_risk_pct": very_high_pct,
            "high_risk_pct": high_pct,
            "medium_risk_pct": medium_pct,
            "low_risk_pct": low_pct,
        }
        if uncertainty is not None:
            stats.update(uncertainty)

        lon_idx = FEATURE_COLS.index("longitude")
        lat_idx = FEATURE_COLS.index("latitude")
        _sample_points = [
            {
                "lon": round(float(X_test[i, lon_idx]), 5),
                "lat": round(float(X_test[i, lat_idx]), 5),
                "risk_class": str(risk_labels[i]),
            }
            for i in range(len(X_test))
        ]
        return {"stats": stats, "charts": charts, "_sample_points": _sample_points}
