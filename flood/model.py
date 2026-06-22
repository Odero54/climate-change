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
from sklearn.model_selection import cross_val_score, train_test_split
from xgboost.sklearn import XGBClassifier

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

VALID_MODEL_TYPES = ("rf", "xgboost", "ensemble")

RISK_COLORS: dict[str, str] = {
    "Low": "#2ECC71",
    "Medium": "#F1C40F",
    "High": "#E67E22",
    "Very High": "#E74C3C",
}


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
    cv_f1 = cross_val_score(rf, X_train, y_train, cv=cv_folds, scoring="f1")
    return rf, {"cv_f1_mean": float(cv_f1.mean()), "cv_f1_std": float(cv_f1.std())}


def train_xgb(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    cv_folds: int = 5,
) -> tuple[XGBClassifier, dict]:
    """Fit XGBoost with scale_pos_weight and eval-set logging. Returns (model, metadata)."""
    scale_pos_weight = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
    xgb = XGBClassifier(
        n_estimators=XGB_N_ESTIMATORS,
        max_depth=XGB_MAX_DEPTH,
        learning_rate=XGB_LR,
        subsample=XGB_SUBSAMPLE,
        colsample_bytree=XGB_COLSAMPLE,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=42,
        verbosity=0,
    )
    xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    cv_f1 = cross_val_score(xgb, X_train, y_train, cv=cv_folds, scoring="f1")
    return xgb, {"cv_f1_mean": float(cv_f1.mean()), "cv_f1_std": float(cv_f1.std())}


# Threshold tuning
def find_best_threshold(probs: np.ndarray, y_true: np.ndarray) -> tuple[float, float]:
    """Return (threshold, best_f1) that maximises F1 on the precision-recall curve."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, probs)
    f1s = 2 * precisions * recalls / (precisions + recalls + 1e-9)
    best_idx = int(f1s.argmax())
    return float(thresholds[best_idx]), float(f1s[best_idx])


# Evaluation
def evaluate_models(
    rf: RandomForestClassifier,
    xgb: XGBClassifier,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    """
    Evaluate RF, XGBoost, and their ensemble on the held-out test set.
    Thresholds are maximised per-model via the precision-recall curve.
    """
    rf_prob = rf.predict_proba(X_test)[:, 1]
    xgb_prob = xgb.predict_proba(X_test)[:, 1]
    ens_prob = (rf_prob + xgb_prob) / 2.0

    def _metrics(prob: np.ndarray, label: str) -> dict:
        thresh, _ = find_best_threshold(prob, y_test)
        pred = (prob >= thresh).astype(int)
        return {
            "label": label,
            "f1": round(float(f1_score(y_test, pred)), 4),
            "auc": round(float(roc_auc_score(y_test, prob)), 4),
            "threshold": round(thresh, 4),
            "predictions": pred.tolist(),
            "probabilities": prob.round(4).tolist(),
        }

    return {
        "rf": _metrics(rf_prob, "Random Forest"),
        "xgb": _metrics(xgb_prob, "XGBoost"),
        "ensemble": _metrics(ens_prob, "Ensemble (mean prob)"),
        "actuals": y_test.tolist(),
    }


# SHAP
def compute_shap_importance(xgb: XGBClassifier, X_test: np.ndarray) -> dict:
    """
    TreeExplainer SHAP values for XGBoost, sorted by descending mean |SHAP|.
    XGBoost is always used for SHAP regardless of model_type — it provides
    the most interpretable tree-based explanations.
    """
    import shap

    explainer = shap.TreeExplainer(xgb)
    shap_vals = explainer.shap_values(X_test)
    mean_abs = np.abs(shap_vals).mean(axis=0)
    rank_idx = np.argsort(mean_abs)[::-1]
    return {
        "features": [FEATURE_COLS[i] for i in rank_idx],
        "mean_abs_shap": mean_abs[rank_idx].round(4).tolist(),
    }


# Uncertainty
def compute_uncertainty(rf_prob: np.ndarray, xgb_prob: np.ndarray) -> dict:
    """
    Epistemic uncertainty from RF–XGBoost probability spread.
    Pixels with spread > 0.20 are flagged for field validation.
    Always computed regardless of model_type so the UI can display it.
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
    uncertainty_payload: dict,
    model_type: str = "ensemble",
) -> dict:
    """
    Assemble frontend-ready chart payloads.

    Risk distribution is derived from the selected model_type's probabilities.
    model_performance always includes all three models for comparison.
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

    # All three models always shown for comparison
    model_performance = {
        "rf": {"f1": eval_result["rf"]["f1"], "auc": eval_result["rf"]["auc"]},
        "xgb": {"f1": eval_result["xgb"]["f1"], "auc": eval_result["xgb"]["auc"]},
        "ensemble": {
            "f1": eval_result["ensemble"]["f1"],
            "auc": eval_result["ensemble"]["auc"],
        },
        "selected": model_type,
    }

    return {
        "risk_distribution": risk_chart,
        "shap": shap_payload,
        "uncertainty": uncertainty_payload,
        "model_performance": model_performance,
    }


# FloodModel orchestrator
class FloodModel:
    """
    Orchestrates the full ML pipeline for a single flood event.
    Always trains both RF and XGBoost. config['model_type'] selects which
    probabilities drive the primary stats, risk map, and COG export:
      "rf"       — Random Forest
      "xgboost"  — XGBoost
      "ensemble" — mean of RF + XGBoost (default)
    Trained models are stored on self.rf / self.xgb so the use case can
    pass them directly to cog_export.export_flood_cog.
    """

    def __init__(self) -> None:
        self.rf: RandomForestClassifier | None = None
        self.xgb: XGBClassifier | None = None

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
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Train RF and XGBoost concurrently when the Dask cluster is running,
        # otherwise fall back to sequential execution (e.g. during unit tests).
        from climate_change.core.dask_engine import DaskEngine

        client = DaskEngine.get_client_if_running()
        if client is not None:
            f_rf = client.submit(train_rf, X_train, y_train, pure=False)
            f_xgb = client.submit(train_xgb, X_train, y_train, X_test, y_test, pure=False)
            (self.rf, rf_meta), (self.xgb, xgb_meta) = cast(list, client.gather([f_rf, f_xgb]))
        else:
            self.rf, rf_meta = train_rf(X_train, y_train)
            self.xgb, xgb_meta = train_xgb(X_train, y_train, X_test, y_test)

        assert self.rf is not None
        assert self.xgb is not None

        eval_result = evaluate_models(self.rf, self.xgb, X_test, y_test)
        shap_payload = compute_shap_importance(self.xgb, X_test)
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
            "rf_cv_f1": round(rf_meta["cv_f1_mean"], 4),
            "rf_f1": eval_result["rf"]["f1"],
            "rf_auc": eval_result["rf"]["auc"],
            "xgb_cv_f1": round(xgb_meta["cv_f1_mean"], 4),
            "xgb_f1": eval_result["xgb"]["f1"],
            "xgb_auc": eval_result["xgb"]["auc"],
            "ensemble_f1": eval_result["ensemble"]["f1"],
            "ensemble_auc": eval_result["ensemble"]["auc"],
            "selected_f1": eval_result[result_key]["f1"],
            "selected_auc": eval_result[result_key]["auc"],
            "selected_threshold": eval_result[result_key]["threshold"],
            "top_flood_driver": shap_payload["features"][0],
            "very_high_risk_pct": very_high_pct,
            "high_risk_pct": high_pct,
            "medium_risk_pct": medium_pct,
            "low_risk_pct": low_pct,
            **uncertainty,
        }

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
