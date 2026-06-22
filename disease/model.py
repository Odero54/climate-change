from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from .features import DISEASE_CLASSES, DISEASE_COLORS, FEATURE_COLS

VALID_MODEL_TYPES = ("gbm", "xgboost", "ensemble")

# GradientBoosting hyperparameters
GBM_N_ESTIMATORS = 200
GBM_LR = 0.05
GBM_MAX_DEPTH = 4
GBM_SUBSAMPLE = 0.8

# XGBoost hyperparameters
XGB_N_ESTIMATORS = 200
XGB_LR = 0.05
XGB_MAX_DEPTH = 5
XGB_SUBSAMPLE = 0.8
XGB_COLSAMPLE = 0.8

# DBSCAN spatial hotspot parameters
# eps ≈ 0.09° ≈ 10 km at the equator; min_samples = 3 for a meaningful cluster
DBSCAN_EPS = 0.09
DBSCAN_MIN_SAMPLES = 3


def train_gbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    cv_folds: int = 5,
) -> tuple[GradientBoostingClassifier, dict]:
    """Fit Gradient Boosting Classifier with sample weights. Returns (model, metadata)."""
    sw = compute_sample_weight("balanced", y_train)
    clf = GradientBoostingClassifier(
        n_estimators=GBM_N_ESTIMATORS,
        learning_rate=GBM_LR,
        max_depth=GBM_MAX_DEPTH,
        subsample=GBM_SUBSAMPLE,
        random_state=42,
    )
    clf.fit(X_train, y_train, sample_weight=sw)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    cv_f1 = cross_val_score(clf, X_train, y_train, cv=cv, scoring="f1_macro")
    return clf, {"cv_f1_mean": float(cv_f1.mean()), "cv_f1_std": float(cv_f1.std())}


def train_xgb(
    X_train: np.ndarray,
    y_train: np.ndarray,
    cv_folds: int = 5,
) -> tuple[XGBClassifier, dict]:
    """Fit XGBoost multi-class classifier with sample weights. Returns (model, metadata)."""
    sw = compute_sample_weight("balanced", y_train)
    clf = XGBClassifier(
        n_estimators=XGB_N_ESTIMATORS,
        learning_rate=XGB_LR,
        max_depth=XGB_MAX_DEPTH,
        subsample=XGB_SUBSAMPLE,
        colsample_bytree=XGB_COLSAMPLE,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )
    clf.fit(X_train, y_train, sample_weight=sw, verbose=False)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    cv_f1 = cross_val_score(clf, X_train, y_train, cv=cv, scoring="f1_macro")
    return clf, {"cv_f1_mean": float(cv_f1.mean()), "cv_f1_std": float(cv_f1.std())}


def evaluate_models(
    gbm: GradientBoostingClassifier,
    xgb: XGBClassifier,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    """Evaluate GBM, XGBoost, and mean-proba ensemble on the held-out test set."""
    gbm_pred = gbm.predict(X_test).astype(int)
    xgb_pred = xgb.predict(X_test).astype(int)
    proba_gbm = gbm.predict_proba(X_test)
    proba_xgb = xgb.predict_proba(X_test)
    ens_pred = np.argmax((proba_gbm + proba_xgb) / 2.0, axis=1).astype(int)

    def _metrics(pred: np.ndarray, label: str) -> dict:
        return {
            "label": label,
            "f1": round(float(f1_score(y_test, pred, average="macro")), 4),
            "accuracy": round(float(accuracy_score(y_test, pred)), 4),
            "predictions": pred.tolist(),
        }

    return {
        "gbm": _metrics(gbm_pred, "Gradient Boosting"),
        "xgb": _metrics(xgb_pred, "XGBoost"),
        "ensemble": _metrics(ens_pred, "Ensemble (mean proba)"),
        "actuals": y_test.tolist(),
    }


def compute_shap_importance(xgb: XGBClassifier, X_test: np.ndarray) -> dict:
    """
    TreeExplainer SHAP on XGBoost (always used for SHAP regardless of model_type).
    Returns features sorted by descending mean |SHAP| averaged across all classes.
    """
    import shap

    X_df = pd.DataFrame(X_test, columns=FEATURE_COLS)
    explainer = shap.TreeExplainer(xgb)
    shap_vals = explainer.shap_values(X_df)

    if isinstance(shap_vals, list):
        mean_abs = np.array([np.abs(sv).mean(axis=0) for sv in shap_vals]).mean(axis=0)
    elif np.asarray(shap_vals).ndim == 3:
        mean_abs = np.abs(shap_vals).mean(axis=(0, 2))
    else:
        mean_abs = np.abs(shap_vals).mean(axis=0)

    rank_idx = np.argsort(mean_abs)[::-1]
    return {
        "features": [FEATURE_COLS[i] for i in rank_idx],
        "mean_abs_shap": mean_abs[rank_idx].round(4).tolist(),
    }


def detect_hotspots(
    df: pd.DataFrame,
    pred_labels: np.ndarray,
    eps: float = DBSCAN_EPS,
    min_samples: int = DBSCAN_MIN_SAMPLES,
) -> list[dict]:
    """
    DBSCAN spatial hotspot detection on High Risk (class 2) pixel centroids.
    Returns a list of cluster dicts with cluster_id, size, lon, lat.
    Requires df to contain 'lon' and 'lat' columns (preserved from GEE sample geometries).
    """
    if "lon" not in df.columns or "lat" not in df.columns:
        return []

    high_risk_mask = pred_labels == 2
    hr_df = df.loc[high_risk_mask, ["lon", "lat"]].reset_index(drop=True)
    if len(hr_df) < min_samples:
        return []

    coords = hr_df[["lon", "lat"]].to_numpy()
    cluster_labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(coords)

    hotspots = []
    for cid in sorted(set(cluster_labels)):
        if cid == -1:
            continue
        mask = cluster_labels == cid
        hotspots.append(
            {
                "cluster_id": int(cid),
                "size": int(mask.sum()),
                "lon": round(float(hr_df.loc[mask, "lon"].mean()), 4),
                "lat": round(float(hr_df.loc[mask, "lat"].mean()), 4),
            }
        )
    return sorted(hotspots, key=lambda h: h["size"], reverse=True)


def build_disease_charts(
    eval_result: dict,
    shap_payload: dict,
    timeseries: dict[str, pd.DataFrame],
    hotspots: list[dict],
    model_type: str = "gbm",
) -> dict:
    """Assemble frontend-ready chart payloads for the disease surveillance module."""
    _KEY = {"gbm": "gbm", "xgboost": "xgb", "ensemble": "ensemble"}
    result_key = _KEY.get(model_type, "gbm")
    predictions = np.array(eval_result[result_key]["predictions"])

    n_total = len(predictions)
    counts = np.array([(predictions == i).sum() for i in range(3)], dtype=np.float64)
    risk_pct = (counts / n_total * 100).round(1)

    risk_dist = {
        "labels": DISEASE_CLASSES,
        "data": risk_pct.tolist(),
        "colors": DISEASE_COLORS,
    }

    # Build time series datasets aligned to the NDVI index
    ndvi_df = timeseries.get("ndvi", pd.DataFrame())
    rain_df = timeseries.get("rain", pd.DataFrame())
    lst_df = timeseries.get("lst", pd.DataFrame())

    ts_labels = ndvi_df.index.tolist() if not ndvi_df.empty else []
    ts_datasets = []
    if not ndvi_df.empty:
        ts_datasets.append(
            {
                "label": "NDVI",
                "data": ndvi_df["ndvi"].round(4).tolist(),
                "color": "#27AE60",
            }
        )
    if not rain_df.empty:
        aligned_rain = rain_df["rain_mm"].reindex(ts_labels).round(1).tolist()
        ts_datasets.append(
            {
                "label": "Monthly rain (mm)",
                "data": aligned_rain,
                "color": "#2980B9",
            }
        )
    if not lst_df.empty:
        aligned_lst = lst_df["lst"].reindex(ts_labels).round(2).tolist()
        ts_datasets.append(
            {
                "label": "LST (°C)",
                "data": aligned_lst,
                "color": "#E74C3C",
            }
        )

    return {
        "riskDist": risk_dist,
        "timeSeries": {"labels": ts_labels, "datasets": ts_datasets},
        "shap": shap_payload,
        "hotspots": hotspots,
        "model_performance": {
            "gbm": {
                "f1": eval_result["gbm"]["f1"],
                "accuracy": eval_result["gbm"]["accuracy"],
            },
            "xgb": {
                "f1": eval_result["xgb"]["f1"],
                "accuracy": eval_result["xgb"]["accuracy"],
            },
            "ensemble": {
                "f1": eval_result["ensemble"]["f1"],
                "accuracy": eval_result["ensemble"]["accuracy"],
            },
            "selected": model_type,
        },
    }


class DiseaseModel:
    """
    Orchestrates the full ML pipeline for a single disease surveillance analysis.
    Always trains Gradient Boosting + XGBoost. config['model_type'] selects which
    predictions drive the primary risk distribution:
      "gbm"      — Gradient Boosting (default, highest accuracy per lab)
      "xgboost"  — XGBoost
      "ensemble" — mean softmax probabilities of GBM + XGBoost
    Trained models and scaler are stored on self for use by cog_export.
    """

    def __init__(self) -> None:
        self.gbm: GradientBoostingClassifier | None = None
        self.xgb: XGBClassifier | None = None
        self.scaler: StandardScaler | None = None

    def predict(
        self,
        df: pd.DataFrame,
        timeseries: dict[str, pd.DataFrame] | None = None,
        config: dict | None = None,
    ) -> dict:
        """
        Parameters
        ----------
        df          : DataFrame with FEATURE_COLS + ['lon', 'lat', 'risk_score', 'label']
        timeseries  : dict of monthly DataFrames (ndvi, rain, lst) from fetch_monthly_timeseries
        config      : optional dict; reads 'model_type' (default 'gbm')

        Returns
        -------
        dict with keys: stats, charts
        """
        cfg = config or {}
        model_type = cfg.get("model_type", "gbm")
        if model_type not in VALID_MODEL_TYPES:
            raise ValueError(f"model_type must be one of {VALID_MODEL_TYPES}, got '{model_type}'")

        ts = timeseries or {}
        X = df[FEATURE_COLS].fillna(df[FEATURE_COLS].median()).to_numpy(dtype=np.float64)
        y = df["label"].to_numpy(dtype=np.intp)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=42, stratify=y
        )
        self.scaler = StandardScaler()
        X_train_s = self.scaler.fit_transform(X_train)
        X_test_s = self.scaler.transform(X_test)

        # Train GBM and XGBoost concurrently when the Dask cluster is running,
        # otherwise fall back to sequential execution (e.g. during unit tests).
        from climate_change.core.dask_engine import DaskEngine

        client = DaskEngine.get_client_if_running()
        if client is not None:
            f_gbm = client.submit(train_gbm, X_train_s, y_train, pure=False)
            f_xgb = client.submit(train_xgb, X_train_s, y_train, pure=False)
            (self.gbm, gbm_meta), (self.xgb, xgb_meta) = cast(list, client.gather([f_gbm, f_xgb]))
        else:
            self.gbm, gbm_meta = train_gbm(X_train_s, y_train)
            self.xgb, xgb_meta = train_xgb(X_train_s, y_train)

        assert self.gbm is not None
        assert self.xgb is not None

        eval_result = evaluate_models(self.gbm, self.xgb, X_test_s, y_test)
        shap_payload = compute_shap_importance(self.xgb, X_test_s)

        # Hotspot detection — applied to all pixels using the selected model
        X_all_s = self.scaler.transform(
            df[FEATURE_COLS].fillna(df[FEATURE_COLS].median()).to_numpy(dtype=np.float64)
        )
        _KEY = {"gbm": "gbm", "xgboost": "xgb", "ensemble": "ensemble"}
        result_key = _KEY.get(model_type, "gbm")

        if model_type == "gbm":
            all_preds = self.gbm.predict(X_all_s).astype(int)
        elif model_type == "xgboost":
            all_preds = self.xgb.predict(X_all_s).astype(int)
        else:
            proba = (self.gbm.predict_proba(X_all_s) + self.xgb.predict_proba(X_all_s)) / 2.0
            all_preds = np.argmax(proba, axis=1).astype(int)

        hotspots = detect_hotspots(df, all_preds)
        charts = build_disease_charts(eval_result, shap_payload, ts, hotspots, model_type)

        # Risk distribution on all pixels
        n_total = len(all_preds)
        counts = np.array([(all_preds == i).sum() for i in range(3)], dtype=np.float64)
        risk_pct = (counts / n_total * 100).round(1)
        high_risk_pct = float(risk_pct[2])

        stats = {
            "model_type": model_type,
            "n_pixels_sampled": int(len(df)),
            "gbm_cv_f1": round(gbm_meta["cv_f1_mean"], 4),
            "gbm_f1": eval_result["gbm"]["f1"],
            "gbm_accuracy": eval_result["gbm"]["accuracy"],
            "xgb_cv_f1": round(xgb_meta["cv_f1_mean"], 4),
            "xgb_f1": eval_result["xgb"]["f1"],
            "xgb_accuracy": eval_result["xgb"]["accuracy"],
            "ensemble_f1": eval_result["ensemble"]["f1"],
            "selected_f1": eval_result[result_key]["f1"],
            "high_risk_pct": round(high_risk_pct, 1),
            "n_hotspot_clusters": len(hotspots),
            "top_driver": shap_payload["features"][0],
        }

        _DISEASE_CLASS_NAMES = ["Low Risk", "Medium Risk", "High Risk"]
        if "lon" in df.columns and "lat" in df.columns:
            _sample_points = [
                {
                    "lon": round(float(df["lon"].iat[i]), 5),
                    "lat": round(float(df["lat"].iat[i]), 5),
                    "risk_class": _DISEASE_CLASS_NAMES[int(all_preds[i])],
                }
                for i in range(len(df))
            ]
        else:
            _sample_points = []
        return {"stats": stats, "charts": charts, "_sample_points": _sample_points}
