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
from xgboost import Booster, DMatrix
from xgboost import train as xgb_train

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


def _safe_cv_folds(y_train: np.ndarray, cv_folds: int) -> int | None:
    """Clamp cv_folds to the rarest class's sample count; None if CV isn't meaningful.

    StratifiedKFold/cross_val_score require every class to have at least
    n_splits members. A small or skewed AOI/date-range can easily sample
    fewer pixels of the rarest risk class than the default 5 folds, which
    otherwise crashes the whole analysis instead of just skipping CV. Uses
    np.unique (not np.bincount) so a class that's entirely absent — a
    legitimate, separately-handled data state — isn't mistaken for a scarce one.
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
    folds = _safe_cv_folds(y_train, cv_folds)
    if folds is None:
        return clf, {"cv_f1_mean": None, "cv_f1_std": None}
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    cv_f1 = cross_val_score(clf, X_train, y_train, cv=cv, scoring="f1_macro")
    return clf, {"cv_f1_mean": float(cv_f1.mean()), "cv_f1_std": float(cv_f1.std())}


XGB_PARAMS = {
    "objective": "multi:softprob",
    "num_class": len(DISEASE_CLASSES),
    "learning_rate": XGB_LR,
    "max_depth": XGB_MAX_DEPTH,
    "subsample": XGB_SUBSAMPLE,
    "colsample_bytree": XGB_COLSAMPLE,
    "eval_metric": "mlogloss",
    "seed": 42,
    "verbosity": 0,
}


def train_xgb(
    X_train: np.ndarray,
    y_train: np.ndarray,
    cv_folds: int = 5,
) -> tuple[Booster, dict]:
    """
    Fit XGBoost multi-class classifier via the low-level Booster API with sample weights.

    Uses xgboost.train() rather than the XGBClassifier sklearn wrapper because the
    wrapper requires every class in DISEASE_CLASSES to appear in y_train (it infers
    num_class from np.unique(y_train) and rejects gaps). Some AOIs/time windows
    genuinely have zero samples of a given risk class, which is a legitimate state,
    not invalid input, so class count is fixed via XGB_PARAMS instead.

    Returns (booster, metadata).
    """
    sw = compute_sample_weight("balanced", y_train)
    dtrain = DMatrix(X_train, label=y_train, weight=sw)
    booster = xgb_train(XGB_PARAMS, dtrain, num_boost_round=XGB_N_ESTIMATORS)

    folds = _safe_cv_folds(y_train, cv_folds)
    if folds is None:
        return booster, {"cv_f1_mean": None, "cv_f1_std": None}
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    cv_f1_scores = []
    for train_idx, val_idx in cv.split(X_train, y_train):
        fold_booster = xgb_train(
            XGB_PARAMS,
            DMatrix(X_train[train_idx], label=y_train[train_idx]),
            num_boost_round=XGB_N_ESTIMATORS,
        )
        fold_pred = np.argmax(fold_booster.predict(DMatrix(X_train[val_idx])), axis=1)
        cv_f1_scores.append(f1_score(y_train[val_idx], fold_pred, average="macro"))

    cv_f1 = np.array(cv_f1_scores)
    return booster, {"cv_f1_mean": float(cv_f1.mean()), "cv_f1_std": float(cv_f1.std())}


def pad_gbm_proba(gbm: GradientBoostingClassifier, proba: np.ndarray) -> np.ndarray:
    """
    Expand GBM's predict_proba output to the full DISEASE_CLASSES width.

    GradientBoostingClassifier only emits a column per class it actually saw
    during fit (via gbm.classes_), whereas train_xgb's Booster always outputs
    len(DISEASE_CLASSES) columns regardless of what the AOI's data contained.
    Without padding, an AOI missing one risk class produces mismatched shapes
    (e.g. (n, 2) vs (n, 3)) the moment GBM and XGBoost probabilities are
    combined for the ensemble.
    """
    n_classes = len(DISEASE_CLASSES)
    if proba.shape[1] == n_classes:
        return proba
    padded = np.zeros((proba.shape[0], n_classes), dtype=proba.dtype)
    padded[:, gbm.classes_.astype(int)] = proba
    return padded


def evaluate_models(
    gbm: GradientBoostingClassifier | None,
    xgb: Booster | None,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    """
    Evaluate whichever of GBM/XGBoost were actually trained (both, for an
    "ensemble" selection; exactly one otherwise) on the held-out test set,
    plus their mean-proba ensemble when both are present.
    """

    def _metrics(pred: np.ndarray, label: str) -> dict:
        return {
            "label": label,
            "f1": round(float(f1_score(y_test, pred, average="macro")), 4),
            "accuracy": round(float(accuracy_score(y_test, pred)), 4),
            "predictions": pred.tolist(),
        }

    proba_gbm = pad_gbm_proba(gbm, gbm.predict_proba(X_test)) if gbm is not None else None
    proba_xgb = xgb.predict(DMatrix(X_test)) if xgb is not None else None

    result: dict = {"actuals": y_test.tolist()}
    if proba_gbm is not None:
        result["gbm"] = _metrics(np.argmax(proba_gbm, axis=1).astype(int), "Gradient Boosting")
    if proba_xgb is not None:
        result["xgb"] = _metrics(np.argmax(proba_xgb, axis=1).astype(int), "XGBoost")
    if proba_gbm is not None and proba_xgb is not None:
        ens_pred = np.argmax((proba_gbm + proba_xgb) / 2.0, axis=1).astype(int)
        result["ensemble"] = _metrics(ens_pred, "Ensemble (mean proba)")
    return result


def compute_shap_importance(
    model: GradientBoostingClassifier | Booster, X_test: np.ndarray
) -> dict:
    """TreeExplainer SHAP values for whichever model was actually selected/trained — see core.shap_utils for the axis-handling details."""
    from climate_change.core.shap_utils import compute_shap_importance as _compute_shap_importance

    X_df = pd.DataFrame(X_test, columns=FEATURE_COLS)
    return _compute_shap_importance(model, X_df, FEATURE_COLS)


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
                "lon": round(cast(float, hr_df.loc[mask, "lon"].mean()), 4),
                "lat": round(cast(float, hr_df.loc[mask, "lat"].mean()), 4),
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
            **{
                key: {"f1": eval_result[key]["f1"], "accuracy": eval_result[key]["accuracy"]}
                for key in ("gbm", "xgb", "ensemble")
                if key in eval_result
            },
            "selected": model_type,
        },
    }


class DiseaseModel:
    """
    Orchestrates the full ML pipeline for a single disease surveillance analysis.
    config['model_type'] selects which model(s) are trained and which
    predictions drive the primary risk distribution:
      "gbm"      — Gradient Boosting only (default, highest accuracy per lab)
      "xgboost"  — XGBoost only
      "ensemble" — both, combined as mean softmax probabilities of GBM + XGBoost
    Only the selected model type is trained. Trained models and scaler are
    stored on self for use by cog_export (whichever model wasn't needed
    stays None).
    """

    def __init__(self) -> None:
        self.gbm: GradientBoostingClassifier | None = None
        self.xgb: Booster | None = None
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
            X, y, test_size=0.20, random_state=42, stratify=_safe_stratify(y)
        )
        self.scaler = StandardScaler()
        X_train_s = self.scaler.fit_transform(X_train)
        X_test_s = self.scaler.transform(X_test)

        # Only train what this selection needs — "ensemble" needs both, "gbm"
        # and "xgboost" need only the one selected. When both are needed,
        # train them concurrently on the Dask cluster if it's running,
        # otherwise fall back to sequential execution (e.g. during unit tests).
        from climate_change.core.dask_engine import DaskEngine

        need_gbm = model_type in ("gbm", "ensemble")
        need_xgb = model_type in ("xgboost", "ensemble")
        gbm_meta: dict = {"cv_f1_mean": None, "cv_f1_std": None}
        xgb_meta: dict = {"cv_f1_mean": None, "cv_f1_std": None}

        client = DaskEngine.get_client_if_running() if (need_gbm and need_xgb) else None
        if client is not None:
            f_gbm = client.submit(train_gbm, X_train_s, y_train, pure=False)
            f_xgb = client.submit(train_xgb, X_train_s, y_train, pure=False)
            (self.gbm, gbm_meta), (self.xgb, xgb_meta) = cast(list, client.gather([f_gbm, f_xgb]))
        else:
            if need_gbm:
                self.gbm, gbm_meta = train_gbm(X_train_s, y_train)
            if need_xgb:
                self.xgb, xgb_meta = train_xgb(X_train_s, y_train)

        eval_result = evaluate_models(self.gbm, self.xgb, X_test_s, y_test)
        if model_type == "gbm":
            assert self.gbm is not None
            shap_payload = compute_shap_importance(self.gbm, X_test_s)
        else:
            assert self.xgb is not None
            shap_payload = compute_shap_importance(self.xgb, X_test_s)

        # Hotspot detection — applied to all pixels using the selected model
        X_all_s = self.scaler.transform(
            df[FEATURE_COLS].fillna(df[FEATURE_COLS].median()).to_numpy(dtype=np.float64)
        )
        _KEY = {"gbm": "gbm", "xgboost": "xgb", "ensemble": "ensemble"}
        result_key = _KEY.get(model_type, "gbm")

        if model_type == "gbm":
            assert self.gbm is not None
            all_preds = self.gbm.predict(X_all_s).astype(int)
        elif model_type == "xgboost":
            assert self.xgb is not None
            all_preds = np.argmax(self.xgb.predict(DMatrix(X_all_s)), axis=1).astype(int)
        else:
            assert self.gbm is not None
            assert self.xgb is not None
            proba_gbm = pad_gbm_proba(self.gbm, self.gbm.predict_proba(X_all_s))
            proba = (proba_gbm + self.xgb.predict(DMatrix(X_all_s))) / 2.0
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
            "gbm_cv_f1": round(gbm_meta["cv_f1_mean"], 4)
            if gbm_meta["cv_f1_mean"] is not None
            else None,
            "gbm_f1": eval_result["gbm"]["f1"] if "gbm" in eval_result else None,
            "gbm_accuracy": eval_result["gbm"]["accuracy"] if "gbm" in eval_result else None,
            "xgb_cv_f1": round(xgb_meta["cv_f1_mean"], 4)
            if xgb_meta["cv_f1_mean"] is not None
            else None,
            "xgb_f1": eval_result["xgb"]["f1"] if "xgb" in eval_result else None,
            "xgb_accuracy": eval_result["xgb"]["accuracy"] if "xgb" in eval_result else None,
            "ensemble_f1": eval_result["ensemble"]["f1"] if "ensemble" in eval_result else None,
            "selected_f1": eval_result[result_key]["f1"],
            "high_risk_pct": round(high_risk_pct, 1),
            "n_hotspot_clusters": len(hotspots),
            "top_driver": shap_payload["features"][0],
        }

        _DISEASE_CLASS_NAMES = ["Low Risk", "Medium Risk", "High Risk"]
        if "lon" in df.columns and "lat" in df.columns:
            _sample_points = [
                {
                    "lon": round(cast(float, df["lon"].iat[i]), 5),
                    "lat": round(cast(float, df["lat"].iat[i]), 5),
                    "risk_class": _DISEASE_CLASS_NAMES[int(all_preds[i])],
                }
                for i in range(len(df))
            ]
        else:
            _sample_points = []
        return {"stats": stats, "charts": charts, "_sample_points": _sample_points}
