from __future__ import annotations

from typing import Optional, cast

import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from .features import FEATURE_COLS, FOOD_CLASSES, FOOD_COLORS

VALID_MODEL_TYPES = ("rf", "xgboost", "ensemble")

# Random Forest hyperparameters
RF_N_ESTIMATORS = 200
RF_MAX_DEPTH = None
RF_MIN_SAMPLES_LEAF = 2

# XGBoost hyperparameters
XGB_N_ESTIMATORS = 200
XGB_LR = 0.05
XGB_MAX_DEPTH = 6
XGB_SUBSAMPLE = 0.8
XGB_COLSAMPLE = 0.8


def train_rf(
    X_train: np.ndarray,
    y_train: np.ndarray,
    cv_folds: int = 5,
) -> tuple[RandomForestClassifier, dict]:
    """Fit a balanced Random Forest classifier. Returns (model, metadata)."""
    clf = RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS,
        max_depth=RF_MAX_DEPTH,
        min_samples_leaf=RF_MIN_SAMPLES_LEAF,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)
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
    rf: RandomForestClassifier,
    xgb: XGBClassifier,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    """Evaluate RF, XGBoost, and mean-proba ensemble on the held-out test set."""
    rf_pred = rf.predict(X_test).astype(int)
    xgb_pred = xgb.predict(X_test).astype(int)
    ens_pred = np.argmax(
        (rf.predict_proba(X_test) + xgb.predict_proba(X_test)) / 2.0, axis=1
    ).astype(int)

    def _metrics(pred: np.ndarray, label: str) -> dict:
        return {
            "label": label,
            "f1": round(float(f1_score(y_test, pred, average="macro")), 4),
            "accuracy": round(float(accuracy_score(y_test, pred)), 4),
            "predictions": pred.tolist(),
        }

    return {
        "rf": _metrics(rf_pred, "Random Forest"),
        "xgb": _metrics(xgb_pred, "XGBoost"),
        "ensemble": _metrics(ens_pred, "Ensemble (mean proba)"),
        "actuals": y_test.tolist(),
    }


def compute_shap_importance(xgb: XGBClassifier, X_test: np.ndarray) -> dict:
    """
    TreeExplainer SHAP on XGBoost (always used for SHAP regardless of model_type).
    Returns features sorted by descending mean |SHAP| averaged across all classes.
    """
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


def build_food_security_charts(
    eval_result: dict,
    shap_payload: dict,
    ndvi_df: pd.DataFrame,
    rain_df: pd.DataFrame,
    vci_mean: float,
    tci_mean: float,
    vhi_mean: float,
    model_type: str = "rf",
) -> dict:
    """Assemble frontend-ready chart payloads for the food security module."""
    _KEY = {"rf": "rf", "xgboost": "xgb", "ensemble": "ensemble"}
    result_key = _KEY.get(model_type, "rf")
    predictions = np.array(eval_result[result_key]["predictions"])

    n_total = len(predictions)
    counts = np.array([(predictions == i).sum() for i in range(3)], dtype=np.float64)
    risk_pct = (counts / n_total * 100).round(1)

    risk_dist = {
        "labels": FOOD_CLASSES,
        "data": risk_pct.tolist(),
        "colors": FOOD_COLORS,
    }

    # Time series: NDVI + monthly rainfall aligned to NDVI index
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
    if not rain_df.empty and ts_labels:
        aligned_rain = rain_df["rain_mm"].reindex(ts_labels).round(1).tolist()
        ts_datasets.append(
            {
                "label": "Monthly rain (mm)",
                "data": aligned_rain,
                "color": "#2980B9",
            }
        )

    return {
        "riskDist": risk_dist,
        "timeSeries": {"labels": ts_labels, "datasets": ts_datasets},
        "shap": shap_payload,
        "indices": {
            "vci_mean": round(vci_mean, 1),
            "tci_mean": round(tci_mean, 1),
            "vhi_mean": round(vhi_mean, 1),
        },
        "model_performance": {
            "rf": {
                "f1": eval_result["rf"]["f1"],
                "accuracy": eval_result["rf"]["accuracy"],
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


class FoodSecurityModel:
    """
    Orchestrates the full ML pipeline for a single food security analysis.
    Always trains Random Forest + XGBoost. config['model_type'] selects which
    predictions drive the primary risk distribution:
      "rf"       — Random Forest (default)
      "xgboost"  — XGBoost
      "ensemble" — mean softmax probabilities of RF + XGBoost
    Trained models and scaler are stored on self for use by cog_export.
    """

    def __init__(self) -> None:
        self.rf: Optional[RandomForestClassifier] = None
        self.xgb: Optional[XGBClassifier] = None
        self.scaler: Optional[StandardScaler] = None

    def predict(
        self,
        df: pd.DataFrame,
        ndvi_df: pd.DataFrame | None = None,
        rain_df: pd.DataFrame | None = None,
        config: dict | None = None,
    ) -> dict:
        """
        Parameters
        ----------
        df       : DataFrame with FEATURE_COLS + ['food_score', 'label']
        ndvi_df  : monthly NDVI DataFrame (date index, 'ndvi' column)
        rain_df  : monthly rainfall DataFrame (date index, 'rain_mm' column)
        config   : optional dict; reads 'model_type' (default 'rf')

        Returns
        -------
        dict with keys: stats, charts
        """
        cfg = config or {}
        model_type = cfg.get("model_type", "rf")
        if model_type not in VALID_MODEL_TYPES:
            raise ValueError(
                f"model_type must be one of {VALID_MODEL_TYPES}, got '{model_type}'"
            )

        _ndvi = ndvi_df if ndvi_df is not None else pd.DataFrame()
        _rain = rain_df if rain_df is not None else pd.DataFrame()

        X = (
            df[FEATURE_COLS]
            .fillna(df[FEATURE_COLS].median())
            .to_numpy(dtype=np.float64)
        )
        y = df["label"].to_numpy(dtype=np.intp)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=42, stratify=y
        )
        self.scaler = StandardScaler()
        X_train_s = self.scaler.fit_transform(X_train)
        X_test_s = self.scaler.transform(X_test)

        # Train RF and XGBoost concurrently when the Dask cluster is running,
        # otherwise fall back to sequential execution (e.g. during unit tests).
        from climate_change.core.dask_engine import DaskEngine

        client = DaskEngine.get_client_if_running()
        if client is not None:
            f_rf = client.submit(train_rf, X_train_s, y_train, pure=False)
            f_xgb = client.submit(train_xgb, X_train_s, y_train, pure=False)
            (self.rf, rf_meta), (self.xgb, xgb_meta) = cast(
                list, client.gather([f_rf, f_xgb])
            )
        else:
            self.rf, rf_meta = train_rf(X_train_s, y_train)
            self.xgb, xgb_meta = train_xgb(X_train_s, y_train)

        assert self.rf is not None
        assert self.xgb is not None

        eval_result = evaluate_models(self.rf, self.xgb, X_test_s, y_test)
        shap_payload = compute_shap_importance(self.xgb, X_test_s)

        # Compute VHI summary scalars from the feature DataFrame
        vci_mean = float(df["vci"].mean()) if "vci" in df.columns else 0.0
        tci_mean = float(df["tci"].mean()) if "tci" in df.columns else 0.0
        vhi_mean = 0.5 * vci_mean + 0.5 * tci_mean

        charts = build_food_security_charts(
            eval_result,
            shap_payload,
            _ndvi,
            _rain,
            vci_mean,
            tci_mean,
            vhi_mean,
            model_type,
        )

        # Risk distribution on all pixels using selected model
        X_all_s = self.scaler.transform(
            df[FEATURE_COLS]
            .fillna(df[FEATURE_COLS].median())
            .to_numpy(dtype=np.float64)
        )
        _KEY = {"rf": "rf", "xgboost": "xgb", "ensemble": "ensemble"}
        result_key = _KEY.get(model_type, "rf")

        if model_type == "rf":
            all_preds = self.rf.predict(X_all_s).astype(int)
        elif model_type == "xgboost":
            all_preds = self.xgb.predict(X_all_s).astype(int)
        else:
            proba = (
                self.rf.predict_proba(X_all_s) + self.xgb.predict_proba(X_all_s)
            ) / 2.0
            all_preds = np.argmax(proba, axis=1).astype(int)

        n_total = len(all_preds)
        counts = np.array([(all_preds == i).sum() for i in range(3)], dtype=np.float64)
        risk_pct = (counts / n_total * 100).round(1)
        high_risk_pct = float(risk_pct[2])

        stats = {
            "model_type": model_type,
            "n_pixels_sampled": int(len(df)),
            "rf_cv_f1": round(rf_meta["cv_f1_mean"], 4),
            "rf_f1": eval_result["rf"]["f1"],
            "rf_accuracy": eval_result["rf"]["accuracy"],
            "xgb_cv_f1": round(xgb_meta["cv_f1_mean"], 4),
            "xgb_f1": eval_result["xgb"]["f1"],
            "xgb_accuracy": eval_result["xgb"]["accuracy"],
            "ensemble_f1": eval_result["ensemble"]["f1"],
            "selected_f1": eval_result[result_key]["f1"],
            "high_risk_pct": round(high_risk_pct, 1),
            "top_driver": shap_payload["features"][0],
            "vci_mean": round(vci_mean, 1),
            "tci_mean": round(tci_mean, 1),
            "vhi_mean": round(vhi_mean, 1),
        }

        _FOOD_CLASS_NAMES = ["Low Risk", "Medium Risk", "High Risk"]
        if "lon" in df.columns and "lat" in df.columns:
            _sample_points = [
                {
                    "lon": round(float(df["lon"].iat[i]), 5),
                    "lat": round(float(df["lat"].iat[i]), 5),
                    "risk_class": _FOOD_CLASS_NAMES[int(all_preds[i])],
                }
                for i in range(len(df))
            ]
        else:
            _sample_points = []
        return {"stats": stats, "charts": charts, "_sample_points": _sample_points}
