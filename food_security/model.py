from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import Booster, DMatrix
from xgboost import train as xgb_train

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
    folds = _safe_cv_folds(y_train, cv_folds)
    if folds is None:
        return clf, {"cv_f1_mean": None, "cv_f1_std": None}
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    cv_f1 = cross_val_score(clf, X_train, y_train, cv=cv, scoring="f1_macro")
    return clf, {"cv_f1_mean": float(cv_f1.mean()), "cv_f1_std": float(cv_f1.std())}


XGB_PARAMS = {
    "objective": "multi:softprob",
    "num_class": len(FOOD_CLASSES),
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
    wrapper requires every class in FOOD_CLASSES to appear in y_train (it infers
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


def pad_rf_proba(rf: RandomForestClassifier, proba: np.ndarray) -> np.ndarray:
    """
    Expand RF's predict_proba output to the full FOOD_CLASSES width.

    RandomForestClassifier only emits a column per class it actually saw
    during fit (via rf.classes_), whereas train_xgb's Booster always outputs
    len(FOOD_CLASSES) columns regardless of what the AOI's data contained.
    Without padding, an AOI missing one risk class produces mismatched shapes
    (e.g. (n, 2) vs (n, 3)) the moment RF and XGBoost probabilities are
    combined for the ensemble.
    """
    n_classes = len(FOOD_CLASSES)
    if proba.shape[1] == n_classes:
        return proba
    padded = np.zeros((proba.shape[0], n_classes), dtype=proba.dtype)
    padded[:, rf.classes_.astype(int)] = proba
    return padded


def evaluate_models(
    rf: RandomForestClassifier | None,
    xgb: Booster | None,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    """
    Evaluate whichever of RF/XGBoost were actually trained (both, for an
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

    proba_rf = pad_rf_proba(rf, rf.predict_proba(X_test)) if rf is not None else None
    proba_xgb = xgb.predict(DMatrix(X_test)) if xgb is not None else None

    result: dict = {"actuals": y_test.tolist()}
    if proba_rf is not None:
        result["rf"] = _metrics(np.argmax(proba_rf, axis=1).astype(int), "Random Forest")
    if proba_xgb is not None:
        result["xgb"] = _metrics(np.argmax(proba_xgb, axis=1).astype(int), "XGBoost")
    if proba_rf is not None and proba_xgb is not None:
        ens_pred = np.argmax((proba_rf + proba_xgb) / 2.0, axis=1).astype(int)
        result["ensemble"] = _metrics(ens_pred, "Ensemble (mean proba)")
    return result


def compute_shap_importance(model: RandomForestClassifier | Booster, X_test: np.ndarray) -> dict:
    """TreeExplainer SHAP values for whichever model was actually selected/trained — see core.shap_utils for the axis-handling details."""
    from climate_change.core.shap_utils import compute_shap_importance as _compute_shap_importance

    X_df = pd.DataFrame(X_test, columns=FEATURE_COLS)
    return _compute_shap_importance(model, X_df, FEATURE_COLS)


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
            **{
                key: {"f1": eval_result[key]["f1"], "accuracy": eval_result[key]["accuracy"]}
                for key in ("rf", "xgb", "ensemble")
                if key in eval_result
            },
            "selected": model_type,
        },
    }


class FoodSecurityModel:
    """
    Orchestrates the full ML pipeline for a single food security analysis.
    config['model_type'] selects which model(s) are trained and which
    predictions drive the primary risk distribution:
      "rf"       — Random Forest only (default)
      "xgboost"  — XGBoost only
      "ensemble" — both, combined as mean softmax probabilities of RF + XGBoost
    Only the selected model type is trained. Trained models and scaler are
    stored on self for use by cog_export (whichever model wasn't needed
    stays None).
    """

    def __init__(self) -> None:
        self.rf: RandomForestClassifier | None = None
        self.xgb: Booster | None = None
        self.scaler: StandardScaler | None = None

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
            raise ValueError(f"model_type must be one of {VALID_MODEL_TYPES}, got '{model_type}'")

        _ndvi = ndvi_df if ndvi_df is not None else pd.DataFrame()
        _rain = rain_df if rain_df is not None else pd.DataFrame()

        X = df[FEATURE_COLS].fillna(df[FEATURE_COLS].median()).to_numpy(dtype=np.float64)
        y = df["label"].to_numpy(dtype=np.intp)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=42, stratify=_safe_stratify(y)
        )
        self.scaler = StandardScaler()
        X_train_s = self.scaler.fit_transform(X_train)
        X_test_s = self.scaler.transform(X_test)

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
            f_rf = client.submit(train_rf, X_train_s, y_train, pure=False)
            f_xgb = client.submit(train_xgb, X_train_s, y_train, pure=False)
            (self.rf, rf_meta), (self.xgb, xgb_meta) = cast(list, client.gather([f_rf, f_xgb]))
        else:
            if need_rf:
                self.rf, rf_meta = train_rf(X_train_s, y_train)
            if need_xgb:
                self.xgb, xgb_meta = train_xgb(X_train_s, y_train)

        eval_result = evaluate_models(self.rf, self.xgb, X_test_s, y_test)
        if model_type == "rf":
            assert self.rf is not None
            shap_payload = compute_shap_importance(self.rf, X_test_s)
        else:
            assert self.xgb is not None
            shap_payload = compute_shap_importance(self.xgb, X_test_s)

        # Compute VHI summary scalars from the feature DataFrame
        vci_mean = float(cast(float, df["vci"].mean())) if "vci" in df.columns else 0.0
        tci_mean = float(cast(float, df["tci"].mean())) if "tci" in df.columns else 0.0
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
            df[FEATURE_COLS].fillna(df[FEATURE_COLS].median()).to_numpy(dtype=np.float64)
        )
        _KEY = {"rf": "rf", "xgboost": "xgb", "ensemble": "ensemble"}
        result_key = _KEY.get(model_type, "rf")

        if model_type == "rf":
            assert self.rf is not None
            all_preds = self.rf.predict(X_all_s).astype(int)
        elif model_type == "xgboost":
            assert self.xgb is not None
            all_preds = np.argmax(self.xgb.predict(DMatrix(X_all_s)), axis=1).astype(int)
        else:
            assert self.rf is not None
            assert self.xgb is not None
            proba_rf = pad_rf_proba(self.rf, self.rf.predict_proba(X_all_s))
            proba = (proba_rf + self.xgb.predict(DMatrix(X_all_s))) / 2.0
            all_preds = np.argmax(proba, axis=1).astype(int)

        n_total = len(all_preds)
        counts = np.array([(all_preds == i).sum() for i in range(3)], dtype=np.float64)
        risk_pct = (counts / n_total * 100).round(1)
        high_risk_pct = float(risk_pct[2])

        stats = {
            "model_type": model_type,
            "n_pixels_sampled": int(len(df)),
            "rf_cv_f1": round(rf_meta["cv_f1_mean"], 4)
            if rf_meta["cv_f1_mean"] is not None
            else None,
            "rf_f1": eval_result["rf"]["f1"] if "rf" in eval_result else None,
            "rf_accuracy": eval_result["rf"]["accuracy"] if "rf" in eval_result else None,
            "xgb_cv_f1": round(xgb_meta["cv_f1_mean"], 4)
            if xgb_meta["cv_f1_mean"] is not None
            else None,
            "xgb_f1": eval_result["xgb"]["f1"] if "xgb" in eval_result else None,
            "xgb_accuracy": eval_result["xgb"]["accuracy"] if "xgb" in eval_result else None,
            "ensemble_f1": eval_result["ensemble"]["f1"] if "ensemble" in eval_result else None,
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
                    "lon": round(cast(float, df["lon"].iat[i]), 5),
                    "lat": round(cast(float, df["lat"].iat[i]), 5),
                    "risk_class": _FOOD_CLASS_NAMES[int(all_preds[i])],
                }
                for i in range(len(df))
            ]
        else:
            _sample_points = []
        return {"stats": stats, "charts": charts, "_sample_points": _sample_points}
