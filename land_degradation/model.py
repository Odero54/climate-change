from __future__ import annotations

from typing import Optional, cast

import lightgbm as lgb
import numpy as np
import pandas as pd
import ruptures as rpt
import shap
from ruptures.utils import sanity_check
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

from .features import DEGRADATION_CLASSES, DEGRADATION_COLORS, FEATURE_COLS

VALID_MODEL_TYPES = ("rf", "lgbm", "ensemble")

RF_N_ESTIMATORS = 200
RF_MAX_DEPTH = None
RF_MIN_SAMPLES_LEAF = 4

LGBM_N_ESTIMATORS = 200
LGBM_LR = 0.05
LGBM_NUM_LEAVES = 63


def train_rf(
    X_train: np.ndarray,
    y_train: np.ndarray,
    cv_folds: int = 5,
) -> tuple[RandomForestClassifier, dict]:
    """Fit a balanced Random Forest and report CV weighted F1. Returns (model, metadata)."""
    rf = RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS,
        max_depth=RF_MAX_DEPTH,
        min_samples_leaf=RF_MIN_SAMPLES_LEAF,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    cv_f1 = cross_val_score(rf, X_train, y_train, cv=cv, scoring="f1_weighted")
    return rf, {"cv_f1_mean": float(cv_f1.mean()), "cv_f1_std": float(cv_f1.std())}


def train_lgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    cv_folds: int = 5,
) -> tuple[lgb.LGBMClassifier, dict]:
    """Fit a balanced LightGBM classifier and report CV weighted F1. Returns (model, metadata)."""
    clf = lgb.LGBMClassifier(
        n_estimators=LGBM_N_ESTIMATORS,
        learning_rate=LGBM_LR,
        num_leaves=LGBM_NUM_LEAVES,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
    )
    clf.fit(X_train, y_train)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    cv_f1 = cross_val_score(clf, X_train, y_train, cv=cv, scoring="f1_weighted")  # pyright: ignore[reportArgumentType]
    return clf, {"cv_f1_mean": float(cv_f1.mean()), "cv_f1_std": float(cv_f1.std())}


def evaluate_models(
    rf: RandomForestClassifier,
    lgbm: lgb.LGBMClassifier,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    """Evaluate RF, LightGBM, and majority-vote ensemble on the held-out test set."""
    rf_pred = np.asarray(rf.predict(X_test)).astype(int)
    lgbm_pred = np.asarray(lgbm.predict(X_test)).astype(int)
    ens_pred = ((rf_pred + lgbm_pred) >= 1).astype(int)

    def _metrics(pred: np.ndarray, label: str) -> dict:
        return {
            "label": label,
            "f1": round(float(f1_score(y_test, pred, average="weighted")), 4),
            "accuracy": round(float(accuracy_score(y_test, pred)), 4),
            "predictions": pred.tolist(),
        }

    return {
        "rf": _metrics(rf_pred, "Random Forest"),
        "lgbm": _metrics(lgbm_pred, "LightGBM"),
        "ensemble": _metrics(ens_pred, "Ensemble (majority vote)"),
        "actuals": y_test.tolist(),
    }


def compute_shap_importance(
    model: RandomForestClassifier | lgb.LGBMClassifier,
    X_test: np.ndarray,
) -> dict:
    """
    TreeExplainer SHAP values sorted by descending mean |SHAP|.
    For multi-output (RF), averages across classes.
    """
    X_df = pd.DataFrame(X_test, columns=FEATURE_COLS)
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X_df)

    arr = np.array(shap_vals) if isinstance(shap_vals, list) else shap_vals
    # arr may be (n_classes, n_samples, n_features) or (n_samples, n_features)
    mean_abs = np.abs(arr).mean(axis=tuple(range(arr.ndim - 1)))

    rank_idx = np.argsort(mean_abs)[::-1]
    return {
        "features": [FEATURE_COLS[i] for i in rank_idx],
        "mean_abs_shap": mean_abs[rank_idx].round(4).tolist(),
    }


def compute_ndvi_trend(ndvi_annual: pd.Series) -> dict:
    """
    OLS linear regression + Mann-Kendall test + Binseg RBF breakpoint detection
    on an annual NDVI series (index = integer years).
    Returns a flat dict of trend statistics for inclusion in the result payload.
    """
    # Drop NaN and align years to the *actual* non-missing entries so that
    # gaps mid-series (cloud cover, data outages) do not shift the year
    # axis relative to the value array.
    valid = ndvi_annual.dropna()
    years = valid.index.values.astype(float)
    vals = np.asarray(valid.values)

    _ols = stats.linregress(years, vals)
    ols_slope = cast(float, _ols[0])
    ols_rvalue = cast(float, _ols[2])
    ols_pvalue = cast(float, _ols[3])
    _mk = stats.kendalltau(years, vals)
    mk_tau = cast(float, _mk[0])
    mk_p = cast(float, _mk[1])

    # Binseg breakpoints — jump=1 ensures every year is a candidate
    signal = vals.reshape(-1, 1)
    n_obs = len(vals)
    min_size = 2
    jump = 1

    n_bkps = next(
        (k for k in range(3, 0, -1) if sanity_check(n_obs, k, jump, min_size)),
        1,
    )
    binseg = rpt.Binseg(model="rbf", min_size=min_size, jump=jump).fit(signal)
    bkps_raw = binseg.predict(n_bkps=n_bkps)
    # bkps_raw indices reference the *valid* (post-dropna) array, so map
    # back through valid.index (not the original full index).
    valid_idx = valid.index.tolist()
    bkp_years = [int(valid_idx[i - 1]) for i in bkps_raw[:-1]]

    return {
        "ndvi_trend_per_year": round(float(ols_slope), 5),
        "ndvi_trend_r2": round(float(ols_rvalue**2), 4),
        "ndvi_trend_p": round(float(ols_pvalue), 4),
        "mk_tau": round(float(mk_tau), 4),
        "mk_p": round(float(mk_p), 4),
        "mk_significant": bool(float(mk_p) < 0.05),
        "breakpoint_years": bkp_years,
        "breakpoint_year": bkp_years[0] if bkp_years else None,
    }


def build_degradation_charts(
    eval_result: dict,
    shap_payload: dict,
    ndvi_annual: pd.Series,
    trend_stats: dict,
    model_type: str = "lgbm",
    scale: int = 1000,
) -> dict:
    """Assemble frontend-ready chart payloads matching LandDegradationUseCase.run() schema."""
    _KEY = {"rf": "rf", "lgbm": "lgbm", "ensemble": "ensemble"}
    result_key = _KEY.get(model_type, "lgbm")
    predictions = np.array(eval_result[result_key]["predictions"])
    actuals = np.array(eval_result["actuals"])

    n_total = len(actuals)
    pixel_ha = (scale**2) / 10_000
    not_deg_cnt = int((predictions == 0).sum())
    deg_cnt = int((predictions == 1).sum())

    return {
        "riskDist": {
            "labels": DEGRADATION_CLASSES,
            "data": [
                round(not_deg_cnt / n_total * 100, 1),
                round(deg_cnt / n_total * 100, 1),
            ],
            "data_ha": [round(not_deg_cnt * pixel_ha, 1), round(deg_cnt * pixel_ha, 1)],
            "colors": DEGRADATION_COLORS,
        },
        "timeSeries": {
            "labels": ndvi_annual.index.tolist(),
            "datasets": [
                {
                    "label": "Annual NDVI",
                    "data": ndvi_annual.round(4).tolist(),
                    "color": "#27AE60",
                },
            ],
        },
        "shap": shap_payload,
        "trend": trend_stats,
        "model_performance": {
            "rf": {
                "f1": eval_result["rf"]["f1"],
                "accuracy": eval_result["rf"]["accuracy"],
            },
            "lgbm": {
                "f1": eval_result["lgbm"]["f1"],
                "accuracy": eval_result["lgbm"]["accuracy"],
            },
            "ensemble": {
                "f1": eval_result["ensemble"]["f1"],
                "accuracy": eval_result["ensemble"]["accuracy"],
            },
            "selected": model_type,
        },
    }


class LandDegradationModel:
    """
    Orchestrates the full ML pipeline for a single land degradation analysis.
    Always trains RF + LightGBM. config['model_type'] selects which predictions
    drive the primary risk distribution:
      "rf"       — Random Forest
      "lgbm"     — LightGBM (default)
      "ensemble" — majority vote of RF + LightGBM
    Trained models and scaler are stored on self for use by cog_export.
    """

    def __init__(self) -> None:
        self.rf: Optional[RandomForestClassifier] = None
        self.lgbm: Optional[lgb.LGBMClassifier] = None
        self.scaler: Optional[StandardScaler] = None

    def predict(
        self,
        df: pd.DataFrame,
        ndvi_annual: pd.Series,
        config: dict | None = None,
    ) -> dict:
        """
        Parameters
        ----------
        df          : DataFrame with FEATURE_COLS + ['deg_score', 'deg_class']
        ndvi_annual : Annual mean NDVI Series (index = int years)
        config      : optional dict; reads 'model_type' (default 'lgbm') and 'scale'

        Returns
        -------
        dict with keys: stats, charts
        """
        cfg = config or {}
        model_type = cfg.get("model_type", "lgbm")
        scale = int(cfg.get("scale", 1000))

        if model_type not in VALID_MODEL_TYPES:
            raise ValueError(
                f"model_type must be one of {VALID_MODEL_TYPES}, got '{model_type}'"
            )

        X = (
            df[FEATURE_COLS]
            .fillna(df[FEATURE_COLS].median())
            .to_numpy(dtype=np.float64)
        )
        y = df["deg_class"].to_numpy(dtype=np.intp)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        self.scaler = StandardScaler()
        X_train_s = self.scaler.fit_transform(X_train)
        X_test_s = self.scaler.transform(X_test)

        # Train RF and LightGBM concurrently when the Dask cluster is running,
        # otherwise fall back to sequential execution (e.g. during unit tests).
        from climate_change.core.dask_engine import DaskEngine

        client = DaskEngine.get_client_if_running()
        if client is not None:
            f_rf = client.submit(train_rf, X_train_s, y_train, pure=False)
            f_lgbm = client.submit(train_lgbm, X_train_s, y_train, pure=False)
            (self.rf, rf_meta), (self.lgbm, lgbm_meta) = cast(
                list, client.gather([f_rf, f_lgbm])
            )
        else:
            self.rf, rf_meta = train_rf(X_train_s, y_train)
            self.lgbm, lgbm_meta = train_lgbm(X_train_s, y_train)

        assert self.rf is not None
        assert self.lgbm is not None

        eval_result = evaluate_models(self.rf, self.lgbm, X_test_s, y_test)
        shap_model = self.rf if model_type == "rf" else self.lgbm
        shap_payload = compute_shap_importance(shap_model, X_test_s)
        trend_stats = compute_ndvi_trend(ndvi_annual)

        charts = build_degradation_charts(
            eval_result, shap_payload, ndvi_annual, trend_stats, model_type, scale
        )

        _KEY = {"rf": "rf", "lgbm": "lgbm", "ensemble": "ensemble"}
        result_key = _KEY.get(model_type, "lgbm")

        stats = {
            "model_type": model_type,
            "n_pixels_sampled": int(len(df)),
            "degraded_label_pct": round(float(y.mean() * 100), 1),
            "rf_cv_f1": round(rf_meta["cv_f1_mean"], 4),
            "rf_f1": eval_result["rf"]["f1"],
            "rf_accuracy": eval_result["rf"]["accuracy"],
            "lgbm_cv_f1": round(lgbm_meta["cv_f1_mean"], 4),
            "lgbm_f1": eval_result["lgbm"]["f1"],
            "lgbm_accuracy": eval_result["lgbm"]["accuracy"],
            "ensemble_f1": eval_result["ensemble"]["f1"],
            "selected_f1": eval_result[result_key]["f1"],
            "top_degradation_driver": shap_payload["features"][0],
            **trend_stats,
        }

        X_all_s = self.scaler.transform(
            df[FEATURE_COLS]
            .fillna(df[FEATURE_COLS].median())
            .to_numpy(dtype=np.float64)
        )
        if model_type == "rf":
            all_preds = np.asarray(self.rf.predict(X_all_s)).astype(int)
        elif model_type == "lgbm":
            all_preds = np.asarray(self.lgbm.predict(X_all_s)).astype(int)
        else:
            rf_preds = np.asarray(self.rf.predict(X_all_s)).astype(int)
            lgbm_preds = np.asarray(self.lgbm.predict(X_all_s)).astype(int)
            all_preds = ((rf_preds + lgbm_preds) >= 1).astype(int)

        _DEG_CLASS_NAMES = ["Not Degraded", "Degraded"]
        if "lon" in df.columns and "lat" in df.columns:
            _sample_points = [
                {
                    "lon": round(float(df["lon"].iat[i]), 5),
                    "lat": round(float(df["lat"].iat[i]), 5),
                    "risk_class": _DEG_CLASS_NAMES[int(all_preds[i])],
                }
                for i in range(len(df))
            ]
        else:
            _sample_points = []
        return {"stats": stats, "charts": charts, "_sample_points": _sample_points}
