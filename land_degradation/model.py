from __future__ import annotations

from typing import Any, cast

import lightgbm as lgb
import numpy as np
import pandas as pd
import ruptures as rpt
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


class FeatureNamedLGBMClassifier(lgb.LGBMClassifier):
    """LightGBM classifier that preserves feature names for array inputs."""

    @staticmethod
    def _with_feature_names(X: Any) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X
        values = np.asarray(X)
        return pd.DataFrame(values, columns=FEATURE_COLS[: values.shape[1]])

    def predict(
        self,
        X: Any,
        raw_score: bool = False,
        start_iteration: int = 0,
        num_iteration: int | None = None,
        pred_leaf: bool = False,
        pred_contrib: bool = False,
        validate_features: bool = False,
        **kwargs: Any,
    ) -> Any:
        return super().predict(
            self._with_feature_names(X),
            raw_score=raw_score,
            start_iteration=start_iteration,
            num_iteration=num_iteration,
            pred_leaf=pred_leaf,
            pred_contrib=pred_contrib,
            validate_features=validate_features,
            **kwargs,
        )


def _safe_cv_folds(y_train: np.ndarray, cv_folds: int) -> int | None:
    """Clamp cv_folds to the rarest class's sample count; None if CV isn't meaningful.

    StratifiedKFold/cross_val_score require every class to have at least
    n_splits members. A small or skewed AOI/date-range can easily sample
    fewer degraded (or non-degraded) pixels than the default 5 folds, which
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
    folds = _safe_cv_folds(y_train, cv_folds)
    if folds is None:
        return rf, {"cv_f1_mean": None, "cv_f1_std": None}
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    cv_f1 = cross_val_score(rf, X_train, y_train, cv=cv, scoring="f1_weighted")
    return rf, {"cv_f1_mean": float(cv_f1.mean()), "cv_f1_std": float(cv_f1.std())}


def train_lgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    cv_folds: int = 5,
) -> tuple[lgb.LGBMClassifier, dict]:
    """Fit a balanced LightGBM classifier and report CV weighted F1. Returns (model, metadata)."""
    clf = FeatureNamedLGBMClassifier(
        n_estimators=LGBM_N_ESTIMATORS,
        learning_rate=LGBM_LR,
        num_leaves=LGBM_NUM_LEAVES,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
    )
    X_train_df = pd.DataFrame(X_train, columns=FEATURE_COLS[: X_train.shape[1]])
    clf.fit(X_train_df, y_train)
    folds = _safe_cv_folds(y_train, cv_folds)
    if folds is None:
        return clf, {"cv_f1_mean": None, "cv_f1_std": None}
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    cv_f1 = cross_val_score(clf, X_train_df, y_train, cv=cv, scoring="f1_weighted")  # pyright: ignore[reportArgumentType]
    return clf, {"cv_f1_mean": float(cv_f1.mean()), "cv_f1_std": float(cv_f1.std())}


def evaluate_models(
    rf: RandomForestClassifier | None,
    lgbm: lgb.LGBMClassifier | None,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    """
    Evaluate whichever of RF/LightGBM were actually trained (both, for an
    "ensemble" selection; exactly one otherwise) on the held-out test set,
    plus their majority-vote ensemble when both are present.
    """

    def _metrics(pred: np.ndarray, label: str) -> dict:
        return {
            "label": label,
            "f1": round(float(f1_score(y_test, pred, average="weighted")), 4),
            "accuracy": round(float(accuracy_score(y_test, pred)), 4),
            "predictions": pred.tolist(),
        }

    rf_pred = np.asarray(rf.predict(X_test)).astype(int) if rf is not None else None
    lgbm_pred = np.asarray(lgbm.predict(X_test)).astype(int) if lgbm is not None else None

    result: dict = {"actuals": y_test.tolist()}
    if rf_pred is not None:
        result["rf"] = _metrics(rf_pred, "Random Forest")
    if lgbm_pred is not None:
        result["lgbm"] = _metrics(lgbm_pred, "LightGBM")
    if rf_pred is not None and lgbm_pred is not None:
        ens_pred = ((rf_pred + lgbm_pred) >= 1).astype(int)
        result["ensemble"] = _metrics(ens_pred, "Ensemble (majority vote)")
    return result


def compute_shap_importance(
    model: RandomForestClassifier | lgb.LGBMClassifier,
    X_test: np.ndarray,
) -> dict:
    """TreeExplainer SHAP values for whichever model was actually selected/trained — see core.shap_utils for the axis-handling details."""
    from climate_change.core.shap_utils import compute_shap_importance as _compute_shap_importance

    X_df = pd.DataFrame(X_test, columns=FEATURE_COLS)
    return _compute_shap_importance(model, X_df, FEATURE_COLS)


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

    # A short/unusual date range (or an AOI where most years drop out via
    # dropna() above due to cloud cover) can leave fewer than 2 valid annual
    # points. linregress/kendalltau silently return NaN rather than raising
    # for <2 points, which would otherwise poison the JSON response — return
    # None for the undefined stats instead.
    if len(years) < 2:
        return {
            "ndvi_trend_per_year": None,
            "ndvi_trend_r2": None,
            "ndvi_trend_p": None,
            "mk_tau": None,
            "mk_p": None,
            "mk_significant": False,
            "breakpoint_years": [],
            "breakpoint_year": None,
        }

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

    # sanity_check(n_obs, k, jump, min_size) is False for every k when n_obs
    # is too small for even 1 breakpoint (Binseg needs n_obs >= (k+1)*min_size).
    # The `next(..., 1)` fallback below must be 0 in that case — falling back
    # to 1 regardless would call binseg.predict(n_bkps=1) on a segmentation
    # sanity_check already rejected, raising ruptures.BadSegmentationParameters.
    n_bkps = next(
        (k for k in range(3, 0, -1) if sanity_check(n_obs, k, jump, min_size)),
        0,
    )
    if n_bkps == 0:
        bkp_years: list[int] = []
    else:
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
            **{
                key: {"f1": eval_result[key]["f1"], "accuracy": eval_result[key]["accuracy"]}
                for key in ("rf", "lgbm", "ensemble")
                if key in eval_result
            },
            "selected": model_type,
        },
    }


class LandDegradationModel:
    """
    Orchestrates the full ML pipeline for a single land degradation analysis.
    config['model_type'] selects which model(s) are trained and which
    predictions drive the primary risk distribution:
      "rf"       — Random Forest only
      "lgbm"     — LightGBM only (default)
      "ensemble" — both, combined as majority vote of RF + LightGBM
    Only the selected model type is trained. Trained models and scaler are
    stored on self for use by cog_export (whichever model wasn't needed
    stays None).
    """

    def __init__(self) -> None:
        self.rf: RandomForestClassifier | None = None
        self.lgbm: lgb.LGBMClassifier | None = None
        self.scaler: StandardScaler | None = None

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
            raise ValueError(f"model_type must be one of {VALID_MODEL_TYPES}, got '{model_type}'")

        X = df[FEATURE_COLS].fillna(df[FEATURE_COLS].median()).to_numpy(dtype=np.float64)
        y = df["deg_class"].to_numpy(dtype=np.intp)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=_safe_stratify(y)
        )
        self.scaler = StandardScaler()
        X_train_s = self.scaler.fit_transform(X_train)
        X_test_s = self.scaler.transform(X_test)

        # Only train what this selection needs — "ensemble" needs both, "rf"
        # and "lgbm" need only the one selected. When both are needed, train
        # them concurrently on the Dask cluster if it's running, otherwise
        # fall back to sequential execution (e.g. during unit tests).
        from climate_change.core.dask_engine import DaskEngine

        need_rf = model_type in ("rf", "ensemble")
        need_lgbm = model_type in ("lgbm", "ensemble")
        rf_meta: dict = {"cv_f1_mean": None, "cv_f1_std": None}
        lgbm_meta: dict = {"cv_f1_mean": None, "cv_f1_std": None}

        client = DaskEngine.get_client_if_running() if (need_rf and need_lgbm) else None
        if client is not None:
            f_rf = client.submit(train_rf, X_train_s, y_train, pure=False)
            f_lgbm = client.submit(train_lgbm, X_train_s, y_train, pure=False)
            (self.rf, rf_meta), (self.lgbm, lgbm_meta) = cast(list, client.gather([f_rf, f_lgbm]))
        else:
            if need_rf:
                self.rf, rf_meta = train_rf(X_train_s, y_train)
            if need_lgbm:
                self.lgbm, lgbm_meta = train_lgbm(X_train_s, y_train)

        eval_result = evaluate_models(self.rf, self.lgbm, X_test_s, y_test)
        if model_type == "rf":
            assert self.rf is not None
            shap_payload = compute_shap_importance(self.rf, X_test_s)
        else:
            assert self.lgbm is not None
            shap_payload = compute_shap_importance(self.lgbm, X_test_s)
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
            "rf_cv_f1": round(rf_meta["cv_f1_mean"], 4)
            if rf_meta["cv_f1_mean"] is not None
            else None,
            "rf_f1": eval_result["rf"]["f1"] if "rf" in eval_result else None,
            "rf_accuracy": eval_result["rf"]["accuracy"] if "rf" in eval_result else None,
            "lgbm_cv_f1": round(lgbm_meta["cv_f1_mean"], 4)
            if lgbm_meta["cv_f1_mean"] is not None
            else None,
            "lgbm_f1": eval_result["lgbm"]["f1"] if "lgbm" in eval_result else None,
            "lgbm_accuracy": eval_result["lgbm"]["accuracy"] if "lgbm" in eval_result else None,
            "ensemble_f1": eval_result["ensemble"]["f1"] if "ensemble" in eval_result else None,
            "selected_f1": eval_result[result_key]["f1"],
            "top_degradation_driver": shap_payload["features"][0],
            **trend_stats,
        }

        X_all_s = self.scaler.transform(
            df[FEATURE_COLS].fillna(df[FEATURE_COLS].median()).to_numpy(dtype=np.float64)
        )
        if model_type == "rf":
            assert self.rf is not None
            all_preds = np.asarray(self.rf.predict(X_all_s)).astype(int)
        elif model_type == "lgbm":
            assert self.lgbm is not None
            all_preds = np.asarray(self.lgbm.predict(X_all_s)).astype(int)
        else:
            assert self.rf is not None
            assert self.lgbm is not None
            rf_preds = np.asarray(self.rf.predict(X_all_s)).astype(int)
            lgbm_preds = np.asarray(self.lgbm.predict(X_all_s)).astype(int)
            all_preds = ((rf_preds + lgbm_preds) >= 1).astype(int)

        _DEG_CLASS_NAMES = ["Not Degraded", "Degraded"]
        if "lon" in df.columns and "lat" in df.columns:
            _sample_points = [
                {
                    "lon": round(cast(float, df["lon"].iat[i]), 5),
                    "lat": round(cast(float, df["lat"].iat[i]), 5),
                    "risk_class": _DEG_CLASS_NAMES[int(all_preds[i])],
                }
                for i in range(len(df))
            ]
        else:
            _sample_points = []
        return {"stats": stats, "charts": charts, "_sample_points": _sample_points}
