from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def compute_shap_importance(
    model: Any, X_test: np.ndarray | pd.DataFrame, feature_cols: list[str]
) -> dict:
    """
    TreeExplainer SHAP values for the given tree-based model, sorted by
    descending mean |SHAP|. Works across RandomForestClassifier, XGBoost
    Booster, LightGBM, and sklearn GradientBoostingClassifier alike.

    SHAP's output axis convention for a binary/multi-class classifier varies
    across both shap versions and estimator types — confirmed live (shap
    0.52.0): XGBoost Booster, LightGBM, and GradientBoostingClassifier each
    return a single (n_samples, n_features) array, but RandomForestClassifier
    returns (n_samples, n_features, n_classes) — features in the *middle*
    axis, not last. Older shap versions have also returned a list of one
    (n_samples, n_features) array per class. Rather than assume a fixed axis
    position (which silently produced wrong, class-indexed "feature"
    importances for RF), this finds whichever axis's length matches the
    known feature count and averages |SHAP| over every other axis —
    correct regardless of which of these shapes comes back.
    """
    import shap

    try:
        shap_vals = shap.TreeExplainer(model).shap_values(X_test)
    except Exception:
        # TreeExplainer's tree-traversal implementation doesn't support every
        # estimator/output-shape combination — confirmed live: sklearn's
        # GradientBoostingClassifier for a 3+-class problem is internally one
        # independent regressor per class, and shap.TreeExplainer explicitly
        # rejects that ("GradientBoostingClassifier is only supported for
        # binary classification right now!"). This only surfaced once SHAP
        # started following the actually-selected model instead of always
        # using XGBoost. Fall back to the general, model-agnostic Explainer
        # (permutation-based, using predict_proba) — slower, but works for
        # any classifier; only triggered for the shapes TreeExplainer can't
        # handle. A small fixed background sample keeps it from scaling with
        # the full test set size.
        if not hasattr(model, "predict_proba"):
            raise
        background = X_test[: min(50, len(X_test))]
        shap_vals = shap.Explainer(model.predict_proba, background)(X_test).values

    arr = np.array(shap_vals) if isinstance(shap_vals, list) else np.asarray(shap_vals)

    n_features = len(feature_cols)
    feature_axis = next(ax for ax, size in enumerate(arr.shape) if size == n_features)
    other_axes = tuple(ax for ax in range(arr.ndim) if ax != feature_axis)
    mean_abs = np.abs(arr).mean(axis=other_axes) if other_axes else np.abs(arr)

    rank_idx = np.argsort(mean_abs)[::-1]
    return {
        "features": [feature_cols[i] for i in rank_idx],
        "mean_abs_shap": mean_abs[rank_idx].round(4).tolist(),
    }
