import copy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler as SScaler
from torch.utils.data import DataLoader

from .cdi_runner import build_drought_charts, compute_spatial_uncertainty
from .features import (
    FORECAST_H,
    INPUT_COLS,
    DroughtSequenceDataset,
    build_features,
    prepare_datasets,
)

# LSTM hyperparameters
HIDDEN = 64
LAYERS = 2
EPOCHS = 200
PATIENCE = 15
BATCH = 32
LR = 1e-3


class DroughtLSTM(nn.Module):
    """
    2-layer LSTM with dropout. Final hidden state → linear head → FORECAST_H outputs.
    All six forecast steps are predicted simultaneously (multi-output regression).
    """

    def __init__(
        self,
        n_features: int = len(INPUT_COLS),
        hidden: int = HIDDEN,
        layers: int = LAYERS,
        horizon: int = FORECAST_H,
    ):
        super().__init__()
        # PyTorch only applies recurrent dropout between stacked LSTM layers.
        # Passing a non-zero value for a single layer has no effect and emits a
        # warning, so disable it for that configuration.
        recurrent_dropout = 0.2 if layers > 1 else 0.0
        self.lstm = nn.LSTM(
            n_features,
            hidden,
            layers,
            batch_first=True,
            dropout=recurrent_dropout,
        )
        self.fc = nn.Linear(hidden, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])  # last time-step → horizon predictions


def train_lstm(
    train_ds: DroughtSequenceDataset,
    test_ds: DroughtSequenceDataset,
    device: torch.device | None = None,
) -> tuple[DroughtLSTM, dict]:
    """
    Train with early stopping (patience = PATIENCE epochs on val MSE).
    Best weights are restored before returning.
    """
    device = device or torch.device("cpu")

    model = DroughtLSTM().to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH)

    train_losses, val_losses = [], []
    best_val, patience_ctr = float("inf"), 0
    best_state = copy.deepcopy(model.state_dict())

    for epoch in range(EPOCHS):
        model.train()
        batch_losses = []
        for Xb, yb in train_loader:
            optimiser.zero_grad()
            loss = criterion(model(Xb.to(device)), yb.to(device))
            loss.backward()
            optimiser.step()
            batch_losses.append(loss.item())
        train_losses.append(float(np.mean(batch_losses)))

        model.eval()
        with torch.no_grad():
            vl = [criterion(model(Xb.to(device)), yb.to(device)).item() for Xb, yb in test_loader]
        val_losses.append(float(np.mean(vl)))

        if val_losses[-1] < best_val:
            best_val = val_losses[-1]
            best_state = copy.deepcopy(model.state_dict())
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                model.load_state_dict(best_state)
                return model, {
                    "train_losses": train_losses,
                    "val_losses": val_losses,
                    "best_val_mse": best_val,
                    "stopped_epoch": epoch + 1,
                }

    model.load_state_dict(best_state)
    return model, {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "best_val_mse": best_val,
        "stopped_epoch": EPOCHS,
    }


def evaluate_lstm(
    model: DroughtLSTM,
    test_ds: DroughtSequenceDataset,
    device: torch.device | None = None,
) -> dict:
    """
    1-step-ahead evaluation on the held-out test set.
    """
    device = device or torch.device("cpu")
    loader = DataLoader(test_ds, batch_size=BATCH)

    model.eval()
    preds_all, true_all = [], []
    with torch.no_grad():
        for Xb, yb in loader:
            preds_all.append(model(Xb.to(device)).cpu().numpy()[:, 0])
            true_all.append(yb[:, 0].numpy())

    preds = np.concatenate(preds_all)
    true = np.concatenate(true_all)

    return {
        "mae": float(mean_absolute_error(true, preds)),
        "rmse": float(np.sqrt(np.mean((preds - true) ** 2))),
        "predictions": preds.round(4).tolist(),
        "actuals": true.round(4).tolist(),
    }


def forecast_with_uncertainty(
    model: DroughtLSTM,
    last_seq: torch.Tensor,
    future_dates: pd.DatetimeIndex,
    n_samples: int = 500,
    device: torch.device | None = None,
) -> dict:
    """
    MC Dropout forecast: enable dropout at inference and run n_samples stochastic
    forward passes. The spread of the resulting distribution is the model's
    epistemic (model) uncertainty — it does NOT capture data or structural uncertainty.
    """
    device = device or torch.device("cpu")
    last_seq = last_seq.to(device)
    model.train()  # activate dropout layers for MC sampling
    mc_samples: list[np.ndarray] = []
    with torch.no_grad():
        for _ in range(n_samples):
            mc_samples.append(model(last_seq).cpu().numpy().flatten())
    model.eval()
    mc_preds: np.ndarray = np.array(mc_samples)  # (n_samples, FORECAST_H)

    ci_lo = np.percentile(mc_preds, 2.5, axis=0)
    ci_hi = np.percentile(mc_preds, 97.5, axis=0)
    ci_half = (ci_hi - ci_lo) / 2
    avg_half = float(ci_half.mean())
    return {
        "dates": future_dates.strftime("%Y-%m").tolist(),
        "mean": mc_preds.mean(axis=0).round(4).tolist(),
        "std": mc_preds.std(axis=0).round(4).tolist(),
        "ci_lower": ci_lo.round(4).tolist(),
        "ci_upper": ci_hi.round(4).tolist(),
        "ci_half_width": ci_half.round(4).tolist(),
        "ci_level": 0.95,
        "ci_summary": f"95% CI: ±{avg_half:.2f}",
        "n_samples": n_samples,
    }


def run_kmeans_typology(ds, n_clusters: int = 4) -> dict:
    """
    Cluster pixel CDI trajectories into drought typologies using KMeans.
    Clusters are ranked by ascending mean CDI (most drought-prone = cluster 0).
    """
    cdi_np = ds["CDI"].values  # (time, lon, lat)
    n_time, n_lon, n_lat = cdi_np.shape
    pixel_matrix = cdi_np.reshape(n_time, n_lon * n_lat).T  # (pixels, time)
    valid_mask = np.isfinite(pixel_matrix).any(axis=1)
    valid_pixels = pixel_matrix[valid_mask]
    if valid_pixels.size == 0:
        return {
            "label_map": np.full((n_lon, n_lat), -1, dtype=int).tolist(),
            "lons": ds.lon.values.tolist(),
            "lats": ds.lat.values.tolist(),
            "clusters": [],
            "n_clusters": 0,
            "nodata_pixel_count": int(pixel_matrix.shape[0]),
        }

    pixel_means = np.nanmean(valid_pixels, axis=1)
    global_mean = float(np.nanmean(valid_pixels))
    pixel_means = np.where(np.isfinite(pixel_means), pixel_means, global_mean)
    row_idx, col_idx = np.where(~np.isfinite(valid_pixels))
    valid_pixels = valid_pixels.copy()
    valid_pixels[row_idx, col_idx] = pixel_means[row_idx]

    effective_clusters = min(n_clusters, len(valid_pixels))
    ss = SScaler()
    pixel_scaled = ss.fit_transform(valid_pixels)

    km = KMeans(n_clusters=effective_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(pixel_scaled)

    # Rank clusters: 0 = most drought-prone (lowest mean CDI)
    cluster_means = {c: float(valid_pixels[labels == c].mean()) for c in range(effective_clusters)}
    rank_map = {
        old: new for new, old in enumerate(sorted(cluster_means, key=lambda c: cluster_means[c]))
    }
    labels_r = np.vectorize(rank_map.get)(labels)
    full_labels = np.full(pixel_matrix.shape[0], -1, dtype=int)
    full_labels[valid_mask] = labels_r
    label_map_r = full_labels.reshape(n_lon, n_lat)

    years = ds.time.dt.year.values.tolist()
    clusters = [
        {
            "cluster_id": int(c),
            "mean_cdi": round(cluster_means[{v: k for k, v in rank_map.items()}[c]], 4),
            "pixel_count": int((labels_r == c).sum()),
            "pixel_pct": round(float((labels_r == c).sum()) / len(labels_r) * 100, 1),
            "trajectory": valid_pixels[labels_r == c].mean(axis=0).round(4).tolist(),
            "years": years,
        }
        for c in range(effective_clusters)
    ]

    return {
        "label_map": label_map_r.tolist(),
        "lons": ds.lon.values.tolist(),
        "lats": ds.lat.values.tolist(),
        "clusters": clusters,
        "n_clusters": effective_clusters,
        "nodata_pixel_count": int((~valid_mask).sum()),
    }


def drought_severity_stats(latest_cdi: np.ndarray) -> dict:
    valid = latest_cdi[np.isfinite(latest_cdi)]
    if valid.size == 0:
        return {
            "latest_mean_cdi": 0.0,
            "aoi_valid_pixel_count": 0,
            "extreme_pct": 0.0,
            "severe_pct": 0.0,
            "moderate_pct": 0.0,
            "mild_pct": 0.0,
            "near_normal_pct": 0.0,
            "mild_wet_pct": 0.0,
            "moderately_wet_pct": 0.0,
            "very_wet_pct": 0.0,
        }
    return {
        "latest_mean_cdi": round(float(np.nanmean(valid)), 4),
        "aoi_valid_pixel_count": int(valid.size),
        "extreme_pct": round(float((valid < 0.50).mean() * 100), 1),
        "severe_pct": round(float(((valid >= 0.50) & (valid < 0.65)).mean() * 100), 1),
        "moderate_pct": round(float(((valid >= 0.65) & (valid < 0.80)).mean() * 100), 1),
        "mild_pct": round(float(((valid >= 0.80) & (valid < 0.90)).mean() * 100), 1),
        "near_normal_pct": round(float(((valid >= 0.90) & (valid < 1.10)).mean() * 100), 1),
        "mild_wet_pct": round(float(((valid >= 1.10) & (valid < 1.20)).mean() * 100), 1),
        "moderately_wet_pct": round(float(((valid >= 1.20) & (valid < 1.30)).mean() * 100), 1),
        "very_wet_pct": round(float((valid >= 1.30).mean() * 100), 1),
    }


VALID_MODEL_TYPES = ("lstm", "drought_monitoring")


def _temporally_fill_forecast(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if numeric_cols.empty:
        return df

    filled = df.copy()
    filled[numeric_cols] = (
        filled[numeric_cols]
        .replace([np.inf, -np.inf], np.nan)
        .interpolate(method="time", limit_direction="both")
        .bfill()
        .ffill()
    )
    for col in numeric_cols:
        if filled[col].isna().any():
            fallback = filled[col].median()
            if not np.isfinite(fallback):
                fallback = 0.0
            filled[col] = filled[col].fillna(float(fallback))
    return filled


class DroughtModel:
    """
    Orchestrates the drought ML pipeline.

    model_type (from config, default 'lstm'):
      'lstm'               — train DroughtLSTM + MC Dropout forecast (existing)
      'drought_monitoring' — use drought_monitoring package's built-in forecast

    Designed to be called from DroughtUseCase.run_model().
    """

    def predict(self, features: dict, config: dict | None = None) -> dict:
        model_type = (config or {}).get("model_type", "lstm")
        if model_type not in VALID_MODEL_TYPES:
            raise ValueError(f"model_type must be one of {VALID_MODEL_TYPES}, got '{model_type}'")

        if model_type == "drought_monitoring":
            return self._predict_drought_monitoring(features, config)
        return self._predict_lstm(features, config)

    # ── LSTM path (default) ───────────────────────────────────────────────────

    def _predict_lstm(self, features: dict, config: dict | None = None) -> dict:
        df = features["cdi_series"]
        ds = features["cdi_maps"]
        device = torch.device("cpu")

        feat_df = build_features(df)
        train_ds, test_ds, scaler, last_seq = prepare_datasets(feat_df)

        model, train_history = train_lstm(train_ds, test_ds, device)
        eval_metrics = evaluate_lstm(model, test_ds, device)

        future_dates = pd.date_range(feat_df.index[-1], periods=FORECAST_H + 1, freq="MS")[1:]
        forecast = forecast_with_uncertainty(model, last_seq, future_dates, device=device)

        typology = run_kmeans_typology(ds)
        uncertainty = compute_spatial_uncertainty(features)
        charts = build_drought_charts(features)
        charts["forecast"] = forecast
        charts["typology"] = typology
        charts["uncertainty"] = uncertainty
        charts["training"] = train_history

        latest_cdi = ds["CDI"].isel(time=-1).values
        severity_stats = drought_severity_stats(latest_cdi)
        stats = {
            "model_type": "lstm",
            "analysis_year": int(ds.time.dt.year.values[-1]),
            "mean_cdi": round(float(df["CDI"].mean()), 4),
            **severity_stats,
            "lstm_mae": eval_metrics["mae"],
            "lstm_rmse": eval_metrics["rmse"],
            "lstm_best_val_mse": train_history["best_val_mse"],
            "stopped_epoch": train_history["stopped_epoch"],
        }
        return {"stats": stats, "charts": charts}

    # ── drought_monitoring path ───────────────────────────────────────────────

    def _predict_drought_monitoring(self, features: dict, config: dict | None = None) -> dict:
        """Use the drought_monitoring package's built-in forecast function."""
        from drought_monitoring.forecast import forecast_all_statistical

        df = features["cdi_series"]
        ds = features["cdi_maps"]

        fc = forecast_all_statistical(
            features["precip"],
            features["temp"],
            features["ndvi"],
            n_months=FORECAST_H,
        )
        fc = _temporally_fill_forecast(fc)
        forecast = {
            "dates": pd.DatetimeIndex(fc.index).strftime("%Y-%m").tolist(),
            "mean": fc["CDI"].round(4).tolist(),
            "ci_lower": fc["CDI_lower"].round(4).tolist(),
            "ci_upper": fc["CDI_upper"].round(4).tolist(),
        }

        typology = run_kmeans_typology(ds)
        uncertainty = compute_spatial_uncertainty(features)
        charts = build_drought_charts(features)
        charts["forecast"] = forecast
        charts["typology"] = typology
        charts["uncertainty"] = uncertainty
        charts["training"] = {"model_type": "drought_monitoring"}

        latest_cdi = ds["CDI"].isel(time=-1).values
        severity_stats = drought_severity_stats(latest_cdi)
        stats = {
            "model_type": "drought_monitoring",
            "analysis_year": int(ds.time.dt.year.values[-1]),
            "mean_cdi": round(float(df["CDI"].mean()), 4),
            **severity_stats,
        }
        return {"stats": stats, "charts": charts}
