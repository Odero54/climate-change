"""Tests for drought/model.py — DroughtLSTM, severity stats, KMeans, forecast, DroughtModel."""
import numpy as np
import pandas as pd
import pytest
import torch
import xarray as xr

from drought.features import (
    FORECAST_H,
    INPUT_COLS,
    SEQ_LEN,
    DroughtSequenceDataset,
    build_features,
    prepare_datasets,
)
from drought.model import (
    DroughtLSTM,
    DroughtModel,
    VALID_MODEL_TYPES,
    _temporally_fill_forecast,
    drought_severity_stats,
    evaluate_lstm,
    forecast_with_uncertainty,
    run_kmeans_typology,
    train_lstm,
)


# ── DroughtLSTM ───────────────────────────────────────────────────────────────

class TestDroughtLSTM:
    def test_forward_pass_shape(self):
        model = DroughtLSTM()
        x = torch.randn(4, SEQ_LEN, len(INPUT_COLS))
        out = model(x)
        assert out.shape == (4, FORECAST_H)

    def test_forward_pass_deterministic_in_eval(self):
        model = DroughtLSTM()
        model.eval()
        x = torch.randn(2, SEQ_LEN, len(INPUT_COLS))
        with torch.no_grad():
            o1 = model(x)
            o2 = model(x)
        assert torch.allclose(o1, o2)

    def test_custom_hidden_and_horizon(self):
        model = DroughtLSTM(n_features=5, hidden=16, layers=1, horizon=3)
        x = torch.randn(2, SEQ_LEN, 5)
        out = model(x)
        assert out.shape == (2, 3)


# ── train_lstm & evaluate_lstm ────────────────────────────────────────────────

class TestTrainAndEvaluateLSTM:
    @pytest.fixture()
    def small_datasets(self, cdi_dataframe):
        feat_df = build_features(cdi_dataframe)
        return prepare_datasets(feat_df, holdout_months=12)

    def test_train_returns_model_and_history(self, small_datasets):
        train_ds, test_ds, _, _ = small_datasets
        model, history = train_lstm(train_ds, test_ds)
        assert isinstance(model, DroughtLSTM)
        assert "train_losses" in history
        assert "val_losses" in history
        assert "best_val_mse" in history

    def test_evaluate_returns_metrics(self, small_datasets):
        train_ds, test_ds, _, _ = small_datasets
        model, _ = train_lstm(train_ds, test_ds)
        metrics = evaluate_lstm(model, test_ds)
        assert "mae" in metrics
        assert "rmse" in metrics
        assert metrics["mae"] >= 0
        assert metrics["rmse"] >= 0


# ── forecast_with_uncertainty ─────────────────────────────────────────────────

class TestForecastWithUncertainty:
    def test_keys_present(self, cdi_dataframe):
        feat_df = build_features(cdi_dataframe)
        _, _, _, last_seq = prepare_datasets(feat_df, holdout_months=12)
        model = DroughtLSTM()
        future = pd.date_range("2024-01-01", periods=FORECAST_H, freq="MS")
        result = forecast_with_uncertainty(model, last_seq, future, n_samples=10)
        for key in ("dates", "mean", "std", "ci_lower", "ci_upper"):
            assert key in result

    def test_forecast_length_matches_horizon(self, cdi_dataframe):
        feat_df = build_features(cdi_dataframe)
        _, _, _, last_seq = prepare_datasets(feat_df, holdout_months=12)
        model = DroughtLSTM()
        future = pd.date_range("2024-01-01", periods=FORECAST_H, freq="MS")
        result = forecast_with_uncertainty(model, last_seq, future, n_samples=5)
        assert len(result["mean"]) == FORECAST_H

    def test_ci_lower_lte_ci_upper(self, cdi_dataframe):
        feat_df = build_features(cdi_dataframe)
        _, _, _, last_seq = prepare_datasets(feat_df, holdout_months=12)
        model = DroughtLSTM()
        future = pd.date_range("2024-01-01", periods=FORECAST_H, freq="MS")
        result = forecast_with_uncertainty(model, last_seq, future, n_samples=5)
        for lo, hi in zip(result["ci_lower"], result["ci_upper"]):
            assert lo <= hi


# ── run_kmeans_typology ───────────────────────────────────────────────────────

def _make_cdi_dataset(n_time=5, n_lon=4, n_lat=3):
    rng = np.random.default_rng(42)
    data = rng.uniform(0.5, 1.3, (n_time, n_lon, n_lat))
    times = pd.date_range("2015", periods=n_time, freq="YS")
    lons = np.linspace(36.0, 37.0, n_lon)
    lats = np.linspace(-1.0, 0.0, n_lat)
    return xr.Dataset(
        {
            "CDI": (["time", "lon", "lat"], data),
            "PDI": (["time", "lon", "lat"], rng.uniform(0.5, 1.3, (n_time, n_lon, n_lat))),
            "TDI": (["time", "lon", "lat"], rng.uniform(0.5, 1.3, (n_time, n_lon, n_lat))),
            "VDI": (["time", "lon", "lat"], rng.uniform(0.5, 1.3, (n_time, n_lon, n_lat))),
        },
        coords={"time": times, "lon": lons, "lat": lats},
    )


class TestRunKmeansTypology:
    def test_returns_expected_keys(self):
        ds = _make_cdi_dataset()
        result = run_kmeans_typology(ds, n_clusters=2)
        for key in ("label_map", "lons", "lats", "clusters", "n_clusters"):
            assert key in result

    def test_cluster_count_bounded_by_n_clusters(self):
        ds = _make_cdi_dataset()
        result = run_kmeans_typology(ds, n_clusters=3)
        assert result["n_clusters"] <= 3

    def test_all_nan_pixels_returns_no_clusters(self):
        ds = _make_cdi_dataset()
        ds["CDI"][:] = np.nan
        result = run_kmeans_typology(ds, n_clusters=3)
        assert result["n_clusters"] == 0


# ── drought_severity_stats ────────────────────────────────────────────────────

class TestDroughtSeverityStats:
    def test_all_extreme_drought(self):
        arr = np.array([0.3, 0.4, 0.45])
        result = drought_severity_stats(arr)
        assert result["extreme_pct"] == 100.0

    def test_all_near_normal(self):
        arr = np.array([0.95, 1.0, 1.05])
        result = drought_severity_stats(arr)
        assert result["near_normal_pct"] == 100.0

    def test_percentages_sum_to_100(self):
        arr = np.linspace(0.3, 1.5, 100)
        result = drought_severity_stats(arr)
        pct_keys = [k for k in result if k.endswith("_pct")]
        total = sum(result[k] for k in pct_keys)
        assert abs(total - 100.0) < 0.5

    def test_empty_array_returns_zeros(self):
        result = drought_severity_stats(np.array([]))
        assert result["aoi_valid_pixel_count"] == 0
        assert result["latest_mean_cdi"] == 0.0

    def test_nan_values_are_ignored(self):
        arr = np.array([0.3, np.nan, 1.0])
        result = drought_severity_stats(arr)
        assert result["aoi_valid_pixel_count"] == 2


# ── _temporally_fill_forecast ─────────────────────────────────────────────────

class TestTemporallyFillForecast:
    def test_fills_nan_values(self):
        idx = pd.date_range("2024-01-01", periods=6, freq="MS")
        df = pd.DataFrame({"CDI": [1.0, np.nan, 0.8, np.nan, 1.1, 0.9]}, index=idx)
        result = _temporally_fill_forecast(df)
        assert not result["CDI"].isna().any()

    def test_passes_through_clean_data(self):
        idx = pd.date_range("2024-01-01", periods=4, freq="MS")
        df = pd.DataFrame({"CDI": [0.9, 1.0, 1.1, 0.8]}, index=idx)
        result = _temporally_fill_forecast(df)
        assert list(result["CDI"]) == [0.9, 1.0, 1.1, 0.8]


# ── DroughtModel ──────────────────────────────────────────────────────────────

class TestDroughtModel:
    def test_invalid_model_type_raises(self):
        model = DroughtModel()
        with pytest.raises(ValueError, match="model_type must be one of"):
            model.predict({}, config={"model_type": "bad_type"})

    def test_valid_model_types_defined(self):
        assert "lstm" in VALID_MODEL_TYPES
        assert "drought_monitoring" in VALID_MODEL_TYPES
