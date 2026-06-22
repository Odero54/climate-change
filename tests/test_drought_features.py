"""Tests for drought/features.py — build_features, DroughtSequenceDataset, prepare_datasets."""
import numpy as np
import pandas as pd
import pytest
import torch

from drought.features import (
    FORECAST_H,
    INPUT_COLS,
    SEQ_LEN,
    DroughtSequenceDataset,
    build_features,
    prepare_datasets,
)


# ── build_features ─────────────────────────────────────────────────────────────

class TestBuildFeatures:
    def test_returns_dataframe_with_expected_columns(self, cdi_dataframe):
        result = build_features(cdi_dataframe)
        for col in INPUT_COLS:
            assert col in result.columns, f"Missing column: {col}"

    def test_drops_nan_warmup_rows(self, cdi_dataframe):
        result = build_features(cdi_dataframe)
        assert not result.isnull().any().any()

    def test_shorter_than_input_due_to_warmup(self, cdi_dataframe):
        result = build_features(cdi_dataframe)
        assert len(result) < len(cdi_dataframe)

    def test_lag_columns_created(self, cdi_dataframe):
        result = build_features(cdi_dataframe, lags=(1, 3))
        assert "CDI_lag1" in result.columns
        assert "CDI_lag3" in result.columns

    def test_seasonal_encoding_bounded(self, cdi_dataframe):
        result = build_features(cdi_dataframe)
        assert (result["sin_month"].abs() <= 1.0).all()
        assert (result["cos_month"].abs() <= 1.0).all()

    def test_index_is_datetime(self, cdi_dataframe):
        result = build_features(cdi_dataframe)
        assert isinstance(result.index, pd.DatetimeIndex)


# ── DroughtSequenceDataset ─────────────────────────────────────────────────────

class TestDroughtSequenceDataset:
    def _make_dataset(self, n=100, seq_len=SEQ_LEN, horizon=FORECAST_H):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((n, len(INPUT_COLS))).astype(np.float32)
        y = rng.uniform(0.5, 1.3, n).astype(np.float64)
        return DroughtSequenceDataset(X, y, seq_len=seq_len, horizon=horizon)

    def test_len_correct(self):
        ds = self._make_dataset(n=100)
        expected = 100 - SEQ_LEN - FORECAST_H + 1
        assert len(ds) == expected

    def test_getitem_returns_tensors(self):
        ds = self._make_dataset()
        x, t = ds[0]
        assert isinstance(x, torch.Tensor)
        assert isinstance(t, torch.Tensor)

    def test_input_tensor_shape(self):
        ds = self._make_dataset()
        x, _ = ds[0]
        assert x.shape == (SEQ_LEN, len(INPUT_COLS))

    def test_target_tensor_shape(self):
        ds = self._make_dataset()
        _, t = ds[0]
        assert t.shape == (FORECAST_H,)

    def test_custom_seq_len_and_horizon(self):
        ds = self._make_dataset(n=50, seq_len=6, horizon=3)
        x, t = ds[0]
        assert x.shape[0] == 6
        assert t.shape[0] == 3


# ── prepare_datasets ──────────────────────────────────────────────────────────

class TestPrepareDatasets:
    def test_returns_four_elements(self, cdi_dataframe):
        feat_df = build_features(cdi_dataframe)
        result = prepare_datasets(feat_df, holdout_months=12)
        assert len(result) == 4

    def test_train_and_test_are_datasets(self, cdi_dataframe):
        feat_df = build_features(cdi_dataframe)
        train_ds, test_ds, scaler, last_seq = prepare_datasets(feat_df, holdout_months=12)
        assert isinstance(train_ds, DroughtSequenceDataset)
        assert isinstance(test_ds, DroughtSequenceDataset)

    def test_last_seq_shape(self, cdi_dataframe):
        feat_df = build_features(cdi_dataframe)
        _, _, _, last_seq = prepare_datasets(feat_df, holdout_months=12)
        assert last_seq.shape == (1, SEQ_LEN, len(INPUT_COLS))

    def test_scaler_has_mean(self, cdi_dataframe):
        feat_df = build_features(cdi_dataframe)
        _, _, scaler, _ = prepare_datasets(feat_df, holdout_months=12)
        assert hasattr(scaler, "mean_")
