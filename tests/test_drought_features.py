"""Tests for drought/features.py — build_features, DroughtSequenceDataset, prepare_datasets."""

import numpy as np
import pandas as pd
import pytest
import torch

from climate_change.drought.features import (
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


# ── prepare_datasets: adaptive holdout ──────────────────────────────────────────


def _make_feat_df(n_rows: int) -> pd.DataFrame:
    """A feat_df shaped like build_features' output (INPUT_COLS + CDI), with
    no warm-up NaNs to drop — exactly n_rows usable monthly rows."""
    rng = np.random.default_rng(0)
    idx = pd.date_range("2010-01-01", periods=n_rows, freq="MS")
    data = {col: rng.uniform(0.5, 1.3, n_rows) for col in INPUT_COLS}
    data["CDI"] = rng.uniform(0.5, 1.3, n_rows)
    return pd.DataFrame(data, index=idx)


class TestPrepareDatasetsAdaptiveHoldout:
    def test_shrinks_holdout_instead_of_producing_negative_length_split(self):
        """Borena-shaped scenario: ~60 usable months (a 6-year date range
        after the 12-month lag-feature warm-up is dropped) against the flat
        48-month default holdout used to leave an 8-12 month train split —
        too short for DroughtSequenceDataset's seq_len(12) + horizon(6), so
        __len__() went negative and torch's DataLoader crashed."""
        feat_df = _make_feat_df(60)
        train_ds, test_ds, _, _ = prepare_datasets(feat_df)
        assert len(train_ds) >= 1
        assert len(test_ds) >= 1

    def test_does_not_shrink_holdout_when_data_is_ample(self):
        """With plenty of history (the original 10-year fixture scale), the
        default 48-month holdout fits comfortably and must be used as-is —
        the adaptive shrink should never kick in unnecessarily."""
        feat_df = _make_feat_df(120)
        train_ds, test_ds, _, _ = prepare_datasets(feat_df)
        assert len(test_ds) == 48 - SEQ_LEN - FORECAST_H + 1
        assert len(train_ds) == (120 - 48) - SEQ_LEN - FORECAST_H + 1

    def test_raises_recognized_error_when_data_is_too_short_for_any_split(self):
        """Fewer than 2*(seq_len+horizon) rows can't support even one real
        training sequence and one real test sequence — raise a clear,
        already-recognized error instead of letting it reach torch and
        crash as an opaque __len__ ValueError."""
        feat_df = _make_feat_df(10)
        with pytest.raises(ValueError, match="No data was available for:"):
            prepare_datasets(feat_df)

    def test_boundary_just_enough_rows_does_not_raise(self):
        min_rows_per_split = SEQ_LEN + FORECAST_H
        feat_df = _make_feat_df(2 * min_rows_per_split)
        train_ds, test_ds, _, _ = prepare_datasets(feat_df)
        assert len(train_ds) >= 1
        assert len(test_ds) >= 1
