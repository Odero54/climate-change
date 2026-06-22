import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

# LSTM hyperparameters
SEQ_LEN = 12  # months of look-back fed to the LSTM
FORECAST_H = 6  # months ahead to predict
HOLDOUT_MONTHS = 48  # test set size

INPUT_COLS = [
    "PDI",
    "TDI",
    "VDI",
    "CDI",
    "CDI_lag1",
    "CDI_lag3",
    "CDI_lag6",
    "CDI_lag12",
    "CDI_roll3_mean",
    "CDI_roll6_std",
    "sin_month",
    "cos_month",
]


def build_features(df: pd.DataFrame, lags: tuple = (1, 3, 6, 12)) -> pd.DataFrame:
    """
    Enrich CDI DataFrame with lag features, rolling statistics, and seasonal
    sine/cosine encodings. Rows with NaN (warm-up period) are dropped.
    """
    feat = df[["PDI", "TDI", "VDI", "CDI"]].copy()
    for lag in lags:
        feat[f"CDI_lag{lag}"] = feat["CDI"].shift(lag)
    feat["CDI_roll3_mean"] = feat["CDI"].rolling(3).mean()
    feat["CDI_roll6_std"] = feat["CDI"].rolling(6).std()
    months = pd.DatetimeIndex(feat.index).month
    feat["sin_month"] = np.sin(2 * np.pi * months / 12)
    feat["cos_month"] = np.cos(2 * np.pi * months / 12)
    return feat.dropna()


class DroughtSequenceDataset(Dataset):
    """
    PyTorch dataset yielding (input_window, target_sequence) pairs.
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        seq_len: int = SEQ_LEN,
        horizon: int = FORECAST_H,
    ):
        self.X = X
        self.y = y
        self.seq_len = seq_len
        self.horizon = horizon

    def __len__(self) -> int:
        return len(self.X) - self.seq_len - self.horizon + 1

    def __getitem__(self, i: int):
        x = torch.tensor(self.X[i : i + self.seq_len], dtype=torch.float32)
        t = torch.tensor(
            self.y[i + self.seq_len : i + self.seq_len + self.horizon],
            dtype=torch.float32,
        )
        return x, t


def prepare_datasets(
    feat_df: pd.DataFrame,
    seq_len: int = SEQ_LEN,
    horizon: int = FORECAST_H,
    holdout_months: int = HOLDOUT_MONTHS,
) -> tuple[
    DroughtSequenceDataset, DroughtSequenceDataset, StandardScaler, torch.Tensor
]:
    """
    Scale features, split train / test, and package into PyTorch datasets.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(feat_df[INPUT_COLS])
    y = feat_df["CDI"].to_numpy(dtype=float)
    split = len(X_scaled) - holdout_months
    train_ds = DroughtSequenceDataset(X_scaled[:split], y[:split], seq_len, horizon)
    test_ds = DroughtSequenceDataset(X_scaled[split:], y[split:], seq_len, horizon)

    last_seq = torch.tensor(
        X_scaled[-seq_len:].reshape(1, seq_len, len(INPUT_COLS)),
        dtype=torch.float32,
    )
    return train_ds, test_ds, scaler, last_seq
