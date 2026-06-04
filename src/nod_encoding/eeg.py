from __future__ import annotations

from pathlib import Path

import h5py
import mne
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from .bids_utils import list_eeg_files, load_events_with_stimuli
from .metrics import pearsonr_columns, r2_columns


def read_eeg(path: str | Path):
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".vhdr":
        return mne.io.read_raw_brainvision(path, preload=True, verbose="ERROR")
    if suffix == ".edf":
        return mne.io.read_raw_edf(path, preload=True, verbose="ERROR")
    if suffix == ".bdf":
        return mne.io.read_raw_bdf(path, preload=True, verbose="ERROR")
    if suffix == ".set":
        return mne.io.read_raw_eeglab(path, preload=True, verbose="ERROR")
    if suffix == ".fif":
        return mne.io.read_raw_fif(path, preload=True, verbose="ERROR")
    raise ValueError(f"Unsupported EEG file: {path}")


def load_feature_table(feature_h5: str | Path, layer: str) -> pd.DataFrame:
    with h5py.File(feature_h5, "r") as h5:
        keys = [k.decode("utf-8") if isinstance(k, bytes) else str(k) for k in h5["stimulus_key"][:]]
        x = h5[layer][:]
    df = pd.DataFrame(x)
    df.insert(0, "stimulus_key", keys)
    return df.groupby("stimulus_key", as_index=False).mean()


def make_lagged_design(X_events: np.ndarray, onset_samples: np.ndarray, n_times: int, sfreq: float, tmin: float, tmax: float):
    lags = np.arange(int(round(tmin * sfreq)), int(round(tmax * sfreq)) + 1, dtype=int)
    p = X_events.shape[1]
    design = np.zeros((n_times, p * len(lags)), dtype=np.float32)
    for ev, onset in zip(X_events, onset_samples):
        for j, lag in enumerate(lags):
            t = onset + lag
            if 0 <= t < n_times:
                design[t, j * p:(j + 1) * p] += ev
    return design, lags


def make_eeg_trf_matrix(
    bids_root: str | Path,
    features_h5: str | Path,
    layer: str = "IT",
    stimulus_column: str | None = None,
    tmin: float = -0.1,
    tmax: float = 0.6,
    decim: int = 4,
):
    events = load_events_with_stimuli(bids_root, stimulus_column)
    feat = load_feature_table(features_h5, layer)
    merged = events.merge(feat, on="stimulus_key", how="inner")
    feature_cols = [c for c in merged.columns if isinstance(c, int)]
    if not feature_cols:
        feature_cols = [c for c in merged.columns if str(c).isdigit()]
    eeg_files = list_eeg_files(bids_root)
    if not eeg_files:
        raise FileNotFoundError("No EEG files found under BIDS root.")
    raw = read_eeg(eeg_files[0])
    raw.pick_types(eeg=True, meg=False, eog=False, stim=False, exclude="bads")
    raw.filter(0.1, 40.0, verbose="ERROR")
    if decim > 1:
        raw.resample(raw.info["sfreq"] / decim, verbose="ERROR")
    data = raw.get_data().T.astype(np.float32)
    sfreq = float(raw.info["sfreq"])
    if "onset" not in merged.columns:
        raise ValueError("events.tsv must contain onset for EEG alignment.")
    onset_samples = np.rint(merged["onset"].to_numpy(dtype=float) * sfreq).astype(int)
    X_events = merged[feature_cols].to_numpy(dtype=np.float32)
    X_lag, lags = make_lagged_design(X_events, onset_samples, data.shape[0], sfreq, tmin, tmax)
    valid = np.any(X_lag != 0, axis=1)
    return X_lag[valid], data[valid], lags, sfreq, raw.ch_names


def cross_validated_trf(X: np.ndarray, Y: np.ndarray, alpha: float, n_splits: int = 5):
    kf = KFold(n_splits=min(n_splits, len(X)), shuffle=True, random_state=42)
    y_pred = np.zeros_like(Y, dtype=np.float32)
    for train, test in kf.split(X):
        sx = StandardScaler(with_mean=False).fit(X[train])
        sy = StandardScaler().fit(Y[train])
        model = Ridge(alpha=alpha, fit_intercept=True)
        model.fit(sx.transform(X[train]), sy.transform(Y[train]))
        y_pred[test] = sy.inverse_transform(model.predict(sx.transform(X[test]))).astype(np.float32)
    return pearsonr_columns(Y, y_pred), r2_columns(Y, y_pred), y_pred
