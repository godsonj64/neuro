from __future__ import annotations

from pathlib import Path

import h5py
import nibabel as nib
import numpy as np
import pandas as pd
from nilearn.masking import compute_epi_mask, apply_mask
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

from .bids_utils import list_fmri_files, load_events_with_stimuli
from .metrics import pearsonr_columns, r2_columns


def load_feature_table(feature_h5: str | Path, layer: str) -> pd.DataFrame:
    with h5py.File(feature_h5, "r") as h5:
        keys = [k.decode("utf-8") if isinstance(k, bytes) else str(k) for k in h5["stimulus_key"][:]]
        x = h5[layer][:]
    df = pd.DataFrame(x)
    df.insert(0, "stimulus_key", keys)
    return df.groupby("stimulus_key", as_index=False).mean()


def make_trialwise_fmri_matrix(
    bids_root: str | Path,
    features_h5: str | Path,
    layer: str = "IT",
    stimulus_column: str | None = None,
    max_voxels: int | None = 10000,
) -> tuple[np.ndarray, np.ndarray]:
    events = load_events_with_stimuli(bids_root, stimulus_column)
    feat = load_feature_table(features_h5, layer)
    merged = events.merge(feat, on="stimulus_key", how="inner")
    feature_cols = [c for c in merged.columns if isinstance(c, int)]
    if not feature_cols:
        feature_cols = [c for c in merged.columns if str(c).isdigit()]
    X = merged[feature_cols].to_numpy(dtype=np.float32)

    fmri_files = list_fmri_files(bids_root)
    if not fmri_files:
        raise FileNotFoundError("No fMRI NIfTI files found under BIDS root.")
    img = nib.load(str(fmri_files[0]))
    mask_img = compute_epi_mask(img)
    Y_time = apply_mask(img, mask_img).astype(np.float32)
    if max_voxels and Y_time.shape[1] > max_voxels:
        rng = np.random.default_rng(42)
        idx = np.sort(rng.choice(Y_time.shape[1], size=max_voxels, replace=False))
        Y_time = Y_time[:, idx]

    if "onset" not in merged.columns:
        raise ValueError("events.tsv must contain onset for fMRI trial-to-volume alignment.")
    tr = float(img.header.get_zooms()[3]) if len(img.header.get_zooms()) > 3 else 2.0
    volume_idx = np.rint(merged["onset"].to_numpy(dtype=float) / tr).astype(int)
    valid = (volume_idx >= 0) & (volume_idx < Y_time.shape[0])
    X = X[valid]
    Y = Y_time[volume_idx[valid]]
    return X, Y


def cross_validated_ridge(X: np.ndarray, Y: np.ndarray, alpha: float, n_splits: int = 5):
    kf = KFold(n_splits=min(n_splits, len(X)), shuffle=True, random_state=42)
    y_pred = np.zeros_like(Y, dtype=np.float32)
    for train, test in kf.split(X):
        sx = StandardScaler().fit(X[train])
        sy = StandardScaler().fit(Y[train])
        model = Ridge(alpha=alpha, fit_intercept=True)
        model.fit(sx.transform(X[train]), sy.transform(Y[train]))
        y_pred[test] = sy.inverse_transform(model.predict(sx.transform(X[test]))).astype(np.float32)
    return pearsonr_columns(Y, y_pred), r2_columns(Y, y_pred), y_pred
