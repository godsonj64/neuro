from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .eeg import cross_validated_trf, make_lagged_design
from .fmri import cross_validated_ridge
from .utils import ensure_dir, save_json, seed_everything


def run_smoke_test(out: Path, n_trials: int = 64, n_features: int = 32, n_voxels: int = 48, n_channels: int = 8) -> dict:
    seed_everything(42)
    rng = np.random.default_rng(42)

    # Synthetic fMRI encoding: known linear mapping plus small noise.
    X_fmri = rng.normal(size=(n_trials, n_features)).astype(np.float32)
    B = rng.normal(scale=0.2, size=(n_features, n_voxels)).astype(np.float32)
    Y_fmri = X_fmri @ B + rng.normal(scale=0.05, size=(n_trials, n_voxels)).astype(np.float32)
    fmri_r, fmri_r2, _ = cross_validated_ridge(X_fmri, Y_fmri, alpha=1.0, n_splits=4)

    # Synthetic EEG TRF encoding: event features expanded into a lagged design.
    n_events = 24
    event_features = rng.normal(size=(n_events, n_features)).astype(np.float32)
    onset_samples = np.arange(20, 20 + 10 * n_events, 10)
    X_trf, lags = make_lagged_design(
        event_features,
        onset_samples,
        n_times=320,
        sfreq=100.0,
        tmin=0.0,
        tmax=0.05,
    )
    W = rng.normal(scale=0.1, size=(X_trf.shape[1], n_channels)).astype(np.float32)
    Y_eeg = X_trf @ W + rng.normal(scale=0.02, size=(X_trf.shape[0], n_channels)).astype(np.float32)
    valid = np.any(X_trf != 0, axis=1)
    eeg_r, eeg_r2, _ = cross_validated_trf(X_trf[valid], Y_eeg[valid], alpha=1.0, n_splits=4)

    ensure_dir(out)
    np.save(out / "smoke_fmri_voxel_r.npy", fmri_r)
    np.save(out / "smoke_fmri_voxel_r2.npy", fmri_r2)
    np.save(out / "smoke_eeg_channel_r.npy", eeg_r)
    np.save(out / "smoke_eeg_channel_r2.npy", eeg_r2)

    summary = {
        "status": "passed",
        "note": "Synthetic small-data test only. No OpenNeuro data downloaded.",
        "fmri_mean_pearson_r": float(np.nanmean(fmri_r)),
        "fmri_mean_r2": float(np.nanmean(fmri_r2)),
        "eeg_mean_pearson_r": float(np.nanmean(eeg_r)),
        "eeg_mean_r2": float(np.nanmean(eeg_r2)),
        "n_trials": int(n_trials),
        "n_features": int(n_features),
        "n_voxels": int(n_voxels),
        "n_channels": int(n_channels),
        "n_lags": int(len(lags)),
    }
    save_json(summary, out / "smoke_metrics.json")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a tiny synthetic smoke test without downloading OpenNeuro files.")
    parser.add_argument("--out", type=Path, default=Path("results/smoke_test"))
    args = parser.parse_args()
    summary = run_smoke_test(args.out)
    print("Small synthetic smoke test completed.")
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
