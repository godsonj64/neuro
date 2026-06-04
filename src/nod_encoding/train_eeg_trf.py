from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .eeg import cross_validated_trf, make_eeg_trf_matrix
from .utils import ensure_dir, save_json, seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description="Train EEG temporal response function from CORnet-S features.")
    parser.add_argument("--bids-root", required=True, type=Path)
    parser.add_argument("--features", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--layer", default="IT")
    parser.add_argument("--stimulus-column", default=None)
    parser.add_argument("--tmin", type=float, default=-0.1)
    parser.add_argument("--tmax", type=float, default=0.6)
    parser.add_argument("--alpha", type=float, default=10.0)
    parser.add_argument("--decim", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    seed_everything(args.seed)
    ensure_dir(args.out)
    X, Y, lags, sfreq, ch_names = make_eeg_trf_matrix(
        args.bids_root, args.features, args.layer, args.stimulus_column, args.tmin, args.tmax, args.decim
    )
    r, r2, _ = cross_validated_trf(X, Y, args.alpha)
    np.save(args.out / "channel_pearson_r.npy", r)
    np.save(args.out / "channel_r2.npy", r2)
    np.save(args.out / "lags_samples.npy", lags)
    save_json({
        "n_timepoints": int(X.shape[0]),
        "n_lagged_features": int(X.shape[1]),
        "n_channels": int(Y.shape[1]),
        "alpha": args.alpha,
        "layer": args.layer,
        "sfreq_after_decim": sfreq,
        "channels": ch_names,
        "mean_pearson_r": float(np.nanmean(r)),
        "median_pearson_r": float(np.nanmedian(r)),
        "mean_r2": float(np.nanmean(r2)),
    }, args.out / "metrics.json")
    print(f"Saved EEG TRF results to {args.out}")


if __name__ == "__main__":
    main()
