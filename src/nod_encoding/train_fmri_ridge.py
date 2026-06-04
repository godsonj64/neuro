from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .fmri import cross_validated_ridge, make_trialwise_fmri_matrix
from .utils import ensure_dir, save_json, seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description="Train voxel-wise fMRI Ridge encoding model.")
    parser.add_argument("--bids-root", required=True, type=Path)
    parser.add_argument("--features", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--layer", default="IT")
    parser.add_argument("--stimulus-column", default=None)
    parser.add_argument("--alpha", type=float, default=100.0)
    parser.add_argument("--max-voxels", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    seed_everything(args.seed)
    ensure_dir(args.out)
    X, Y = make_trialwise_fmri_matrix(args.bids_root, args.features, args.layer, args.stimulus_column, args.max_voxels)
    r, r2, _ = cross_validated_ridge(X, Y, args.alpha)
    np.save(args.out / "voxel_pearson_r.npy", r)
    np.save(args.out / "voxel_r2.npy", r2)
    save_json({
        "n_trials": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "n_voxels": int(Y.shape[1]),
        "alpha": args.alpha,
        "layer": args.layer,
        "mean_pearson_r": float(np.nanmean(r)),
        "median_pearson_r": float(np.nanmedian(r)),
        "mean_r2": float(np.nanmean(r2)),
    }, args.out / "metrics.json")
    print(f"Saved fMRI results to {args.out}")


if __name__ == "__main__":
    main()
