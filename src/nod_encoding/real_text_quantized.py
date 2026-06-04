from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .utils import ensure_dir, save_json

TEXT_INCLUDE_PATTERNS = [
    "dataset_description.json",
    "participants.tsv",
    "participants.json",
    "task-*.json",
    "sub-*/**/*_events.tsv",
    "sub-*/**/*_scans.tsv",
    "sub-*/**/*_channels.tsv",
    "sub-*/**/*_electrodes.tsv",
    "sub-*/**/*_coordsystem.json",
    "sub-*/**/*_eeg.json",
    "sub-*/**/*_bold.json",
]

TEXT_EXCLUDE_PATTERNS = [
    "*.nii", "*.nii.gz", "*.vhdr", "*.eeg", "*.vmrk", "*.edf", "*.bdf", "*.set", "*.fif",
    "*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff", "*.webp", "*.mp4", "*.mov", "*.avi",
]


def require_aws_cli() -> str:
    aws = shutil.which("aws")
    if aws:
        return aws
    print("aws CLI not found. Installing awscli because selective OpenNeuro text sync uses public S3.")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "awscli"])
    aws = shutil.which("aws")
    if not aws:
        raise RuntimeError("awscli installation finished, but the aws executable was not found on PATH.")
    return aws


def s3_sync_text(dataset: str, target: Path, dry_run: bool = False) -> None:
    aws = require_aws_cli()
    ensure_dir(target)
    cmd = [
        aws, "s3", "sync", f"s3://openneuro.org/{dataset}/", str(target),
        "--no-sign-request",
        "--only-show-errors",
        "--exclude", "*",
    ]
    for pat in TEXT_INCLUDE_PATTERNS:
        cmd.extend(["--include", pat])
    for pat in TEXT_EXCLUDE_PATTERNS:
        cmd.extend(["--exclude", pat])
    if dry_run:
        cmd.append("--dryrun")
    print("Running selective OpenNeuro text sync:")
    print(" ".join(cmd))
    subprocess.check_call(cmd)


def stable_hash(value: object, modulo: int = 4096) -> int:
    text = "" if pd.isna(value) else str(value)
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).hexdigest()
    return int(digest, 16) % modulo


def quantize_float_series(series: pd.Series, scale: float, dtype=np.int16) -> np.ndarray:
    numeric = pd.to_numeric(series, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    arr = np.rint(numeric * scale)
    info = np.iinfo(dtype)
    return np.clip(arr, info.min, info.max).astype(dtype)


def infer_string_columns(df: pd.DataFrame) -> list[str]:
    preferred = [
        "trial_type", "stimulus", "stim_file", "stimulus_file", "image", "image_file",
        "condition", "category", "response", "task", "event_type",
    ]
    cols = [c for c in preferred if c in df.columns]
    for c in df.columns:
        if c not in cols and df[c].dtype == object:
            cols.append(c)
    return cols[:8]


def process_events_to_quantized(dataset_root: Path, out_dir: Path, dataset_name: str, hash_dim: int = 4096) -> dict:
    ensure_dir(out_dir)
    event_files = sorted(dataset_root.glob("sub-*/**/*_events.tsv"))
    rows = []
    for event_file in event_files:
        try:
            df = pd.read_csv(event_file, sep="\t")
        except Exception as exc:
            print(f"Skipping unreadable events file {event_file}: {exc}")
            continue
        if df.empty:
            continue
        rel = event_file.relative_to(dataset_root)
        df = df.copy()
        df["__source_file"] = str(rel)
        df["__dataset"] = dataset_name
        rows.append(df)
    if not rows:
        raise FileNotFoundError(f"No *_events.tsv files were downloaded under {dataset_root}")

    events = pd.concat(rows, ignore_index=True, sort=False)
    string_cols = infer_string_columns(events)

    onset_i16 = quantize_float_series(events["onset"] if "onset" in events else pd.Series(0, index=events.index), scale=100.0)
    duration_i16 = quantize_float_series(events["duration"] if "duration" in events else pd.Series(0, index=events.index), scale=100.0)
    source_hash_u16 = np.array([stable_hash(v, hash_dim) for v in events["__source_file"]], dtype=np.uint16)

    hashed_cols = []
    hashed_names = []
    for col in string_cols:
        hashed_cols.append(np.array([stable_hash(v, hash_dim) for v in events[col]], dtype=np.uint16))
        hashed_names.append(col)
    if hashed_cols:
        hashed_matrix = np.stack(hashed_cols, axis=1)
    else:
        hashed_matrix = np.zeros((len(events), 0), dtype=np.uint16)

    X_quantized = np.column_stack([
        onset_i16.astype(np.int32),
        duration_i16.astype(np.int32),
        source_hash_u16.astype(np.int32),
        hashed_matrix.astype(np.int32),
    ]).astype(np.int16, copy=False)

    preview_cols = [c for c in ["onset", "duration", "trial_type", "stimulus", "stim_file", "condition", "response", "__source_file"] if c in events.columns]
    preview = events[preview_cols].head(200) if preview_cols else events.head(200)
    preview_path = out_dir / f"{dataset_name}_events_preview.csv"
    preview.to_csv(preview_path, index=False)

    npz_path = out_dir / f"{dataset_name}_events_quantized.npz"
    np.savez_compressed(
        npz_path,
        X_quantized=X_quantized,
        onset_centiseconds=onset_i16,
        duration_centiseconds=duration_i16,
        source_hash=source_hash_u16,
        hashed_categorical=hashed_matrix,
        hashed_column_names=np.array(hashed_names, dtype="U64"),
    )

    summary = {
        "dataset": dataset_name,
        "dataset_root": str(dataset_root),
        "n_event_files": len(event_files),
        "n_event_rows": int(len(events)),
        "quantized_npz": str(npz_path),
        "preview_csv": str(preview_path),
        "X_quantized_shape": list(X_quantized.shape),
        "X_quantized_dtype": str(X_quantized.dtype),
        "hash_dim": int(hash_dim),
        "hashed_columns": hashed_names,
        "download_policy": "text-only BIDS metadata/events; excludes NIfTI, EEG binary, stimulus images, and videos",
    }
    save_json(summary, out_dir / f"{dataset_name}_summary.json")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download only small real OpenNeuro text/BIDS files and convert events into compact quantized arrays."
    )
    parser.add_argument("--datasets", nargs="+", default=["ds004496", "ds005811"])
    parser.add_argument("--data-root", type=Path, default=Path("data/openneuro_text"))
    parser.add_argument("--out", type=Path, default=Path("results/real_text_quantized"))
    parser.add_argument("--hash-dim", type=int, default=4096)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    all_summaries = []
    for dataset in args.datasets:
        target = args.data_root / dataset
        s3_sync_text(dataset, target, dry_run=args.dry_run)
        if args.dry_run:
            continue
        summary = process_events_to_quantized(target, args.out, dataset, args.hash_dim)
        all_summaries.append(summary)
        print(json.dumps(summary, indent=2))

    if all_summaries:
        save_json({"datasets": all_summaries}, args.out / "combined_summary.json")
        print(f"Saved combined summary to {args.out / 'combined_summary.json'}")


if __name__ == "__main__":
    main()
