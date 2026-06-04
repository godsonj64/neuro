from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from torchvision import transforms

from .cornet import FeatureHook, load_cornet_s, pooled_flatten, resolve_layers
from .eeg import make_lagged_design, read_eeg
from .metrics import pearsonr_columns, r2_columns
from .real_text_quantized import require_aws_cli
from .utils import ensure_dir, save_json, seed_everything

IMAGE_EXTS = (".jpeg", ".jpg", ".png", ".bmp", ".tif", ".tiff", ".webp")


def run(cmd: list[str]) -> None:
    print(" ".join(cmd))
    subprocess.check_call(cmd)


def parse_bids_run(path: Path) -> dict[str, str]:
    out = {}
    for key in ["sub", "ses", "task", "run"]:
        m = re.search(rf"{key}-([^_]+)", path.name)
        if m:
            out[key] = m.group(1)
    return out


def find_raw_eeg_file(root: Path, preferred: str | None = None) -> Path:
    if preferred:
        p = Path(preferred)
        return p if p.is_absolute() else root / p
    candidates = []
    for ext in [".set", ".vhdr", ".edf", ".bdf", ".fif"]:
        candidates.extend(root.glob(f"sub-*/**/*_eeg{ext}"))
    candidates = [p for p in candidates if "derivatives" not in str(p).lower() and "_ica" not in p.name.lower()]
    if not candidates:
        raise FileNotFoundError(f"No raw EEG file found under {root}")
    return sorted(candidates, key=lambda p: p.stat().st_size)[0]


def expected_events_path(eeg_path: Path) -> Path:
    return eeg_path.with_name(eeg_path.name.replace("_eeg" + eeg_path.suffix, "_events.tsv"))


def download_matching_events(dataset: str, eeg_path: Path, dataset_root: Path) -> Path:
    events = expected_events_path(eeg_path)
    if events.exists():
        return events
    rel = events.relative_to(dataset_root)
    key = f"{dataset}/{rel.as_posix()}"
    aws = require_aws_cli()
    ensure_dir(events.parent)
    run([aws, "s3", "cp", f"s3://openneuro.org/{key}", str(events), "--no-sign-request", "--only-show-errors"])
    return events


def infer_stimulus_column(df: pd.DataFrame) -> tuple[str, bool]:
    """Return (column, exact_image_names_available).

    Some NOD-EEG events use StimulusPresentation as an ID rather than a literal
    ImageNet filename. In that case we still run an engineering micro-TRF using
    locally sampled real images as a deterministic feature dictionary.
    """
    preferred = ["StimulusPresentation", "stimulus", "stim_file", "stimulus_file", "image", "image_file", "filename", "trial_type", "value"]
    image_regex = r"\.jpe?g|\.png|n\d{8}_\d+"
    for col in preferred:
        if col in df.columns:
            vals = df[col].astype(str)
            if vals.str.contains(image_regex, case=False, regex=True).any():
                return col, True
    for col in df.columns:
        vals = df[col].astype(str)
        if vals.str.contains(image_regex, case=False, regex=True).any():
            return col, True
    for col in ["StimulusPresentation", "trial_type", "value"]:
        if col in df.columns:
            return col, False
    raise ValueError(f"Could not infer a usable stimulus/id column from {list(df.columns)}")


def normalize_image_name(value: object) -> str | None:
    text = "" if pd.isna(value) else str(value)
    m = re.search(r"(n\d{8}_\d+\.(?:JPEG|JPG|jpeg|jpg|png|PNG))", text)
    if m:
        return m.group(1).replace(".jpg", ".JPEG").replace(".JPG", ".JPEG").replace(".jpeg", ".JPEG")
    m = re.search(r"(n\d{8}_\d+)", text)
    if m:
        return m.group(1) + ".JPEG"
    if text.lower().endswith(IMAGE_EXTS):
        return Path(text).name
    return None


def local_image_pool(dataset_root: Path) -> dict[str, Path]:
    roots = [dataset_root / "stimuli" / "ImageNet", dataset_root / "stimuli" / "imagenet"]
    found = {}
    for root in roots:
        if root.exists():
            for p in root.rglob("*"):
                if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                    found[p.name] = p
    return dict(sorted(found.items()))


def download_stimuli(dataset: str, image_names: list[str], dataset_root: Path) -> dict[str, Path]:
    aws = require_aws_cli()
    out_dir = dataset_root / "stimuli" / "ImageNet"
    ensure_dir(out_dir)
    mapping: dict[str, Path] = {}
    for name in image_names:
        dst = out_dir / name
        if not dst.exists():
            key = f"{dataset}/stimuli/ImageNet/{name}"
            try:
                run([aws, "s3", "cp", f"s3://openneuro.org/{key}", str(dst), "--no-sign-request", "--only-show-errors"])
            except subprocess.CalledProcessError:
                print(f"Could not download stimulus {name}; skipping.")
                continue
        if dst.exists():
            mapping[name] = dst
    return mapping


def assign_fallback_images(events: pd.DataFrame, stim_col: str, dataset_root: Path) -> tuple[pd.DataFrame, dict[str, Path]]:
    pool = local_image_pool(dataset_root)
    if not pool:
        raise FileNotFoundError("No local stimulus image pool found. Run real_small_sample with --max-images > 0 first.")
    image_names = list(pool.keys())
    ids = events[stim_col].astype(str).fillna("missing")
    assigned = []
    for value in ids:
        idx = abs(hash(value)) % len(image_names)
        assigned.append(image_names[idx])
    out = events.copy()
    out["__image_name"] = assigned
    out["__stimulus_assignment_mode"] = "fallback_hash_to_local_real_images"
    return out, pool


def extract_cornet_features(image_paths: list[Path], layer: str, device: str) -> dict[str, np.ndarray]:
    tf = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    model = load_cornet_s(device)
    module = resolve_layers(model, [layer])[layer]
    hook = FeatureHook(module)
    feats = {}
    with torch.no_grad():
        for p in image_paths:
            img = tf(Image.open(p).convert("RGB")).unsqueeze(0).to(device)
            _ = model(img)
            vec = pooled_flatten(hook.output).cpu().numpy()[0].astype(np.float32)
            feats[p.name] = vec
    hook.close()
    return feats


def pca_reduce(X: np.ndarray, n_components: int) -> np.ndarray:
    X = X.astype(np.float32) - X.astype(np.float32).mean(axis=0, keepdims=True)
    if min(X.shape) <= 1:
        return X
    u, s, _ = np.linalg.svd(X, full_matrices=False)
    k = min(n_components, u.shape[1])
    return (u[:, :k] * s[:k]).astype(np.float32)


def fit_tiny_trf(X_lag: np.ndarray, Y: np.ndarray, alpha: float, n_splits: int = 3):
    n_splits = min(n_splits, len(X_lag))
    if n_splits < 2:
        raise ValueError("Not enough samples for cross-validation.")
    y_pred = np.zeros_like(Y, dtype=np.float32)
    for train, test in KFold(n_splits=n_splits, shuffle=True, random_state=42).split(X_lag):
        sx = StandardScaler(with_mean=False).fit(X_lag[train])
        sy = StandardScaler().fit(Y[train])
        model = Ridge(alpha=alpha)
        model.fit(sx.transform(X_lag[train]), sy.transform(Y[train]))
        y_pred[test] = sy.inverse_transform(model.predict(sx.transform(X_lag[test]))).astype(np.float32)
    return pearsonr_columns(Y, y_pred), r2_columns(Y, y_pred)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a matched real EEG-events-stimuli micro TRF using CORnet-S features.")
    parser.add_argument("--dataset", default="ds005811")
    parser.add_argument("--data-root", type=Path, default=Path("data/openneuro_small_real/ds005811"))
    parser.add_argument("--out", type=Path, default=Path("results/matched_micro_trf"))
    parser.add_argument("--eeg-file", default=None)
    parser.add_argument("--layer", default="IT")
    parser.add_argument("--max-events", type=int, default=64)
    parser.add_argument("--feature-dim", type=int, default=32)
    parser.add_argument("--tmin", type=float, default=0.0)
    parser.add_argument("--tmax", type=float, default=0.4)
    parser.add_argument("--alpha", type=float, default=10.0)
    parser.add_argument("--target-sfreq", type=float, default=100.0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    seed_everything(42)
    ensure_dir(args.out)
    eeg_path = find_raw_eeg_file(args.data_root, args.eeg_file)
    bids = parse_bids_run(eeg_path)
    events_path = download_matching_events(args.dataset, eeg_path, args.data_root)
    events = pd.read_csv(events_path, sep="\t")
    stim_col, exact_names = infer_stimulus_column(events)
    events = events.copy().dropna(subset=["onset"]).head(args.max_events)

    if exact_names:
        events["__image_name"] = events[stim_col].map(normalize_image_name)
        events = events.dropna(subset=["__image_name"])
        image_map = download_stimuli(args.dataset, sorted(events["__image_name"].unique()), args.data_root)
        events = events[events["__image_name"].isin(image_map.keys())].copy()
        assignment_mode = "exact_event_image_names"
    else:
        events, image_map = assign_fallback_images(events, stim_col, args.data_root)
        assignment_mode = "fallback_event_id_hash_to_local_real_images"

    if events.empty:
        raise ValueError("No usable events remained after stimulus processing.")

    feats_by_name = extract_cornet_features([image_map[n] for n in sorted(set(events["__image_name"]))], args.layer, args.device)
    raw_X = np.stack([feats_by_name[name] for name in events["__image_name"]], axis=0)
    X_events = pca_reduce(raw_X, args.feature_dim)

    raw = read_eeg(eeg_path)
    raw.pick_types(eeg=True, meg=False, eog=False, stim=False, exclude=[])
    raw.resample(args.target_sfreq, verbose="ERROR")
    Y = raw.get_data().T.astype(np.float32) * 1e6
    sfreq = float(raw.info["sfreq"])
    onset_samples = np.rint(events["onset"].to_numpy(dtype=float) * sfreq).astype(int)
    valid_events = (onset_samples >= 0) & (onset_samples < Y.shape[0])
    X_events = X_events[valid_events]
    onset_samples = onset_samples[valid_events]
    events = events.iloc[np.where(valid_events)[0]].copy()

    X_lag, lags = make_lagged_design(X_events, onset_samples, Y.shape[0], sfreq, args.tmin, args.tmax)
    valid_time = np.any(X_lag != 0, axis=1)
    X_fit, Y_fit = X_lag[valid_time], Y[valid_time]
    if len(X_fit) < 10:
        raise ValueError(f"Too few aligned time samples for TRF: {len(X_fit)}")

    r, r2 = fit_tiny_trf(X_fit, Y_fit, args.alpha)
    np.savez_compressed(
        args.out / "matched_micro_trf_arrays.npz",
        X_lag=X_fit.astype(np.float32), Y_microvolts=Y_fit.astype(np.float32),
        channel_pearson_r=r.astype(np.float32), channel_r2=r2.astype(np.float32),
        lags_samples=lags.astype(np.int32), sfreq=np.float32(sfreq), ch_names=np.array(raw.ch_names, dtype="U64"),
    )
    events.to_csv(args.out / "matched_events_used.csv", index=False)
    summary = {
        "status": "completed", "dataset": args.dataset, "bids": bids,
        "eeg_file": str(eeg_path), "events_file": str(events_path),
        "stimulus_column": stim_col, "stimulus_assignment_mode": assignment_mode,
        "n_events_used": int(len(events)), "n_unique_images": int(len(set(events["__image_name"]))),
        "cornet_layer": args.layer, "raw_feature_dim": int(raw_X.shape[1]),
        "reduced_feature_dim": int(X_events.shape[1]), "n_lags": int(len(lags)),
        "X_lag_shape": list(X_fit.shape), "Y_shape": list(Y_fit.shape), "sfreq": sfreq,
        "mean_channel_pearson_r": float(np.nanmean(r)), "median_channel_pearson_r": float(np.nanmedian(r)),
        "mean_channel_r2": float(np.nanmean(r2)),
        "note": "Tiny matched-run engineering validation. If fallback mode is used, stimulus IDs were not exact filenames.",
    }
    save_json(summary, args.out / "summary.json")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
