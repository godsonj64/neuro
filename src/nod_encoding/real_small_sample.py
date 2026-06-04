from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from .real_text_quantized import process_events_to_quantized, require_aws_cli
from .utils import ensure_dir, save_json

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
EEG_EXTS = {".vhdr", ".vmrk", ".eeg", ".edf", ".bdf", ".set", ".fif"}
NIFTI_EXTS = (".nii", ".nii.gz")
TEXT_EXTS = {".json", ".tsv"}


@dataclass(frozen=True)
class S3Object:
    key: str
    size: int


def run(cmd: list[str]) -> None:
    print(" ".join(cmd))
    subprocess.check_call(cmd)


def list_s3_objects(dataset: str) -> list[S3Object]:
    aws = require_aws_cli()
    cmd = [aws, "s3", "ls", f"s3://openneuro.org/{dataset}/", "--recursive", "--no-sign-request"]
    print("Listing OpenNeuro S3 keys. This does not download data.")
    proc = subprocess.run(cmd, check=True, text=True, capture_output=True)
    objects: list[S3Object] = []
    pattern = re.compile(r"^\S+\s+\S+\s+(\d+)\s+(.+)$")
    for line in proc.stdout.splitlines():
        m = pattern.match(line.strip())
        if not m:
            continue
        size = int(m.group(1))
        key = m.group(2)
        objects.append(S3Object(key=key, size=size))
    return objects


def suffix_of_key(key: str) -> str:
    lower = key.lower()
    if lower.endswith(".nii.gz"):
        return ".nii.gz"
    return Path(lower).suffix


def is_nifti(key: str) -> bool:
    lower = key.lower()
    return lower.endswith(NIFTI_EXTS)


def is_eeg(key: str) -> bool:
    return suffix_of_key(key) in EEG_EXTS


def is_image(key: str) -> bool:
    return suffix_of_key(key) in IMAGE_EXTS


def is_text(key: str) -> bool:
    return suffix_of_key(key) in TEXT_EXTS or key.endswith("_events.tsv")


def choose_small_objects(
    objects: list[S3Object],
    max_nifti: int,
    max_eeg_sets: int,
    max_images: int,
    max_text: int,
    max_file_mb: float,
) -> list[S3Object]:
    max_bytes = int(max_file_mb * 1024 * 1024)
    small = [o for o in objects if o.size <= max_bytes and o.size > 0]
    chosen: list[S3Object] = []

    niftis = sorted([o for o in small if is_nifti(o.key) and "bold" in o.key.lower()], key=lambda o: o.size)
    if len(niftis) < max_nifti:
        niftis = sorted([o for o in small if is_nifti(o.key)], key=lambda o: o.size)
    chosen.extend(niftis[:max_nifti])

    # For BrainVision, include complete .vhdr/.vmrk/.eeg triplets with the same stem.
    eeg_by_stem: dict[str, list[S3Object]] = defaultdict(list)
    for o in small:
        if is_eeg(o.key):
            lower = o.key.lower()
            stem = re.sub(r"\.(vhdr|vmrk|eeg|edf|bdf|set|fif)$", "", lower)
            eeg_by_stem[stem].append(o)
    eeg_sets = []
    for stem, group in eeg_by_stem.items():
        exts = {suffix_of_key(g.key) for g in group}
        if {".vhdr", ".vmrk", ".eeg"}.issubset(exts) or exts & {".edf", ".bdf", ".set", ".fif"}:
            eeg_sets.append(sorted(group, key=lambda o: o.key))
    eeg_sets = sorted(eeg_sets, key=lambda group: sum(o.size for o in group))
    for group in eeg_sets[:max_eeg_sets]:
        chosen.extend(group)

    images = sorted([o for o in small if is_image(o.key)], key=lambda o: o.size)
    chosen.extend(images[:max_images])

    text_priority = []
    for o in small:
        key = o.key.lower()
        if key.endswith("dataset_description.json") or key.endswith("participants.tsv") or key.endswith("_events.tsv") or key.endswith("_scans.tsv") or key.endswith("_channels.tsv") or key.endswith(".json"):
            text_priority.append(o)
    text_priority = sorted(text_priority, key=lambda o: (0 if o.key.endswith("_events.tsv") else 1, o.size))
    chosen.extend(text_priority[:max_text])

    # Deduplicate while preserving order.
    seen = set()
    unique = []
    for o in chosen:
        if o.key not in seen:
            unique.append(o)
            seen.add(o.key)
    return unique


def download_objects(dataset: str, objects: list[S3Object], target: Path) -> list[dict]:
    aws = require_aws_cli()
    downloaded = []
    for obj in objects:
        dst = target / obj.key
        ensure_dir(dst.parent)
        run([aws, "s3", "cp", f"s3://openneuro.org/{dataset}/{obj.key}", str(dst), "--no-sign-request", "--only-show-errors"])
        downloaded.append({"key": obj.key, "local_path": str(dst), "size_bytes": obj.size})
    return downloaded


def quantize_nifti(path: Path, out_dir: Path, max_volumes: int = 4, spatial_stride: int = 4) -> dict | None:
    try:
        import nibabel as nib
    except Exception as exc:
        print(f"Skipping NIfTI quantization because nibabel failed: {exc}")
        return None
    try:
        img = nib.load(str(path))
        data = np.asanyarray(img.dataobj)
        if data.ndim == 4:
            data = data[..., :max_volumes]
        data = data[::spatial_stride, ::spatial_stride, ::spatial_stride, ...]
        arr = np.asarray(data, dtype=np.float32)
        finite = np.isfinite(arr)
        if not np.any(finite):
            return None
        lo, hi = np.percentile(arr[finite], [1, 99])
        if hi <= lo:
            hi = lo + 1.0
        q = np.rint((np.clip(arr, lo, hi) - lo) / (hi - lo) * 255.0).astype(np.uint8)
        out = out_dir / (path.name.replace(".nii.gz", "").replace(".nii", "") + "_nifti_uint8.npz")
        np.savez_compressed(out, volume_uint8=q, lo=np.float32(lo), hi=np.float32(hi), original_shape=np.array(img.shape))
        return {"source": str(path), "quantized": str(out), "shape": list(q.shape), "dtype": "uint8", "lo": float(lo), "hi": float(hi)}
    except Exception as exc:
        print(f"Could not quantize NIfTI {path}: {exc}")
        return None


def read_eeg_any(path: Path):
    import mne
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
    raise ValueError(f"Unsupported direct EEG reader for {path}")


def quantize_eeg(path: Path, out_dir: Path, seconds: float = 30.0, target_sfreq: float = 100.0) -> dict | None:
    if path.suffix.lower() not in {".vhdr", ".edf", ".bdf", ".set", ".fif"}:
        return None
    try:
        raw = read_eeg_any(path)
        raw.pick_types(eeg=True, meg=False, eog=False, stim=False, exclude=[])
        raw.resample(target_sfreq, verbose="ERROR")
        n = min(raw.n_times, int(seconds * raw.info["sfreq"]))
        data = raw.get_data(start=0, stop=n).astype(np.float32)  # volts
        microvolts = data * 1e6
        q = np.clip(np.rint(microvolts * 10.0), -32768, 32767).astype(np.int16)  # 0.1 uV units
        out = out_dir / (path.name.replace(path.suffix, "") + "_eeg_int16.npz")
        np.savez_compressed(out, eeg_int16=q, sfreq=np.float32(raw.info["sfreq"]), ch_names=np.array(raw.ch_names, dtype="U64"), unit="0.1 microvolt")
        return {"source": str(path), "quantized": str(out), "shape": list(q.shape), "dtype": "int16", "sfreq": float(raw.info["sfreq"])}
    except Exception as exc:
        print(f"Could not quantize EEG {path}: {exc}")
        return None


def quantize_image(path: Path, out_dir: Path, size: int = 64) -> dict | None:
    try:
        img = Image.open(path).convert("RGB").resize((size, size))
        arr = np.asarray(img, dtype=np.uint8)
        out = out_dir / (path.stem + "_image_uint8.npz")
        np.savez_compressed(out, image_uint8=arr)
        return {"source": str(path), "quantized": str(out), "shape": list(arr.shape), "dtype": "uint8"}
    except Exception as exc:
        print(f"Could not quantize image {path}: {exc}")
        return None


def preprocess_downloaded(root: Path, out_dir: Path, dataset: str) -> dict:
    ensure_dir(out_dir)
    results = {"dataset": dataset, "nifti": [], "eeg": [], "images": [], "events": None}
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if is_nifti(str(p)):
            q = quantize_nifti(p, out_dir)
            if q:
                results["nifti"].append(q)
        elif p.suffix.lower() in {".vhdr", ".edf", ".bdf", ".set", ".fif"}:
            q = quantize_eeg(p, out_dir)
            if q:
                results["eeg"].append(q)
        elif is_image(str(p)):
            q = quantize_image(p, out_dir)
            if q:
                results["images"].append(q)
    try:
        results["events"] = process_events_to_quantized(root, out_dir, dataset)
    except Exception as exc:
        results["events_error"] = str(exc)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Download a tiny real OpenNeuro sample including NIfTI, EEG binary, and images, then quantize it.")
    parser.add_argument("--datasets", nargs="+", default=["ds004496", "ds005811"])
    parser.add_argument("--data-root", type=Path, default=Path("data/openneuro_small_real"))
    parser.add_argument("--out", type=Path, default=Path("results/openneuro_small_quantized"))
    parser.add_argument("--max-nifti", type=int, default=1)
    parser.add_argument("--max-eeg-sets", type=int, default=1)
    parser.add_argument("--max-images", type=int, default=8)
    parser.add_argument("--max-text", type=int, default=20)
    parser.add_argument("--max-file-mb", type=float, default=80.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    combined = []
    for dataset in args.datasets:
        objects = list_s3_objects(dataset)
        chosen = choose_small_objects(objects, args.max_nifti, args.max_eeg_sets, args.max_images, args.max_text, args.max_file_mb)
        plan = [{"key": o.key, "size_mb": round(o.size / 1024 / 1024, 3)} for o in chosen]
        dataset_root = args.data_root / dataset
        dataset_out = args.out / dataset
        ensure_dir(dataset_out)
        save_json({"dataset": dataset, "chosen": plan}, dataset_out / "download_plan.json")
        print(json.dumps({"dataset": dataset, "chosen": plan}, indent=2))
        if args.dry_run:
            continue
        downloaded = download_objects(dataset, chosen, dataset_root)
        quantized = preprocess_downloaded(dataset_root, dataset_out, dataset)
        summary = {"dataset": dataset, "downloaded": downloaded, "quantized": quantized}
        save_json(summary, dataset_out / "summary.json")
        combined.append(summary)
    if combined:
        save_json({"datasets": combined}, args.out / "combined_summary.json")
        print(f"Saved combined summary to {args.out / 'combined_summary.json'}")


if __name__ == "__main__":
    main()
