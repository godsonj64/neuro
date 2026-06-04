from __future__ import annotations

from pathlib import Path

import pandas as pd

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
FMRI_PATTERNS = ("*_bold.nii", "*_bold.nii.gz", "*_func.nii", "*_func.nii.gz")
EEG_EXTS = {".vhdr", ".edf", ".bdf", ".set", ".fif"}


def list_events(bids_root: str | Path) -> list[Path]:
    return sorted(Path(bids_root).glob("sub-*/**/*_events.tsv"))


def list_stimulus_images(bids_root: str | Path) -> dict[str, Path]:
    root = Path(bids_root)
    found: dict[str, Path] = {}
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            found[p.name] = p
            found[str(p.relative_to(root))] = p
    return found


def infer_stimulus_column(events: pd.DataFrame, preferred: str | None = None) -> str:
    if preferred and preferred in events.columns:
        return preferred
    candidates = [
        "stimulus", "stim_file", "stimulus_file", "image", "image_file",
        "filename", "file", "trial_image", "stimulus_name", "condition"
    ]
    for c in candidates:
        if c in events.columns:
            return c
    object_cols = [c for c in events.columns if events[c].dtype == object]
    if object_cols:
        return object_cols[0]
    raise ValueError(f"Could not infer stimulus column. Columns: {list(events.columns)}")


def load_events_with_stimuli(
    bids_root: str | Path,
    stimulus_column: str | None = None,
) -> pd.DataFrame:
    root = Path(bids_root)
    image_index = list_stimulus_images(root)
    rows = []
    for event_file in list_events(root):
        df = pd.read_csv(event_file, sep="\t")
        if df.empty:
            continue
        col = infer_stimulus_column(df, stimulus_column)
        for _, row in df.iterrows():
            raw = str(row[col])
            path = None
            if raw in image_index:
                path = image_index[raw]
            else:
                candidate = root / raw
                if candidate.exists():
                    path = candidate
                else:
                    candidate2 = event_file.parent / raw
                    if candidate2.exists():
                        path = candidate2
            if path is None:
                continue
            rec = row.to_dict()
            rec["event_file"] = str(event_file)
            rec["stimulus_path"] = str(path)
            rec["stimulus_key"] = Path(path).name
            rows.append(rec)
    if not rows:
        raise FileNotFoundError(
            "No events could be matched to stimulus image files. "
            "Check BIDS stimuli paths or pass --stimulus-column."
        )
    return pd.DataFrame(rows)


def list_fmri_files(bids_root: str | Path) -> list[Path]:
    root = Path(bids_root)
    files: list[Path] = []
    for pattern in FMRI_PATTERNS:
        files.extend(root.glob(f"sub-*/**/{pattern}"))
    return sorted(set(files))


def list_eeg_files(bids_root: str | Path) -> list[Path]:
    root = Path(bids_root)
    return sorted(p for p in root.glob("sub-*/**/*") if p.is_file() and p.suffix.lower() in EEG_EXTS)
