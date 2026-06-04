from __future__ import annotations

import argparse
import fnmatch
from pathlib import Path


def download_dataset(dataset: str, target: Path) -> None:
    try:
        import openneuro as on
    except ImportError as exc:
        raise SystemExit("Install openneuro-py first: pip install openneuro-py") from exc
    target.mkdir(parents=True, exist_ok=True)
    on.download(dataset=dataset, target_dir=str(target))


def prune_by_pattern(root: Path, include: str | None, max_files: int | None) -> None:
    if include is None and max_files is None:
        return
    all_files = [p for p in root.rglob("*") if p.is_file()]
    keep = set()
    count = 0
    for p in all_files:
        rel = str(p.relative_to(root))
        matched = include is None or fnmatch.fnmatch(rel, include)
        if matched and (max_files is None or count < max_files):
            keep.add(p)
            count += 1
    for p in all_files:
        if p not in keep:
            try:
                p.unlink()
            except OSError:
                pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Download an OpenNeuro dataset with optional local pruning.")
    parser.add_argument("--dataset", required=True, help="OpenNeuro accession, e.g. ds004496")
    parser.add_argument("--target", required=True, type=Path)
    parser.add_argument("--include", default=None, help="Optional glob relative to dataset root, e.g. 'sub-01/**'")
    parser.add_argument("--max-files", type=int, default=None, help="Keep at most this many matched files after download.")
    args = parser.parse_args()
    download_dataset(args.dataset, args.target)
    prune_by_pattern(args.target, args.include, args.max_files)
    print(f"Downloaded {args.dataset} into {args.target}")


if __name__ == "__main__":
    main()
