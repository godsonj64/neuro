from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def describe_npz(path: Path) -> None:
    arrs = np.load(path, allow_pickle=True)
    print(f"\nFILE: {path}")
    for key in arrs.files:
        x = arrs[key]
        print(f"  {key}: shape={getattr(x, 'shape', None)}, dtype={getattr(x, 'dtype', None)}")
        if np.issubdtype(x.dtype, np.number):
            flat = np.asarray(x).ravel()
            if flat.size:
                print(
                    f"    min={np.nanmin(flat):.6g}, max={np.nanmax(flat):.6g}, "
                    f"mean={np.nanmean(flat):.6g}, std={np.nanstd(flat):.6g}, "
                    f"unique<=10={np.unique(flat[:min(flat.size, 10000)]).size <= 10}"
                )


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect quantized NPZ outputs.")
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()
    for p in args.paths:
        describe_npz(p)


if __name__ == "__main__":
    main()
