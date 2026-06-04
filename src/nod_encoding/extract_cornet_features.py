from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from .bids_utils import load_events_with_stimuli
from .cornet import FeatureHook, load_cornet_s, pooled_flatten, resolve_layers
from .utils import ensure_dir, seed_everything


class ImagePathDataset(Dataset):
    def __init__(self, paths: list[str]):
        self.paths = paths
        self.tf = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        p = self.paths[idx]
        img = Image.open(p).convert("RGB")
        return self.tf(img), Path(p).name


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract pretrained CORnet-S features for BIDS stimulus images.")
    parser.add_argument("--bids-root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--layers", nargs="+", default=["V1", "V2", "V4", "IT"])
    parser.add_argument("--stimulus-column", default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    seed_everything(args.seed)
    events = load_events_with_stimuli(args.bids_root, args.stimulus_column)
    unique_paths = sorted(events["stimulus_path"].drop_duplicates().tolist())
    dataset = ImagePathDataset(unique_paths)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = load_cornet_s(args.device)
    layers = resolve_layers(model, args.layers)
    hooks = {name: FeatureHook(module) for name, module in layers.items()}
    layer_features: dict[str, list[np.ndarray]] = {name: [] for name in layers}
    keys: list[str] = []

    with torch.no_grad():
        for batch, names in tqdm(loader, desc="CORnet-S feature extraction"):
            batch = batch.to(args.device)
            _ = model(batch)
            keys.extend(list(names))
            for name, hook in hooks.items():
                if hook.output is None:
                    raise RuntimeError(f"No activation captured for layer {name}")
                feat = pooled_flatten(hook.output).cpu().numpy().astype("float32")
                layer_features[name].append(feat)

    ensure_dir(args.out.parent)
    with h5py.File(args.out, "w") as h5:
        h5.create_dataset("stimulus_key", data=np.array(keys, dtype="S"))
        for name, chunks in layer_features.items():
            arr = np.concatenate(chunks, axis=0)
            h5.create_dataset(name, data=arr, compression="gzip")
    for hook in hooks.values():
        hook.close()
    print(f"Saved features to {args.out}")


if __name__ == "__main__":
    main()
