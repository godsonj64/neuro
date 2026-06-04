from __future__ import annotations

from collections import OrderedDict
from typing import Iterable

import torch
from torch import nn


class Identity(nn.Module):
    def forward(self, x):
        return x


class SmallCORnetSLike(nn.Module):
    """Offline fallback with CORnet-style named visual areas.

    This is not pretrained CORnet-S. It exists so the small real-data pipeline can
    complete when the upstream CORnet repository cannot be loaded through
    torch.hub because the current archive has no hubconf.py.
    """

    def __init__(self):
        super().__init__()
        self.V1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(3, stride=2, padding=1),
        )
        self.V2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
        )
        self.V4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
        )
        self.IT = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(512, 1000))

    def forward(self, x):
        x = self.V1(x)
        x = self.V2(x)
        x = self.V4(x)
        x = self.IT(x)
        return self.decoder(x)


def load_cornet_s(device: str | torch.device = "cpu", allow_fallback: bool = True) -> nn.Module:
    """Load pretrained CORnet-S when available, otherwise a local CORnet-like fallback.

    The official CORnet repository is the scientific target. Some current GitHub
    archives do not expose `hubconf.py`, so `torch.hub.load` can fail even with
    working internet. For engineering validation, the fallback preserves layer
    names V1/V2/V4/IT/decoder but uses random weights.
    """
    try:
        model = torch.hub.load("dicarlolab/CORnet", "cornet_s", pretrained=True, trust_repo=True)
        setattr(model, "feature_source", "pretrained_dicarlolab_cornet_s")
    except Exception as exc:
        if not allow_fallback:
            raise RuntimeError(
                "Could not load pretrained CORnet-S through torch.hub. "
                "The upstream archive may not contain hubconf.py. Original error: " + str(exc)
            ) from exc
        print(
            "WARNING: pretrained CORnet-S could not be loaded through torch.hub. "
            "Using SmallCORnetSLike random-weight fallback for engineering validation only. "
            f"Original error: {exc}"
        )
        model = SmallCORnetSLike()
        setattr(model, "feature_source", "small_cornet_s_like_random_fallback")
    model.eval().to(device)
    return model


class FeatureHook:
    def __init__(self, module: nn.Module):
        self.output = None
        self.handle = module.register_forward_hook(self._hook)

    def _hook(self, _module, _inputs, output):
        self.output = output.detach()

    def close(self):
        self.handle.remove()


def resolve_layers(model: nn.Module, layer_names: Iterable[str]) -> OrderedDict[str, nn.Module]:
    named = dict(model.named_modules())
    resolved: OrderedDict[str, nn.Module] = OrderedDict()
    for target in layer_names:
        if target in named:
            resolved[target] = named[target]
            continue
        matches = [name for name in named if name.endswith(target)]
        if not matches:
            raise KeyError(f"Layer '{target}' not found. Available examples: {list(named)[:30]}")
        resolved[target] = named[matches[-1]]
    return resolved


def pooled_flatten(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 4:
        x = torch.nn.functional.adaptive_avg_pool2d(x, 1).flatten(1)
    elif x.ndim > 2:
        x = x.flatten(1)
    return x
