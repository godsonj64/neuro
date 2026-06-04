from __future__ import annotations

from collections import OrderedDict
from typing import Iterable

import torch
from torch import nn


class Identity(nn.Module):
    def forward(self, x):
        return x


def load_cornet_s(device: str | torch.device = "cpu") -> nn.Module:
    """Load pretrained CORnet-S.

    Preferred path uses torch.hub from dicarlolab/CORnet. If that fails, the
    function raises a clear error rather than silently returning random weights.
    """
    try:
        model = torch.hub.load("dicarlolab/CORnet", "cornet_s", pretrained=True, trust_repo=True)
    except Exception as exc:
        raise RuntimeError(
            "Could not load pretrained CORnet-S through torch.hub. "
            "Check internet access in Colab, then retry. Original error: " + str(exc)
        ) from exc
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
