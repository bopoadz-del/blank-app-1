"""Pure-Python implementation of the YOLOv8 nano network definition."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

try:  # pragma: no cover - optional dependency
    import numpy as np
except Exception:  # pragma: no cover - testing fallback
    np = None  # type: ignore


@dataclass
class LayerConfig:
    name: str
    type: str
    args: Tuple
    kwargs: Dict[str, object]


class Module:
    def forward(self, x: "np.ndarray") -> "np.ndarray":  # pragma: no cover - base API
        raise NotImplementedError


class Conv(Module):
    def __init__(self, weight: "np.ndarray", bias: "np.ndarray", stride: int = 1, padding: int = 0) -> None:
        self.weight = weight
        self.bias = bias
        self.stride = stride
        self.padding = padding

    def forward(self, x: "np.ndarray") -> "np.ndarray":
        if np is None:
            raise RuntimeError("NumPy required")
        out_channels, in_channels, kh, kw = self.weight.shape
        batch, _, h, w = x.shape
        pad_h = h + 2 * self.padding
        pad_w = w + 2 * self.padding
        padded = np.zeros((batch, in_channels, pad_h, pad_w), dtype=x.dtype)
        padded[:, :, self.padding : self.padding + h, self.padding : self.padding + w] = x
        out_h = (pad_h - kh) // self.stride + 1
        out_w = (pad_w - kw) // self.stride + 1
        output = np.zeros((batch, out_channels, out_h, out_w), dtype=x.dtype)
        for b in range(batch):
            for oc in range(out_channels):
                for i in range(out_h):
                    for j in range(out_w):
                        region = padded[
                            b,
                            :,
                            i * self.stride : i * self.stride + kh,
                            j * self.stride : j * self.stride + kw,
                        ]
                        output[b, oc, i, j] = np.sum(region * self.weight[oc]) + self.bias[oc]
        return output


class SiLU(Module):
    def forward(self, x: "np.ndarray") -> "np.ndarray":
        if np is None:
            raise RuntimeError("NumPy required")
        return x * (1.0 / (1.0 + np.exp(-x)))


class Bottleneck(Module):
    def __init__(self, conv1: Conv, conv2: Conv, shortcut: bool = True) -> None:
        self.conv1 = conv1
        self.conv2 = conv2
        self.shortcut = shortcut

    def forward(self, x: "np.ndarray") -> "np.ndarray":
        y = self.conv2.forward(self.conv1.forward(x))
        if self.shortcut:
            y += x
        return y


class C2f(Module):
    def __init__(self, conv1: Conv, conv2: Conv, bottlenecks: List[Bottleneck]) -> None:
        self.conv1 = conv1
        self.conv2 = conv2
        self.bottlenecks = bottlenecks

    def forward(self, x: "np.ndarray") -> "np.ndarray":
        y = self.conv1.forward(x)
        splits = np.array_split(y, 2, axis=1)
        concat = [splits[0], splits[1]]
        for bottleneck in self.bottlenecks:
            splits[1] = bottleneck.forward(splits[1])
            concat.append(splits[1])
        out = np.concatenate(concat, axis=1)
        return self.conv2.forward(out)


class Upsample(Module):
    def __init__(self, scale_factor: int) -> None:
        self.scale_factor = scale_factor

    def forward(self, x: "np.ndarray") -> "np.ndarray":
        if np is None:
            raise RuntimeError("NumPy required")
        batch, channels, height, width = x.shape
        new_height = height * self.scale_factor
        new_width = width * self.scale_factor
        return np.repeat(np.repeat(x, self.scale_factor, axis=2), self.scale_factor, axis=3)


class Concat(Module):
    def forward(self, tensors: Iterable["np.ndarray"], axis: int = 1) -> "np.ndarray":
        if np is None:
            raise RuntimeError("NumPy required")
        return np.concatenate(list(tensors), axis=axis)


class Detect(Module):
    def __init__(self, conv: Conv, anchors: "np.ndarray") -> None:
        self.conv = conv
        self.anchors = anchors

    def forward(self, x: List["np.ndarray"]) -> "np.ndarray":
        if np is None:
            raise RuntimeError("NumPy required")
        outputs = []
        for tensor in x:
            outputs.append(self.conv.forward(tensor))
        return np.concatenate(outputs, axis=1)


class YOLOv8Nano(Module):
    def __init__(self, layers: List[LayerConfig]) -> None:
        self.layers = layers
        self.modules = self._build_layers(layers)

    def _build_layers(self, layers: List[LayerConfig]) -> List[Module]:
        modules: List[Module] = []
        for cfg in layers:
            module = self._build_module(cfg)
            modules.append(module)
        return modules

    def _build_module(self, cfg: LayerConfig) -> Module:
        if cfg.type == "Conv":
            weight, bias = cfg.args
            stride = cfg.kwargs.get("stride", 1)
            padding = cfg.kwargs.get("padding", 0)
            return Conv(weight, bias, stride=stride, padding=padding)
        if cfg.type == "SiLU":
            return SiLU()
        if cfg.type == "Bottleneck":
            return Bottleneck(*cfg.args, **cfg.kwargs)
        if cfg.type == "C2f":
            return C2f(*cfg.args, **cfg.kwargs)
        if cfg.type == "Upsample":
            return Upsample(*cfg.args, **cfg.kwargs)
        if cfg.type == "Concat":
            return Concat()
        if cfg.type == "Detect":
            return Detect(*cfg.args, **cfg.kwargs)
        raise ValueError(f"Unknown module type {cfg.type}")

    def forward(self, x: "np.ndarray") -> "np.ndarray":
        outputs: Dict[str, "np.ndarray"] = {}
        for cfg, module in zip(self.layers, self.modules):
            if cfg.type == "Concat":
                input_tensors = [outputs[name] for name in cfg.args]
                result = module.forward(input_tensors, axis=cfg.kwargs.get("axis", 1))
            elif cfg.type == "Detect":
                input_tensors = [outputs[name] for name in cfg.args]
                result = module.forward(input_tensors)
            else:
                input_tensor = outputs.get(cfg.kwargs.get("from", "input"), x)
                result = module.forward(input_tensor)
            outputs[cfg.name] = result
        return outputs[self.layers[-1].name]


__all__ = ["LayerConfig", "YOLOv8Nano"]
