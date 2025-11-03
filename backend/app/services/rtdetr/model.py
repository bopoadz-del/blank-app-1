"""Minimal RT-DETR model scaffold for TensorRT integration.

This module intentionally avoids heavyweight deep learning dependencies.
Instead it provides a lightweight representation of the architecture,
mirroring tensor shapes and metadata required by the TensorRT builder.

The implementation focuses on debuggability: every transformation logs
its inputs and outputs, enabling rapid tracing when connecting to a real
backend implementation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Tuple

import math
import logging

LOGGER = logging.getLogger(__name__)


@dataclass
class TransformerLayerConfig:
    hidden_dim: int
    num_heads: int
    ff_dim: int
    dropout: float
    activation: str

    def to_mapping(self) -> Dict[str, object]:
        return {
            "hidden_dim": self.hidden_dim,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "dropout": self.dropout,
            "activation": self.activation,
        }


@dataclass
class BackboneStageConfig:
    channels: int
    stride: int
    use_dcn: bool
    num_blocks: int

    def to_mapping(self) -> Dict[str, object]:
        return {
            "channels": self.channels,
            "stride": self.stride,
            "use_dcn": self.use_dcn,
            "num_blocks": self.num_blocks,
        }


@dataclass
class RTDETRArchitecture:
    """Describes the RT-DETR model graph in a framework-agnostic format."""

    name: str
    backbone_stages: List[BackboneStageConfig] = field(default_factory=list)
    encoder_layers: List[TransformerLayerConfig] = field(default_factory=list)
    decoder_layers: List[TransformerLayerConfig] = field(default_factory=list)
    num_queries: int = 300
    num_classes: int = 80
    hidden_dim: int = 256
    positional_encoding_type: str = "sine"
    activation: str = "gelu"

    metadata: MutableMapping[str, object] = field(default_factory=dict)

    def describe(self) -> Dict[str, object]:
        LOGGER.debug("Describing RT-DETR architecture: %s", self.name)
        return {
            "name": self.name,
            "backbone": [stage.to_mapping() for stage in self.backbone_stages],
            "encoder": [layer.to_mapping() for layer in self.encoder_layers],
            "decoder": [layer.to_mapping() for layer in self.decoder_layers],
            "num_queries": self.num_queries,
            "num_classes": self.num_classes,
            "hidden_dim": self.hidden_dim,
            "positional_encoding_type": self.positional_encoding_type,
            "activation": self.activation,
            "metadata": dict(self.metadata),
        }

    def summary(self) -> str:
        description = self.describe()
        lines = ["RT-DETR Architecture Summary:"]
        for key, value in description.items():
            lines.append(f"  {key}: {value}")
        summary = "\n".join(lines)
        LOGGER.debug(summary)
        return summary


def default_architecture() -> RTDETRArchitecture:
    LOGGER.debug("Creating default RT-DETR architecture definition")
    backbone = [
        BackboneStageConfig(channels=64, stride=2, use_dcn=False, num_blocks=3),
        BackboneStageConfig(channels=128, stride=2, use_dcn=False, num_blocks=4),
        BackboneStageConfig(channels=256, stride=2, use_dcn=True, num_blocks=6),
        BackboneStageConfig(channels=512, stride=2, use_dcn=True, num_blocks=3),
    ]
    encoder = [
        TransformerLayerConfig(hidden_dim=256, num_heads=8, ff_dim=1024, dropout=0.1, activation="relu")
        for _ in range(6)
    ]
    decoder = [
        TransformerLayerConfig(hidden_dim=256, num_heads=8, ff_dim=1024, dropout=0.1, activation="relu")
        for _ in range(6)
    ]
    return RTDETRArchitecture(
        name="rtdetr_resnet50",
        backbone_stages=backbone,
        encoder_layers=encoder,
        decoder_layers=decoder,
        num_queries=300,
        num_classes=80,
        hidden_dim=256,
    )


class ArchitectureRegistry:
    """Maintains reusable RT-DETR architecture templates."""

    def __init__(self) -> None:
        self._architectures: Dict[str, RTDETRArchitecture] = {}

    def register(self, architecture: RTDETRArchitecture) -> None:
        LOGGER.debug("Registering architecture %s", architecture.name)
        self._architectures[architecture.name] = architecture

    def get(self, name: str) -> RTDETRArchitecture:
        LOGGER.debug("Fetching architecture %s", name)
        try:
            return self._architectures[name]
        except KeyError as exc:
            raise KeyError(f"Unknown architecture {name!r}") from exc

    def list_available(self) -> List[str]:
        return sorted(self._architectures)


ARCHITECTURES = ArchitectureRegistry()
ARCHITECTURES.register(default_architecture())


@dataclass
class LayerShape:
    name: str
    shape: Tuple[int, ...]

    def to_mapping(self) -> Dict[str, object]:
        return {"name": self.name, "shape": list(self.shape)}


@dataclass
class GraphSummary:
    """Captures tensor shapes for debugging and TensorRT network creation."""

    layers: List[LayerShape] = field(default_factory=list)

    def add(self, name: str, shape: Iterable[int]) -> None:
        shape_tuple = tuple(int(dim) for dim in shape)
        LOGGER.debug("Recording layer %s with shape %s", name, shape_tuple)
        self.layers.append(LayerShape(name=name, shape=shape_tuple))

    def to_mapping(self) -> Dict[str, object]:
        return {"layers": [layer.to_mapping() for layer in self.layers]}

    def __str__(self) -> str:
        lines = ["Graph Summary:"]
        for layer in self.layers:
            lines.append(f"  {layer.name}: {layer.shape}")
        return "\n".join(lines)


def compute_query_positions(num_queries: int, hidden_dim: int) -> List[float]:
    LOGGER.debug("Computing query positions for %s queries", num_queries)
    positions = [math.sin(i / hidden_dim) for i in range(num_queries)]
    return positions


def compute_positional_encoding(height: int, width: int, hidden_dim: int) -> List[float]:
    LOGGER.debug(
        "Generating positional encoding for size (%s, %s) with hidden_dim=%s",
        height,
        width,
        hidden_dim,
    )
    encoding = []
    for y in range(height):
        for x in range(width):
            angle = math.sqrt(x * x + y * y) / max(1, hidden_dim)
            encoding.append(math.sin(angle))
            encoding.append(math.cos(angle))
    return encoding


def build_graph_summary(architecture: RTDETRArchitecture, input_shape: Tuple[int, int, int, int]) -> GraphSummary:
    LOGGER.debug("Building graph summary for %s", architecture.name)
    batch, channels, height, width = input_shape
    summary = GraphSummary()
    summary.add("input", input_shape)
    current_channels = channels
    current_height = height
    current_width = width
    for index, stage in enumerate(architecture.backbone_stages):
        current_channels = int(stage.channels)
        current_height = max(1, current_height // stage.stride)
        current_width = max(1, current_width // stage.stride)
        summary.add(f"backbone_stage_{index}", (batch, current_channels, current_height, current_width))
    summary.add("flatten", (batch, current_channels, current_height * current_width))
    summary.add("project", (batch, architecture.hidden_dim, architecture.num_queries))
    summary.add("decoder", (batch, architecture.num_queries, architecture.hidden_dim))
    summary.add("head", (batch, architecture.num_queries, architecture.num_classes + 4))
    return summary


def attach_metadata(architecture: RTDETRArchitecture, metadata: Mapping[str, object]) -> RTDETRArchitecture:
    LOGGER.debug("Attaching metadata %s to architecture %s", metadata, architecture.name)
    architecture.metadata.update(metadata)
    return architecture


def clone_architecture(architecture: RTDETRArchitecture) -> RTDETRArchitecture:
    LOGGER.debug("Cloning architecture %s", architecture.name)
    return RTDETRArchitecture(
        name=architecture.name,
        backbone_stages=list(architecture.backbone_stages),
        encoder_layers=list(architecture.encoder_layers),
        decoder_layers=list(architecture.decoder_layers),
        num_queries=architecture.num_queries,
        num_classes=architecture.num_classes,
        hidden_dim=architecture.hidden_dim,
        positional_encoding_type=architecture.positional_encoding_type,
        activation=architecture.activation,
        metadata=dict(architecture.metadata),
    )


class ArchitectureDebugger:
    """Utility for logging deep architecture insights during debugging."""

    def __init__(self, architecture: RTDETRArchitecture) -> None:
        self.architecture = architecture

    def dump(self) -> str:
        LOGGER.info("Architecture dump requested for %s", self.architecture.name)
        lines = ["Architecture Debug Dump:"]
        lines.append(self.architecture.summary())
        positions = compute_query_positions(self.architecture.num_queries, self.architecture.hidden_dim)
        lines.append(f"Sample positions: {positions[:5]}")
        return "\n".join(lines)

    def check_consistency(self) -> None:
        LOGGER.debug("Checking architecture consistency for %s", self.architecture.name)
        if self.architecture.encoder_layers and self.architecture.decoder_layers:
            enc_dim = self.architecture.encoder_layers[0].hidden_dim
            dec_dim = self.architecture.decoder_layers[0].hidden_dim
            if enc_dim != dec_dim:
                raise ValueError(
                    "Encoder and decoder hidden dimensions must match for TensorRT export"
                )


def create_architecture_from_config(config: Mapping[str, object]) -> RTDETRArchitecture:
    LOGGER.debug("Creating architecture from config: %s", config)
    name = str(config.get("name", "rtdetr_custom"))
    backbone_configs = config.get("backbone", [])
    encoder_configs = config.get("encoder", [])
    decoder_configs = config.get("decoder", [])
    backbone = [
        BackboneStageConfig(
            channels=int(stage.get("channels", 64)),
            stride=int(stage.get("stride", 2)),
            use_dcn=bool(stage.get("use_dcn", False)),
            num_blocks=int(stage.get("num_blocks", 3)),
        )
        for stage in backbone_configs
    ] or default_architecture().backbone_stages
    encoder = [
        TransformerLayerConfig(
            hidden_dim=int(layer.get("hidden_dim", 256)),
            num_heads=int(layer.get("num_heads", 8)),
            ff_dim=int(layer.get("ff_dim", 1024)),
            dropout=float(layer.get("dropout", 0.1)),
            activation=str(layer.get("activation", "relu")),
        )
        for layer in encoder_configs
    ] or default_architecture().encoder_layers
    decoder = [
        TransformerLayerConfig(
            hidden_dim=int(layer.get("hidden_dim", 256)),
            num_heads=int(layer.get("num_heads", 8)),
            ff_dim=int(layer.get("ff_dim", 1024)),
            dropout=float(layer.get("dropout", 0.1)),
            activation=str(layer.get("activation", "relu")),
        )
        for layer in decoder_configs
    ] or default_architecture().decoder_layers
    architecture = RTDETRArchitecture(
        name=name,
        backbone_stages=backbone,
        encoder_layers=encoder,
        decoder_layers=decoder,
        num_queries=int(config.get("num_queries", 300)),
        num_classes=int(config.get("num_classes", 80)),
        hidden_dim=int(config.get("hidden_dim", 256)),
        positional_encoding_type=str(config.get("positional_encoding_type", "sine")),
        activation=str(config.get("activation", "gelu")),
    )
    metadata = config.get("metadata")
    if isinstance(metadata, Mapping):
        architecture.metadata.update(metadata)
    debugger = ArchitectureDebugger(architecture)
    debugger.check_consistency()
    return architecture


def architecture_to_trt_layers(architecture: RTDETRArchitecture) -> List[Dict[str, object]]:
    LOGGER.debug("Converting architecture %s to TensorRT layer metadata", architecture.name)
    layers = []
    input_channels = 3
    spatial_size = 640
    for index, stage in enumerate(architecture.backbone_stages):
        layers.append(
            {
                "name": f"backbone_conv_{index}",
                "type": "conv",
                "in_channels": input_channels,
                "out_channels": stage.channels,
                "kernel_size": 3,
                "stride": stage.stride,
                "padding": 1,
                "use_dcn": stage.use_dcn,
            }
        )
        input_channels = stage.channels
        spatial_size = max(1, spatial_size // stage.stride)
        layers.append(
            {
                "name": f"backbone_norm_{index}",
                "type": "batchnorm",
                "channels": stage.channels,
                "eps": 1e-5,
            }
        )
        layers.append(
            {
                "name": f"backbone_activation_{index}",
                "type": architecture.activation,
                "channels": stage.channels,
            }
        )
        layers.append(
            {
                "name": f"backbone_output_{index}",
                "type": "tensor",
                "shape": (stage.channels, spatial_size, spatial_size),
            }
        )
    layers.append(
        {
            "name": "encoder_input",
            "type": "reshape",
            "shape": (architecture.hidden_dim, architecture.num_queries),
        }
    )
    for index, layer in enumerate(architecture.encoder_layers):
        layers.append(
            {
                "name": f"encoder_self_attn_{index}",
                "type": "multi_head_attention",
                "hidden_dim": layer.hidden_dim,
                "num_heads": layer.num_heads,
                "dropout": layer.dropout,
            }
        )
        layers.append(
            {
                "name": f"encoder_ffn_{index}",
                "type": "feed_forward",
                "hidden_dim": layer.hidden_dim,
                "ff_dim": layer.ff_dim,
                "activation": layer.activation,
            }
        )
    for index, layer in enumerate(architecture.decoder_layers):
        layers.append(
            {
                "name": f"decoder_cross_attn_{index}",
                "type": "multi_head_attention",
                "hidden_dim": layer.hidden_dim,
                "num_heads": layer.num_heads,
                "dropout": layer.dropout,
            }
        )
        layers.append(
            {
                "name": f"decoder_ffn_{index}",
                "type": "feed_forward",
                "hidden_dim": layer.hidden_dim,
                "ff_dim": layer.ff_dim,
                "activation": layer.activation,
            }
        )
    layers.append(
        {
            "name": "detection_head",
            "type": "linear",
            "in_features": architecture.hidden_dim,
            "out_features": architecture.num_classes + 4,
        }
    )
    return layers


def generate_engine_inputs(config: Mapping[str, object]) -> Dict[str, Tuple[int, ...]]:
    LOGGER.debug("Generating engine inputs from config: %s", config)
    batch_size = int(config.get("batch_size", 1))
    channels = int(config.get("channels", 3))
    height = int(config.get("height", 640))
    width = int(config.get("width", 640))
    return {
        "images": (batch_size, channels, height, width),
        "queries": (batch_size, int(config.get("num_queries", 300)), int(config.get("hidden_dim", 256))),
    }


def generate_engine_outputs(config: Mapping[str, object]) -> Dict[str, Tuple[int, ...]]:
    LOGGER.debug("Generating engine outputs from config: %s", config)
    batch_size = int(config.get("batch_size", 1))
    num_queries = int(config.get("num_queries", 300))
    num_classes = int(config.get("num_classes", 80))
    return {
        "pred_logits": (batch_size, num_queries, num_classes),
        "pred_boxes": (batch_size, num_queries, 4),
    }


def pretty_print_layers(layers: Iterable[Mapping[str, object]]) -> str:
    lines = ["RT-DETR TensorRT Layer Configuration:"]
    for layer in layers:
        lines.append(f"  - {layer['name']}: {layer}")
    summary = "\n".join(lines)
    LOGGER.debug(summary)
    return summary


__all__ = [
    "TransformerLayerConfig",
    "BackboneStageConfig",
    "RTDETRArchitecture",
    "ARCHITECTURES",
    "GraphSummary",
    "build_graph_summary",
    "default_architecture",
    "ArchitectureDebugger",
    "create_architecture_from_config",
    "architecture_to_trt_layers",
    "generate_engine_inputs",
    "generate_engine_outputs",
    "pretty_print_layers",
]
