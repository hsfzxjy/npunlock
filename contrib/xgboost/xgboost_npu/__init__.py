"""Restricted XGBoost-to-ACT-SHAVE demonstration support."""

from .generate import generate_kernel_source
from .model import Ensemble, Tree, load_ensemble
from .packing import (
    FP16_TRANSPORT,
    FP32_TRANSPORT,
    Predictions,
    TransportSpec,
    pack_features,
    unpack_predictions,
)
from .runtime import RuntimeNpuXGBoost, pack_runtime_features, pack_runtime_model

__all__ = [
    "Ensemble",
    "FP16_TRANSPORT",
    "FP32_TRANSPORT",
    "Predictions",
    "RuntimeNpuXGBoost",
    "Tree",
    "TransportSpec",
    "generate_kernel_source",
    "load_ensemble",
    "pack_features",
    "pack_runtime_features",
    "pack_runtime_model",
    "unpack_predictions",
]
