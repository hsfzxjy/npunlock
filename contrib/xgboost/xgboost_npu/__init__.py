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

__all__ = [
    "Ensemble",
    "FP16_TRANSPORT",
    "FP32_TRANSPORT",
    "Predictions",
    "Tree",
    "TransportSpec",
    "generate_kernel_source",
    "load_ensemble",
    "pack_features",
    "unpack_predictions",
]
