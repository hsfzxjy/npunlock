"""Restricted XGBoost-to-ACT-SHAVE demonstration support."""

from .generate import generate_kernel_source
from .model import Ensemble, Tree, load_ensemble
from .packing import Predictions, pack_features, unpack_predictions

__all__ = [
    "Ensemble",
    "Predictions",
    "Tree",
    "generate_kernel_source",
    "load_ensemble",
    "pack_features",
    "unpack_predictions",
]
