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
from .runtime import (
    ConnectedRuntimeNpuXGBoost,
    PartitionedRuntimeNpuXGBoost,
    RuntimeNpuXGBoost,
    pack_connected_runtime_features,
    pack_runtime_features,
    pack_runtime_model,
    partitioned_runtime_batch_capacity,
    partition_runtime_ensemble,
)
from .runtime_api import XGBoostRuntime, runtime_backend_kind

__all__ = [
    "Ensemble",
    "FP16_TRANSPORT",
    "FP32_TRANSPORT",
    "Predictions",
    "ConnectedRuntimeNpuXGBoost",
    "PartitionedRuntimeNpuXGBoost",
    "RuntimeNpuXGBoost",
    "Tree",
    "TransportSpec",
    "XGBoostRuntime",
    "generate_kernel_source",
    "load_ensemble",
    "pack_connected_runtime_features",
    "pack_features",
    "pack_runtime_features",
    "pack_runtime_model",
    "partitioned_runtime_batch_capacity",
    "partition_runtime_ensemble",
    "runtime_backend_kind",
    "unpack_predictions",
]
