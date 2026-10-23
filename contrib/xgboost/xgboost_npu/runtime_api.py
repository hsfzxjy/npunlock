from __future__ import annotations

from pathlib import Path

import numpy as np

from .model import Ensemble, load_ensemble
from .packing import Predictions
from .runtime import (
    RUNTIME_MAX_TOTAL_NODES,
    PartitionedRuntimeNpuXGBoost,
    RuntimeNpuXGBoost,
    partitioned_runtime_batch_capacity,
    partition_runtime_ensemble,
    runtime_batch_capacity,
)


def runtime_backend_kind(model: Ensemble) -> str:
    """Return the narrow validated runtime backend required by *model*."""
    if model.node_count <= RUNTIME_MAX_TOTAL_NODES:
        return "single"
    partition_runtime_ensemble(model)
    return "partitioned"


class XGBoostRuntime:
    """Automatically selected and batch-chunked runtime-loaded XGBoost backend."""

    def __init__(
        self,
        model: str | Path | Ensemble,
        *,
        movi_dll_dir: str | Path,
        native_dir: str | Path | None = None,
        timeout_ms: int = 20_000,
    ):
        self._movi_dll_dir = movi_dll_dir
        self._native_dir = native_dir
        self._timeout_ms = timeout_ms
        ensemble = model if isinstance(model, Ensemble) else load_ensemble(model)
        self._backend_kind = runtime_backend_kind(ensemble)
        self._backend = self._create_backend(ensemble, self._backend_kind)
        self.ensemble = ensemble

    def _create_backend(
        self,
        ensemble: Ensemble,
        kind: str,
    ) -> RuntimeNpuXGBoost | PartitionedRuntimeNpuXGBoost:
        backend_type = RuntimeNpuXGBoost if kind == "single" else PartitionedRuntimeNpuXGBoost
        return backend_type(
            ensemble,
            movi_dll_dir=self._movi_dll_dir,
            native_dir=self._native_dir,
            timeout_ms=self._timeout_ms,
        )

    @property
    def backend_kind(self) -> str:
        return self._backend_kind

    @property
    def batch_capacity(self) -> int:
        if self._backend_kind == "single":
            return runtime_batch_capacity(self.ensemble.feature_count)
        return partitioned_runtime_batch_capacity(self.ensemble.feature_count)

    @property
    def patch_evidence(self) -> dict[str, object] | dict[str, dict[str, object]]:
        return self._backend.patch_evidence

    def load_model(self, model: str | Path | Ensemble) -> None:
        ensemble = model if isinstance(model, Ensemble) else load_ensemble(model)
        kind = runtime_backend_kind(ensemble)
        if kind == self._backend_kind:
            self._backend.load_model(ensemble)
        else:
            replacement = self._create_backend(ensemble, kind)
            previous = self._backend
            self._backend = replacement
            self._backend_kind = kind
            previous.close()
        self.ensemble = ensemble

    def run(self, features: object) -> tuple[Predictions, dict[str, float]]:
        values = np.asarray(features, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != self.ensemble.feature_count:
            raise ValueError(f"features must have shape (N, {self.ensemble.feature_count})")
        if np.isinf(values).any():
            raise ValueError("features may contain finite values or NaN, but not infinity")
        if values.shape[0] == 0:
            return Predictions(
                margins=np.empty(0, dtype=np.float32),
                probabilities=np.empty(0, dtype=np.float32),
                classes=np.empty(0, dtype=np.int32),
            ), {
                "pack_ms": 0.0,
                "upload_ms": 0.0,
                "graphinfer_wall_ms": 0.0,
                "download_ms": 0.0,
                "chunk_count": 0.0,
            }

        predictions: list[Predictions] = []
        timing_totals: dict[str, float] = {}
        capacity = self.batch_capacity
        for start in range(0, values.shape[0], capacity):
            chunk_predictions, chunk_timings = self._backend.run(values[start : start + capacity])
            predictions.append(chunk_predictions)
            for name, value in chunk_timings.items():
                timing_totals[name] = timing_totals.get(name, 0.0) + value
        timing_totals["chunk_count"] = float(len(predictions))
        return (
            Predictions(
                margins=np.concatenate([item.margins for item in predictions]),
                probabilities=np.concatenate([item.probabilities for item in predictions]),
                classes=np.concatenate([item.classes for item in predictions]),
            ),
            timing_totals,
        )

    def predict(self, features: object) -> np.ndarray:
        predictions, _ = self.run(features)
        return predictions.classes

    def predict_margin(self, features: object) -> np.ndarray:
        predictions, _ = self.run(features)
        return predictions.margins

    def predict_proba(self, features: object) -> np.ndarray:
        predictions, _ = self.run(features)
        return np.column_stack((np.float32(1.0) - predictions.probabilities, predictions.probabilities))

    def close(self) -> None:
        self._backend.close()

    def __enter__(self) -> "XGBoostRuntime":
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.close()
