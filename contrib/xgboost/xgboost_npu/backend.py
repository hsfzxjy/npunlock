from __future__ import annotations

from pathlib import Path
from time import perf_counter

import numpy as np
import npunlock as npu

from .generate import generate_kernel_source
from .model import Ensemble, load_ensemble
from .packing import (
    Predictions,
    SPECIALIZED_ACT_TARGETS,
    TransportSpec,
    pack_features,
    transport_spec,
    unpack_predictions,
)


class NpuXGBoost:
    """A compile-time-specialized binary XGBoost classifier on ACT-SHAVE."""

    def __init__(
        self,
        model: str | Path | Ensemble,
        *,
        movi_dll_dir: str | Path,
        native_dir: str | Path | None = None,
        timeout_ms: int = 20_000,
        transport: str | TransportSpec = "f16",
    ):
        self.ensemble = model if isinstance(model, Ensemble) else load_ensemble(model)
        self.transport = transport_spec(transport)
        self.source = generate_kernel_source(self.ensemble, transport=self.transport)
        input_tensor = npu.input(
            "xgboost_input",
            shape=(1, self.transport.logical_elements),
            dtype=self.transport.dtype,
        )
        output_tensor = npu.custom(
            input_tensor,
            source=self.source,
            carrier="Abs",
            name="xgboost_output",
        )
        prepared = npu.prepare(
            npu.Graph([input_tensor], [output_tensor], name="xgboost_binary_classifier"),
            native_dir=native_dir,
            build_flags="NPU_TILES=1" if self.transport.name == "f16" else "",
            timeout_ms=timeout_ms,
        )
        prepared.require(
            npu.CarrierContract(
                (
                    npu.ActGroupContract(
                        (self.transport.dtype,),
                        self.transport.dtype,
                        (self.transport.chunk_elements,),
                        self.transport.chunk_elements,
                        SPECIALIZED_ACT_TARGETS,
                    ),
                ),
                mapping_indices=((0,),),
            )
        )
        started = perf_counter()
        self.program = prepared.build(movi_dll_dir=movi_dll_dir).require(
            npu.ProgramContract(
                inputs=(("xgboost_input", (1, self.transport.logical_elements), self.transport.dtype),),
                outputs=(("xgboost_output", (1, self.transport.logical_elements), self.transport.dtype),),
                patch_count=1,
                targets_per_patch=(SPECIALIZED_ACT_TARGETS,),
            )
        )
        self.build_ms = (perf_counter() - started) * 1000.0
        self.carrier_explanation = prepared.explain()
        self.shared_input = self.program.shared_input()
        self.shared_output = self.program.shared_output()

    @property
    def patch_evidence(self) -> dict[str, object]:
        return self.program.patch_evidence[0].summary()

    @property
    def build_manifest(self) -> dict[str, object]:
        return self.program.manifest

    def run(self, features: object, *, shared: bool = True) -> tuple[Predictions, dict[str, float]]:
        values = np.asarray(features, dtype=np.float32)
        started = perf_counter()
        packed = pack_features(
            values,
            feature_count=self.ensemble.feature_count,
            transport=self.transport,
        )
        pack_ms = (perf_counter() - started) * 1000.0
        if shared:
            started = perf_counter()
            np.copyto(self.shared_input, packed)
            upload_ms = (perf_counter() - started) * 1000.0
            started = perf_counter()
            outputs = self.program.run(
                {"xgboost_input": self.shared_input},
                outputs={"xgboost_output": self.shared_output},
            )
            graphinfer_wall_ms = (perf_counter() - started) * 1000.0
            started = perf_counter()
            copied = np.array(outputs["xgboost_output"], copy=True)
            download_ms = (perf_counter() - started) * 1000.0
        else:
            upload_ms = 0.0
            download_ms = 0.0
            started = perf_counter()
            outputs = self.program.run({"xgboost_input": packed})
            graphinfer_wall_ms = (perf_counter() - started) * 1000.0
            copied = np.asarray(outputs["xgboost_output"])
        predictions = unpack_predictions(
            copied,
            expected_samples=values.shape[0],
            feature_count=self.ensemble.feature_count,
            transport=self.transport,
        )
        return predictions, {
            "pack_ms": pack_ms,
            "upload_ms": upload_ms,
            "graphinfer_wall_ms": graphinfer_wall_ms,
            "download_ms": download_ms,
        }

    def close(self) -> None:
        self.program.close()

    def __enter__(self) -> "NpuXGBoost":
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.close()
