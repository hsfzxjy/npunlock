from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

import numpy as np
import npunlock as npu

from .generate import generate_kernel_source
from .model import Ensemble, load_ensemble
from .packing import CHUNK_WORDS, LOGICAL_WORDS, Predictions, pack_features, unpack_predictions


class NpuXGBoost:
    """A compile-time-specialized binary XGBoost classifier on ACT-SHAVE."""

    def __init__(
        self,
        model: str | Path | Ensemble,
        *,
        movi_dll_dir: str | Path,
        native_dir: str | Path | None = None,
        timeout_ms: int = 20_000,
    ):
        self.ensemble = model if isinstance(model, Ensemble) else load_ensemble(model)
        self.source = generate_kernel_source(self.ensemble)
        npu.configure(movi_dll_dir=movi_dll_dir)
        input_tensor = npu.input("xgboost_input", shape=(1, LOGICAL_WORDS), dtype="f16")
        output_tensor = npu.custom(
            input_tensor,
            source=self.source,
            carrier="Abs",
            _name="xgboost_output",
        )
        prepared = npu.prepare(
            npu.Graph([input_tensor], [output_tensor], name="xgboost_binary_classifier"),
            native_dir=native_dir,
            build_flags="NPU_TILES=1",
            timeout_ms=timeout_ms,
        )
        if len(prepared.groups) != 1:
            raise RuntimeError(f"XGBoost carrier exposed {len(prepared.groups)} ACT groups, expected one")
        group = prepared.groups[0]
        if group.per_invocation_input_element_counts != (CHUNK_WORDS,) or (
            group.per_invocation_output_element_count != CHUNK_WORDS
        ):
            raise RuntimeError(
                "XGBoost carrier contract changed: "
                f"inputs={group.per_invocation_input_element_counts!r}, "
                f"output={group.per_invocation_output_element_count!r}"
            )
        started = perf_counter()
        self.program = prepared.build(movi_dll_dir=movi_dll_dir)
        self.build_ms = (perf_counter() - started) * 1000.0
        self.carrier_explanation = prepared.explain()
        self.shared_input = self.program.shared_input()
        self.shared_output = self.program.shared_output()

    @property
    def patch_evidence(self) -> dict[str, object]:
        report = json.loads(self.program.patch_reports[0].decode("utf-8"))
        return {
            "schema": report["schema"],
            "input_graph_sha256": report["input_graph_sha256"],
            "shave_elf_sha256": report["shave_elf_sha256"],
            "output_graph_sha256": report["output_graph_sha256"],
            "graph_contract": report["graph_contract"],
            "append": report["append"],
            "validation": report["validation"],
            "patched_target_count": len(report["targets"]),
        }

    @property
    def build_manifest(self) -> dict[str, object] | None:
        payload = self.program.artifact_manifest
        return None if payload is None else json.loads(payload.decode("utf-8"))

    def run(self, features: object, *, shared: bool = True) -> tuple[Predictions, dict[str, float]]:
        values = np.asarray(features, dtype=np.float32)
        started = perf_counter()
        packed = pack_features(values, feature_count=self.ensemble.feature_count)
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
