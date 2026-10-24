from __future__ import annotations

from pathlib import Path
from time import perf_counter
import json

import numpy as np
import npunlock as npu

from .packing import (
    NMS_CHUNK_WORDS,
    NMS_LOGICAL_WORDS,
    pack_candidates,
    unpack_detections,
    validate_ranked_candidates,
)
from .reference import DetectionBatch


class NpuNms:
    """Compiled npunlock NMS program with copied and shared-buffer execution."""

    def __init__(
        self,
        *,
        kernel_path: str | Path,
        movi_dll_dir: str | Path,
        native_dir: str | Path | None = None,
        timeout_ms: int = 20_000,
    ):
        source = Path(kernel_path).read_bytes()
        npu.configure(movi_dll_dir=movi_dll_dir)
        input_tensor = npu.input("nms_input", shape=(1, NMS_LOGICAL_WORDS), dtype="f16")
        output_tensor = npu.custom(
            input_tensor,
            source=source,
            carrier="Abs",
            _name="nms_output",
        )
        prepared = npu.prepare(
            npu.Graph([input_tensor], [output_tensor], name="yolo11_nms"),
            native_dir=native_dir,
            build_flags="NPU_TILES=1",
            timeout_ms=timeout_ms,
        )
        if len(prepared.groups) != 1:
            raise RuntimeError(f"NMS carrier exposed {len(prepared.groups)} ACT groups, expected one")
        group = prepared.groups[0]
        if group.per_invocation_input_element_counts != (NMS_CHUNK_WORDS,):
            raise RuntimeError(
                "NMS carrier input contract changed: "
                f"{group.per_invocation_input_element_counts!r}"
            )
        if group.per_invocation_output_element_count != NMS_CHUNK_WORDS:
            raise RuntimeError(
                "NMS carrier output contract changed: "
                f"{group.per_invocation_output_element_count!r}"
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

    def run(
        self,
        candidates: DetectionBatch,
        *,
        confidence_threshold: float,
        iou_threshold: float,
        max_detections: int,
        shared: bool = True,
    ) -> tuple[DetectionBatch, dict[str, float]]:
        validate_ranked_candidates(candidates)
        started = perf_counter()
        packed = pack_candidates(
            candidates,
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold,
            max_detections=max_detections,
        )
        pack_ms = (perf_counter() - started) * 1000.0
        if shared:
            started = perf_counter()
            np.copyto(self.shared_input, packed)
            upload_ms = (perf_counter() - started) * 1000.0
            started = perf_counter()
            outputs = self.program.run(
                {"nms_input": self.shared_input},
                outputs={"nms_output": self.shared_output},
            )
            execution_wall_ms = (perf_counter() - started) * 1000.0
            started = perf_counter()
            copied_output = np.array(outputs["nms_output"], copy=True)
            download_ms = (perf_counter() - started) * 1000.0
        else:
            upload_ms = 0.0
            download_ms = 0.0
            started = perf_counter()
            outputs = self.program.run({"nms_input": packed})
            execution_wall_ms = (perf_counter() - started) * 1000.0
            copied_output = np.asarray(outputs["nms_output"])
        return unpack_detections(copied_output), {
            "pack_ms": pack_ms,
            "upload_ms": upload_ms,
            "execution_wall_ms": execution_wall_ms,
            "download_ms": download_ms,
        }

    def close(self) -> None:
        self.program.close()
