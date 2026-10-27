from __future__ import annotations

import hashlib
from pathlib import Path
from time import perf_counter

import numpy as np
import openvino as ov
import npunlock as npu

from .openvino_backend import make_yolo_nms_carrier_model
from .packing import (
    NMS_CAPACITY,
    NMS_CHUNK_WORDS,
    NMS_LOGICAL_WORDS,
    pack_nms_header,
    unpack_detections,
)
from .reference import DetectionBatch


class SingleGraphBindingError(RuntimeError):
    """The compiled graph does not match a reviewed explicit ACT binding."""


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class SingleGraphYoloNms:
    """Version-bound YOLO + static preselection + ACT-SHAVE NMS graph."""

    def __init__(
        self,
        *,
        raw_model: str | Path,
        kernel_path: str | Path,
        binding_path: str | Path,
        artifact_dir: str | Path,
        movi_dll_dir: str | Path,
        native_dir: str | Path | None = None,
        timeout_ms: int = 60_000,
    ):
        started = perf_counter()
        binding_bytes = Path(binding_path).read_bytes()
        try:
            binding = npu.ReviewedPatchBinding.from_bytes(binding_bytes)
        except ValueError as error:
            raise SingleGraphBindingError(str(error)) from error
        self.binding_sha256 = _sha256(binding_bytes)
        expected_openvino = binding.metadata.get("openvino_version")
        if ov.__version__ != expected_openvino:
            raise SingleGraphBindingError(
                f"binding requires OpenVINO {expected_openvino!r}, installed version is {ov.__version__!r}"
            )

        raw_path = Path(raw_model)
        raw_bytes = raw_path.read_bytes()
        expected_raw_hash = binding.metadata.get("raw_onnx_sha256")
        if _sha256(raw_bytes) != expected_raw_hash:
            raise SingleGraphBindingError("raw YOLO ONNX hash does not match the reviewed binding")

        artifacts = Path(artifact_dir)
        artifacts.mkdir(parents=True, exist_ok=True)
        xml_path = artifacts / "yolo11_single_graph.xml"
        weights_path = artifacts / "yolo11_single_graph.bin"
        model = make_yolo_nms_carrier_model(ov.Core(), raw_path)
        ov.serialize(model, xml_path, weights_path)
        xml = xml_path.read_bytes()
        weights = weights_path.read_bytes()
        self.serialized_ir_hashes = {"xml_sha256": _sha256(xml), "weights_sha256": _sha256(weights)}
        self.serialize_ms = (perf_counter() - started) * 1000.0

        started = perf_counter()
        try:
            prepared = npu.prepare_ir(
                xml,
                weights,
                binding=binding,
                native_dir=native_dir,
                timeout_ms=timeout_ms,
            )
        except ValueError as error:
            raise SingleGraphBindingError(str(error)) from error
        self.ir_compile_ms = (perf_counter() - started) * 1000.0
        started = perf_counter()
        self.program = prepared.build(
            npu.KernelSpec(kernel_path),
            movi_dll_dir=movi_dll_dir,
        )
        self.kernel_build_ms = (perf_counter() - started) * 1000.0
        self.patch_report = self.program.patch_evidence[0]

        started = perf_counter()
        shared_inputs = self.program.shared_inputs()
        self.image = shared_inputs["images"]
        self.header = shared_inputs["nms_header"]
        self.output = self.program.shared_output()
        self.session_create_ms = (perf_counter() - started) * 1000.0
        self.provenance = prepared.provenance.to_json()
        self.native_graph_sha256 = binding.native_graph_sha256

    @property
    def patch_evidence(self) -> dict[str, object]:
        report = self.patch_report
        return {**report.summary(), "targets": [dict(target) for target in report.targets]}

    def run(
        self,
        input_value: np.ndarray,
        *,
        confidence_threshold: float,
        iou_threshold: float,
        max_detections: int,
    ) -> tuple[DetectionBatch, dict[str, float]]:
        value = np.asarray(input_value, dtype=np.float32)
        if value.shape != self.image.shape or not value.flags.c_contiguous:
            value = np.ascontiguousarray(value, dtype=np.float32)
        if value.shape != self.image.shape:
            raise ValueError(f"single-graph input shape must be {self.image.shape}, got {value.shape}")
        header = pack_nms_header(
            NMS_CAPACITY,
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold,
            max_detections=max_detections,
        )

        started = perf_counter()
        np.copyto(self.image, value)
        np.copyto(self.header, header)
        upload_ms = (perf_counter() - started) * 1000.0
        started = perf_counter()
        self.program.run(
            {"images": self.image, "nms_header": self.header},
            outputs={"nms_carrier": self.output},
        )
        execution_wall_ms = (perf_counter() - started) * 1000.0
        started = perf_counter()
        copied_output = np.array(self.output, copy=True)
        download_ms = (perf_counter() - started) * 1000.0
        chunks = copied_output.reshape(4, NMS_CHUNK_WORDS)
        if any(not np.array_equal(chunks[0], chunk) for chunk in chunks[1:]):
            raise RuntimeError("single-graph NMS carrier invocations returned inconsistent outputs")
        copied = copied_output[:, :NMS_LOGICAL_WORDS]
        started = perf_counter()
        detections = unpack_detections(copied)
        unpack_ms = (perf_counter() - started) * 1000.0
        return detections, {
            "upload_ms": upload_ms,
            "execution_wall_ms": execution_wall_ms,
            "download_ms": download_ms,
            "unpack_ms": unpack_ms,
        }

    def close(self) -> None:
        self.program.close()
