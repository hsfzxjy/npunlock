from __future__ import annotations

import ctypes
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import openvino as ov
from npunlock._native import NativeLibraries, PatchTargetV2, PatchTensorContract

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


def _shared_array(buffer: object, dtype: np.dtype[object], shape: tuple[int, ...]) -> tuple[object, np.ndarray]:
    size = int(getattr(buffer, "size"))
    owner = (ctypes.c_uint8 * size).from_address(int(getattr(buffer, "address")))
    value = np.frombuffer(owner, dtype=dtype).reshape(shape)
    return owner, value


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
        binding = json.loads(binding_bytes.decode("utf-8"))
        self.binding_sha256 = _sha256(binding_bytes)
        if binding.get("schema") != "npunlock.yolo11.single-graph-binding.v1":
            raise SingleGraphBindingError("single-graph binding has an unsupported schema")
        expected_openvino = binding.get("openvino_version")
        if ov.__version__ != expected_openvino:
            raise SingleGraphBindingError(
                f"binding requires OpenVINO {expected_openvino!r}, installed version is {ov.__version__!r}"
            )

        raw_path = Path(raw_model)
        raw_bytes = raw_path.read_bytes()
        expected_raw_hash = binding.get("raw_onnx_sha256")
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
        expected_ir = binding.get("serialized_ir", {})
        self.serialized_ir_hashes = {"xml_sha256": _sha256(xml), "weights_sha256": _sha256(weights)}
        if self.serialized_ir_hashes != expected_ir:
            raise SingleGraphBindingError("serialized single-graph IR does not match the reviewed binding")
        self.serialize_ms = (perf_counter() - started) * 1000.0

        libraries = NativeLibraries(native_dir)
        build_flags = binding.get("build_flags")
        if not isinstance(build_flags, str):
            raise SingleGraphBindingError("binding build_flags must be a string")
        started = perf_counter()
        compiled = libraries.compile_ir(
            xml,
            weights,
            build_flags=build_flags,
            timeout_ms=timeout_ms,
        )
        self.ir_compile_ms = (perf_counter() - started) * 1000.0
        actual_provenance = {
            "driver_index": compiled.driver_index,
            "device_index": compiled.device_index,
            "driver_version": compiled.driver_version,
            "vendor_id": compiled.vendor_id,
            "device_id": compiled.device_id,
            "graph_extension_version": compiled.graph_extension_version,
            "compiler_version": list(compiled.compiler_version),
        }
        if actual_provenance != binding.get("provenance"):
            raise SingleGraphBindingError(
                "compiled graph provenance does not match the reviewed binding: "
                f"expected {binding.get('provenance')!r}, got {actual_provenance!r}"
            )
        graph_hash = _sha256(compiled.graph_blob)
        if graph_hash != binding.get("native_graph_sha256"):
            raise SingleGraphBindingError(
                "compiled native graph hash does not match the reviewed binding; refusing positional ACT patch"
            )

        targets = []
        for item in binding.get("targets", []):
            tensors = tuple(
                PatchTensorContract(
                    tensor["role"],
                    tensor["index"],
                    tensor["dtype"],
                    tensor["element_count"],
                    tensor["span_bytes"],
                    tensor["observed_flags"],
                )
                for tensor in item["tensors"]
            )
            targets.append(
                PatchTargetV2(
                    item["invocation_index"],
                    item["range_index"],
                    tensors,
                    item.get("target_flags", 0),
                )
            )
        if len(targets) != 4:
            raise SingleGraphBindingError(f"binding contains {len(targets)} targets, expected four")

        started = perf_counter()
        elf = libraries.compile_shave(
            Path(kernel_path).read_bytes(),
            movi_dll_dir=movi_dll_dir,
            timeout_ms=timeout_ms,
        )
        self.shave_compile_ms = (perf_counter() - started) * 1000.0
        started = perf_counter()
        patched = libraries.patch_graph(compiled.graph_blob, elf, targets)
        self.patch_ms = (perf_counter() - started) * 1000.0
        self.patch_report = json.loads(patched.report_json.decode("utf-8"))
        if self.patch_report.get("input_graph_sha256") != graph_hash:
            raise RuntimeError("patch report input hash does not match the reviewed graph")

        started = perf_counter()
        self.session = libraries.create_inference_session(patched.graph_blob, timeout_ms=timeout_ms)
        self.image_buffer = self.session.create_buffer(1 * 3 * 640 * 640 * np.dtype(np.float32).itemsize)
        self.header_buffer = self.session.create_buffer(4 * np.dtype(np.float16).itemsize)
        self.output_buffer = self.session.create_buffer(8192 * np.dtype(np.float16).itemsize)
        self._image_owner, self.image = _shared_array(
            self.image_buffer, np.dtype(np.float32), (1, 3, 640, 640)
        )
        self._header_owner, self.header = _shared_array(self.header_buffer, np.dtype(np.float16), (1, 4))
        self._output_owner, self.output = _shared_array(self.output_buffer, np.dtype(np.float16), (1, 8192))
        self.session_create_ms = (perf_counter() - started) * 1000.0
        self.provenance = actual_provenance
        self.native_graph_sha256 = graph_hash

    @property
    def patch_evidence(self) -> dict[str, object]:
        report = self.patch_report
        return {
            "schema": report["schema"],
            "input_graph_sha256": report["input_graph_sha256"],
            "shave_elf_sha256": report["shave_elf_sha256"],
            "output_graph_sha256": report["output_graph_sha256"],
            "graph_contract": report["graph_contract"],
            "append": report["append"],
            "validation": report["validation"],
            "patched_target_count": len(report["targets"]),
            "targets": report["targets"],
        }

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
        self.session.infer(
            (("images", self.image_buffer), ("nms_header", self.header_buffer)),
            (("nms_carrier", self.output_buffer),),
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
        self.session.close()
