from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
import openvino as ov

from .packing import NMS_CAPACITY, NMS_CHUNK_WORDS
from .reference import DetectionBatch


@dataclass(frozen=True, slots=True)
class LetterboxInfo:
    original_shape: tuple[int, int]
    input_shape: tuple[int, int]
    scale: float
    pad_left: int
    pad_top: int


def letterbox(image: np.ndarray, size: int = 640) -> tuple[np.ndarray, LetterboxInfo]:
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("input image must be HWC BGR with three channels")
    height, width = image.shape[:2]
    scale = min(size / height, size / width)
    resized_width = round(width * scale)
    resized_height = round(height * scale)
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    horizontal = size - resized_width
    vertical = size - resized_height
    left = round(horizontal / 2 - 0.1)
    right = round(horizontal / 2 + 0.1)
    top = round(vertical / 2 - 0.1)
    bottom = round(vertical / 2 + 0.1)
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    tensor = np.ascontiguousarray(padded[:, :, ::-1].transpose(2, 0, 1), dtype=np.float32)[None] / 255.0
    return tensor, LetterboxInfo((height, width), (size, size), scale, left, top)


def scale_boxes_to_image(boxes: np.ndarray, info: LetterboxInfo) -> np.ndarray:
    scaled = np.asarray(boxes, dtype=np.float32).copy()
    scaled[:, (0, 2)] = (scaled[:, (0, 2)] - info.pad_left) / info.scale
    scaled[:, (1, 3)] = (scaled[:, (1, 3)] - info.pad_top) / info.scale
    height, width = info.original_shape
    scaled[:, (0, 2)] = np.clip(scaled[:, (0, 2)], 0, width)
    scaled[:, (1, 3)] = np.clip(scaled[:, (1, 3)], 0, height)
    return scaled


def make_preselector_model(candidate_count: int, topk: int = NMS_CAPACITY) -> ov.Model:
    if candidate_count <= 0 or not 0 < topk <= candidate_count:
        raise ValueError("preselector dimensions are invalid")
    op = ov.opset13
    predictions = op.parameter([1, 84, candidate_count], np.float32, name="predictions")
    boxes = op.strided_slice(
        predictions,
        [0, 0, 0],
        [1, 4, candidate_count],
        [1, 1, 1],
        [0, 0, 0],
        [0, 0, 0],
    )
    scores = op.strided_slice(
        predictions,
        [0, 4, 0],
        [1, 84, candidate_count],
        [1, 1, 1],
        [0, 0, 0],
        [0, 0, 0],
    )
    class_top = op.topk(scores, 1, axis=1, mode="max", sort="value", stable=True)
    max_scores = op.squeeze(class_top.output(0), [1])
    class_ids = op.squeeze(class_top.output(1), [1])
    candidate_top = op.topk(max_scores, topk, axis=1, mode="max", sort="value", stable=True)
    top_scores = candidate_top.output(0)
    top_indices = candidate_top.output(1)
    top_classes = op.gather(class_ids, top_indices, 1, batch_dims=1)
    transposed_boxes = op.transpose(boxes, [0, 2, 1])
    top_xywh = op.gather(transposed_boxes, top_indices, 1, batch_dims=1)
    center_x, center_y, width, height = op.split(top_xywh, 2, 4).outputs()
    half_width = op.multiply(width, np.float32(0.5))
    half_height = op.multiply(height, np.float32(0.5))
    top_xyxy = op.concat(
        [
            op.subtract(center_x, half_width),
            op.subtract(center_y, half_height),
            op.add(center_x, half_width),
            op.add(center_y, half_height),
        ],
        2,
    )
    top_xyxy.output(0).get_tensor().set_names({"boxes"})
    top_scores.get_tensor().set_names({"scores"})
    top_classes.output(0).get_tensor().set_names({"classes"})
    top_indices.get_tensor().set_names({"indices"})
    return ov.Model(
        [top_xyxy.output(0), top_scores, top_classes.output(0), top_indices],
        [predictions],
        "yolo11_npu_preselector",
    )


def make_yolo_preselector_model(
    core: ov.Core,
    raw_model: str | Path,
    topk: int = NMS_CAPACITY,
) -> ov.Model:
    """Attach the static preselector directly to a raw YOLO OpenVINO model."""

    model = core.read_model(raw_model)
    if len(model.outputs) != 1:
        raise ValueError(f"expected one raw YOLO output, received {len(model.outputs)}")
    predictions = model.get_results()[0].input_value(0)
    shape = predictions.get_shape()
    if len(shape) != 3 or shape[0] != 1 or shape[1] != 84:
        raise ValueError(f"expected static YOLO11 detection output (1, 84, N), received {shape!r}")

    preselector = make_preselector_model(shape[2], topk)
    preselector.get_parameters()[0].output(0).replace(predictions)
    return ov.Model(
        preselector.get_results(),
        model.get_parameters(),
        "yolo11_with_npu_preselector",
    )


def make_yolo_nms_carrier_model(
    core: ov.Core,
    raw_model: str | Path,
    topk: int = NMS_CAPACITY,
) -> ov.Model:
    """Build the static single-graph YOLO/preselection/NMS carrier IR."""

    base = make_yolo_preselector_model(core, raw_model, topk)
    boxes, scores, classes, indices = [result.input_value(0) for result in base.get_results()]
    op = ov.opset13
    header = op.parameter([1, 4], ov.Type.f16, name="nms_header")
    header.output(0).get_tensor().set_names({"nms_header"})
    axis_2 = op.constant(np.asarray([2], dtype=np.int64))
    score_column = op.unsqueeze(op.convert(scores, ov.Type.f16), axis_2)
    class_column = op.unsqueeze(op.convert(classes, ov.Type.f16), axis_2)
    low = op.floor_mod(indices, op.constant(np.asarray(256, dtype=np.int32)))
    high = op.floor(
        op.divide(
            op.convert(indices, ov.Type.f32),
            op.constant(np.asarray(256.0, dtype=np.float32)),
        )
    )
    records = op.concat(
        [
            op.convert(boxes, ov.Type.f16),
            score_column,
            class_column,
            op.unsqueeze(op.convert(low, ov.Type.f16), axis_2),
            op.unsqueeze(op.convert(high, ov.Type.f16), axis_2),
        ],
        2,
    )
    record_words = NMS_CAPACITY * 8
    flat_records = op.reshape(records, op.constant(np.asarray([1, record_words], dtype=np.int64)), False)
    zero_header = op.constant(np.zeros((1, 4), dtype=np.float16))
    chunk = op.concat([header, zero_header, flat_records], 1)
    if chunk.get_output_shape(0) != [1, NMS_CHUNK_WORDS]:
        raise RuntimeError(f"single-graph NMS chunk shape changed: {chunk.get_output_shape(0)!r}")

    # Four chunks produce four 2,048-element ACT invocations on the reviewed
    # NPU3720 compiler. Each invocation therefore sees a complete candidate
    # buffer without an unsupported cross-invocation read.
    logical = op.concat([chunk, chunk, chunk, chunk], 1)
    carrier = op.absolute(logical)
    carrier.set_friendly_name("nms_carrier")
    carrier.output(0).get_tensor().set_names({"nms_output"})
    model = ov.Model(
        [carrier],
        [*base.get_parameters(), header],
        "yolo11_single_graph_nms_carrier",
    )
    for index, node in enumerate(model.get_ordered_ops()):
        node.set_friendly_name(f"single_graph_{index:04d}_{node.get_type_name()}")
    carrier.set_friendly_name("nms_carrier")
    header.set_friendly_name("nms_header")
    for parameter in model.get_parameters():
        if "images" in parameter.output(0).get_tensor().get_names():
            parameter.set_friendly_name("images")
    model.get_results()[0].set_friendly_name("nms_carrier_result")
    return model


class CompiledModel:
    def __init__(self, core: ov.Core, model: str | Path | ov.Model, device: str):
        started = perf_counter()
        self.compiled = core.compile_model(model, device, {"PERFORMANCE_HINT": "LATENCY"})
        self.compile_ms = (perf_counter() - started) * 1000.0
        self.execution_devices = self.compiled.get_property("EXECUTION_DEVICES")
        self.request = self.compiled.create_infer_request()
        self.last_output_copy_ms = 0.0

    def infer(self, input_value: np.ndarray) -> tuple[list[np.ndarray], float]:
        started = perf_counter()
        self.request.infer({0: input_value})
        elapsed_ms = (perf_counter() - started) * 1000.0
        started = perf_counter()
        outputs = [np.array(self.request.get_output_tensor(index).data, copy=True) for index in range(len(self.compiled.outputs))]
        self.last_output_copy_ms = (perf_counter() - started) * 1000.0
        return outputs, elapsed_ms


class NpuYoloPreselector:
    """One OpenVINO NPU workload containing YOLO11 and static preselection."""

    def __init__(
        self,
        core: ov.Core,
        raw_model: str | Path,
        topk: int = NMS_CAPACITY,
    ):
        fused = make_yolo_preselector_model(core, raw_model, topk)
        self.model = CompiledModel(core, fused, "NPU")
        if self.model.execution_devices != "NPU":
            raise RuntimeError(
                "fused YOLO/preselector did not execute exclusively on NPU: "
                f"{self.model.execution_devices!r}"
            )

    def run(self, input_value: np.ndarray) -> tuple[DetectionBatch, float]:
        outputs, elapsed_ms = self.model.infer(np.asarray(input_value, dtype=np.float32))
        return _detection_batch(self.model, outputs), elapsed_ms


def _detection_batch(model: CompiledModel, outputs: list[np.ndarray]) -> DetectionBatch:
    by_name = {
        output.any_name: value
        for output, value in zip(model.compiled.outputs, outputs)
    }
    return DetectionBatch(
        boxes=by_name["boxes"].reshape(-1, 4).astype(np.float32),
        scores=by_name["scores"].reshape(-1).astype(np.float32),
        classes=by_name["classes"].reshape(-1).astype(np.int32),
        indices=by_name["indices"].reshape(-1).astype(np.int32),
    )


def load_image(path: str | Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"failed to load image: {path}")
    return image
