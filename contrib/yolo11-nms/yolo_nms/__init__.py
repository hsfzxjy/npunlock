"""YOLO11 NPU-side NMS demonstration helpers."""

from .packing import NMS_CAPACITY, pack_candidates, unpack_detections
from .reference import DetectionBatch, reference_nms

__all__ = [
    "DetectionBatch",
    "NMS_CAPACITY",
    "pack_candidates",
    "reference_nms",
    "unpack_detections",
]
