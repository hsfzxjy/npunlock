from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import platform
import statistics
import sys
from time import perf_counter


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
for source_root in (REPOSITORY_ROOT / "contrib" / "bytetrack", REPOSITORY_ROOT / "contrib" / "yolo11-nms"):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

import cv2
import numpy as np
import openvino as ov

from bytetrack_npu.assignment import CpuLapjvAssignment, NpuAssignment
from bytetrack_npu.tracker import ByteTracker, Detection
from yolo_nms.openvino_backend import NpuYoloPreselector, letterbox, scale_boxes_to_image
from yolo_nms.reference import DetectionBatch, reference_nms


def summary(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0, "median": 0.0, "p95": 0.0, "mean": 0.0}
    ordered = sorted(values)
    p95_index = min(len(ordered) - 1, int(np.ceil(len(ordered) * 0.95)) - 1)
    return {
        "count": len(values),
        "median": statistics.median(values),
        "p95": ordered[p95_index],
        "mean": statistics.fmean(values),
    }


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Track a real video with ByteTrack and NPU-side assignment")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("contrib/bytetrack/outputs"))
    parser.add_argument("--movi-dll-dir", type=Path, required=True)
    parser.add_argument("--native-dir", type=Path)
    parser.add_argument("--max-frames", type=int, default=300)
    parser.add_argument("--class-id", type=int, action="append", default=None)
    parser.add_argument("--track-high-threshold", type=float, default=0.5)
    parser.add_argument("--track-low-threshold", type=float, default=0.1)
    parser.add_argument("--new-track-threshold", type=float, default=0.6)
    parser.add_argument("--match-threshold", type=float, default=0.8)
    parser.add_argument("--nms-iou", type=float, default=0.7)
    parser.add_argument("--skip-cpu-reference", action="store_true")
    return parser.parse_args()


def filtered_detections(
    candidates: DetectionBatch,
    classes: set[int],
    *,
    confidence_threshold: float,
    iou_threshold: float,
) -> DetectionBatch:
    selected = np.asarray(
        [index for index, class_id in enumerate(candidates.classes.tolist()) if class_id in classes],
        dtype=np.int64,
    )
    filtered = DetectionBatch(
        boxes=candidates.boxes[selected],
        scores=candidates.scores[selected],
        classes=candidates.classes[selected],
        indices=candidates.indices[selected],
    )
    return reference_nms(
        filtered,
        confidence_threshold=confidence_threshold,
        iou_threshold=iou_threshold,
        max_detections=16,
    )


def make_tracker(assignment: object, fps: float, options: argparse.Namespace) -> ByteTracker:
    return ByteTracker(
        assignment,
        frame_rate=fps,
        track_high_threshold=options.track_high_threshold,
        track_low_threshold=options.track_low_threshold,
        new_track_threshold=options.new_track_threshold,
        first_match_threshold=options.match_threshold,
    )


def main() -> int:
    options = arguments()
    if options.max_frames <= 0:
        raise ValueError("max-frames must be positive")
    raw_model = options.model_dir.resolve() / "yolo11n_raw.onnx"
    if not raw_model.is_file():
        raise FileNotFoundError(f"raw YOLO11 model not found: {raw_model}")
    capture = cv2.VideoCapture(str(options.video.resolve()))
    if not capture.isOpened():
        raise ValueError(f"failed to open video: {options.video}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
    options.output_dir.mkdir(parents=True, exist_ok=True)
    video_output = options.output_dir / "bytetrack_npu.mp4"
    writer = cv2.VideoWriter(
        str(video_output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"failed to open output video: {video_output}")

    core = ov.Core()
    detector = NpuYoloPreselector(core, raw_model)
    assignment = NpuAssignment(
        movi_dll_dir=options.movi_dll_dir,
        native_dir=options.native_dir,
    )
    tracker = make_tracker(assignment, fps, options)
    cpu_tracker = None if options.skip_cpu_reference else make_tracker(CpuLapjvAssignment(), fps, options)
    classes = set(options.class_id or [0])
    timings: dict[str, list[float]] = defaultdict(list)
    track_frame_counts: dict[int, int] = defaultdict(int)
    cpu_agreement_frames = 0
    processed = 0
    try:
        while processed < options.max_frames:
            ok, frame = capture.read()
            if not ok:
                break
            frame_started = perf_counter()
            input_value, letterbox_info = letterbox(frame)
            candidates, detector_ms = detector.run(input_value)
            detections = filtered_detections(
                candidates,
                classes,
                confidence_threshold=options.track_low_threshold,
                iou_threshold=options.nms_iou,
            )
            boxes = scale_boxes_to_image(detections.boxes, letterbox_info)
            tracker_inputs = [
                Detection(box, float(score), int(class_id))
                for box, score, class_id in zip(boxes, detections.scores, detections.classes)
            ]
            track_started = perf_counter()
            tracks = tracker.update(tracker_inputs)
            track_ms = (perf_counter() - track_started) * 1000.0
            if cpu_tracker is not None:
                cpu_tracks = cpu_tracker.update(tracker_inputs)
                npu_state = sorted((track.track_id, track.state, np.round(track.xyxy, 4).tolist()) for track in tracks)
                cpu_state = sorted((track.track_id, track.state, np.round(track.xyxy, 4).tolist()) for track in cpu_tracks)
                if npu_state != cpu_state:
                    raise RuntimeError(f"CPU/NPU tracker state diverged at frame {processed + 1}")
                cpu_agreement_frames += 1
            for track in tracks:
                track_frame_counts[track.track_id] += 1
                x1, y1, x2, y2 = np.rint(track.xyxy).astype(np.int32).tolist()
                color = (
                    64 + (track.track_id * 67) % 192,
                    64 + (track.track_id * 97) % 192,
                    64 + (track.track_id * 131) % 192,
                )
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    frame,
                    f"ID {track.track_id} {track.score:.2f}",
                    (x1, max(16, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2,
                    cv2.LINE_AA,
                )
            writer.write(frame)
            processed += 1
            timings["detector_ms"].append(detector_ms)
            timings["tracker_ms"].append(track_ms)
            timings["frame_wall_ms"].append((perf_counter() - frame_started) * 1000.0)
            for assignment_timing in tracker.last_assignment_timings:
                for name, value in assignment_timing.items():
                    timings[f"assignment_{name}"].append(value)
    finally:
        capture.release()
        writer.release()
        assignment.close()

    report = {
        "schema": "npunlock.bytetrack.demo.v1",
        "platform": platform.platform(),
        "python": sys.version,
        "openvino": ov.__version__,
        "video": str(options.video.resolve()),
        "output_video": str(video_output.resolve()),
        "frames": processed,
        "fps": fps,
        "classes": sorted(classes),
        "assignment": {
            "device": "NPU3720 ACT-SHAVE",
            "algorithm": "bounded shortest augmenting path",
            "capacity": [16, 16],
            "build_ms": assignment.build_ms,
            "source_sha256": assignment.source_sha256,
            "target_count": assignment.target_count,
            "patch_evidence": assignment.patch_evidence,
        },
        "cpu_reference": {
            "backend": "lap.lapjv",
            "enabled": cpu_tracker is not None,
            "agreement_frames": cpu_agreement_frames,
        },
        "persistent_tracks": [
            {"track_id": track_id, "visible_frames": count}
            for track_id, count in sorted(track_frame_counts.items())
        ],
        "timings_ms": {name: summary(values) for name, values in timings.items()},
    }
    report_path = options.output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
