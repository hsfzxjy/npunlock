from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from time import perf_counter

import numpy as np

from .assignment import AssignmentBackend, MAX_COLUMNS, MAX_ROWS
from .kalman import KalmanFilter


class TrackState(IntEnum):
    NEW = 0
    TRACKED = 1
    LOST = 2
    REMOVED = 3


@dataclass(frozen=True, slots=True)
class Detection:
    xyxy: np.ndarray
    score: float
    class_id: int = 0

    def __post_init__(self) -> None:
        box = np.asarray(self.xyxy, dtype=np.float64)
        if box.shape != (4,) or not np.isfinite(box).all() or box[2] <= box[0] or box[3] <= box[1]:
            raise ValueError("detection boxes must be finite XYXY values with positive area")
        if not np.isfinite(self.score) or not 0.0 <= self.score <= 1.0:
            raise ValueError("detection scores must be finite values in [0, 1]")
        object.__setattr__(self, "xyxy", box)


class Track:
    def __init__(self, detection: Detection):
        self._tlwh = self.xyxy_to_tlwh(detection.xyxy)
        self.score = float(detection.score)
        self.class_id = int(detection.class_id)
        self.track_id = 0
        self.state = TrackState.NEW
        self.is_activated = False
        self.frame_id = 0
        self.start_frame = 0
        self.tracklet_length = 0
        self.mean: np.ndarray | None = None
        self.covariance: np.ndarray | None = None
        self.kalman_filter: KalmanFilter | None = None

    @property
    def end_frame(self) -> int:
        return self.frame_id

    @property
    def tlwh(self) -> np.ndarray:
        if self.mean is None:
            return self._tlwh.copy()
        result = self.mean[:4].copy()
        result[2] *= result[3]
        result[:2] -= result[2:] * 0.5
        return result

    @property
    def xyxy(self) -> np.ndarray:
        result = self.tlwh
        result[2:] += result[:2]
        return result

    @staticmethod
    def xyxy_to_tlwh(xyxy: np.ndarray) -> np.ndarray:
        result = np.asarray(xyxy, dtype=np.float64).copy()
        result[2:] -= result[:2]
        return result

    @staticmethod
    def tlwh_to_xyah(tlwh: np.ndarray) -> np.ndarray:
        result = np.asarray(tlwh, dtype=np.float64).copy()
        result[:2] += result[2:] * 0.5
        result[2] /= result[3]
        return result

    def predict(self) -> None:
        if self.mean is None or self.covariance is None or self.kalman_filter is None:
            return
        mean = self.mean.copy()
        if self.state != TrackState.TRACKED:
            mean[7] = 0.0
        self.mean, self.covariance = self.kalman_filter.predict(mean, self.covariance)

    def activate(self, kalman_filter: KalmanFilter, frame_id: int, track_id: int) -> None:
        self.kalman_filter = kalman_filter
        self.track_id = track_id
        self.mean, self.covariance = kalman_filter.initiate(self.tlwh_to_xyah(self._tlwh))
        self.tracklet_length = 0
        self.state = TrackState.TRACKED
        self.is_activated = frame_id == 1
        self.frame_id = frame_id
        self.start_frame = frame_id

    def _consume_detection(self, detection: "Track", frame_id: int) -> None:
        if self.mean is None or self.covariance is None or self.kalman_filter is None:
            raise RuntimeError("cannot update an uninitialized track")
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean,
            self.covariance,
            self.tlwh_to_xyah(detection.tlwh),
        )
        self.state = TrackState.TRACKED
        self.is_activated = True
        self.frame_id = frame_id
        self.score = detection.score
        self.class_id = detection.class_id

    def update(self, detection: "Track", frame_id: int) -> None:
        self.tracklet_length += 1
        self._consume_detection(detection, frame_id)

    def reactivate(self, detection: "Track", frame_id: int) -> None:
        self.tracklet_length = 0
        self._consume_detection(detection, frame_id)

    def mark_lost(self) -> None:
        self.state = TrackState.LOST

    def mark_removed(self) -> None:
        self.state = TrackState.REMOVED


def _boxes(items: list[Track]) -> np.ndarray:
    if not items:
        return np.empty((0, 4), dtype=np.float64)
    return np.asarray([item.xyxy for item in items], dtype=np.float64)


def iou_distance(left: list[Track], right: list[Track]) -> np.ndarray:
    left_boxes = _boxes(left)
    right_boxes = _boxes(right)
    if not left or not right:
        return np.empty((len(left), len(right)), dtype=np.float32)
    top_left = np.maximum(left_boxes[:, None, :2], right_boxes[None, :, :2])
    bottom_right = np.minimum(left_boxes[:, None, 2:], right_boxes[None, :, 2:])
    intersection_size = np.maximum(bottom_right - top_left, 0.0)
    intersection = intersection_size[:, :, 0] * intersection_size[:, :, 1]
    left_area = np.prod(left_boxes[:, 2:] - left_boxes[:, :2], axis=1)
    right_area = np.prod(right_boxes[:, 2:] - right_boxes[:, :2], axis=1)
    union = left_area[:, None] + right_area[None, :] - intersection
    return np.asarray(1.0 - intersection / np.maximum(union, np.finfo(np.float64).tiny), dtype=np.float32)


def fuse_detection_scores(costs: np.ndarray, detections: list[Track]) -> np.ndarray:
    if costs.size == 0:
        return costs
    scores = np.asarray([detection.score for detection in detections], dtype=np.float32)
    return np.asarray(1.0 - (1.0 - costs) * scores[None, :], dtype=np.float32)


def _join(first: list[Track], second: list[Track]) -> list[Track]:
    result: list[Track] = []
    seen: set[int] = set()
    for track in first + second:
        if track.track_id not in seen:
            seen.add(track.track_id)
            result.append(track)
    return result


def _subtract(first: list[Track], second: list[Track]) -> list[Track]:
    removed = {track.track_id for track in second}
    return [track for track in first if track.track_id not in removed]


def _remove_duplicates(tracked: list[Track], lost: list[Track]) -> tuple[list[Track], list[Track]]:
    costs = iou_distance(tracked, lost)
    duplicate_tracked: set[int] = set()
    duplicate_lost: set[int] = set()
    for tracked_index, lost_index in np.argwhere(costs < 0.15).tolist():
        tracked_age = tracked[tracked_index].frame_id - tracked[tracked_index].start_frame
        lost_age = lost[lost_index].frame_id - lost[lost_index].start_frame
        if tracked_age > lost_age:
            duplicate_lost.add(lost_index)
        else:
            duplicate_tracked.add(tracked_index)
    return (
        [track for index, track in enumerate(tracked) if index not in duplicate_tracked],
        [track for index, track in enumerate(lost) if index not in duplicate_lost],
    )


class ByteTracker:
    """Original ByteTrack lifecycle with every LAP call delegated to one backend."""

    def __init__(
        self,
        assignment: AssignmentBackend,
        *,
        frame_rate: float = 30.0,
        track_high_threshold: float = 0.5,
        track_low_threshold: float = 0.1,
        new_track_threshold: float = 0.6,
        first_match_threshold: float = 0.8,
        second_match_threshold: float = 0.5,
        unconfirmed_match_threshold: float = 0.7,
        track_buffer: int = 30,
        fuse_score: bool = True,
    ):
        if frame_rate <= 0 or track_buffer < 0:
            raise ValueError("frame rate must be positive and track buffer non-negative")
        thresholds = (
            track_high_threshold,
            track_low_threshold,
            new_track_threshold,
            first_match_threshold,
            second_match_threshold,
            unconfirmed_match_threshold,
        )
        if any(not np.isfinite(value) or value < 0 or value > 1 for value in thresholds):
            raise ValueError("ByteTrack thresholds must be finite values in [0, 1]")
        if track_low_threshold >= track_high_threshold:
            raise ValueError("track_low_threshold must be lower than track_high_threshold")
        self.assignment = assignment
        self.track_high_threshold = track_high_threshold
        self.track_low_threshold = track_low_threshold
        self.new_track_threshold = new_track_threshold
        self.first_match_threshold = first_match_threshold
        self.second_match_threshold = second_match_threshold
        self.unconfirmed_match_threshold = unconfirmed_match_threshold
        self.max_time_lost = int(frame_rate / 30.0 * track_buffer)
        self.fuse_score = fuse_score
        self.frame_id = 0
        self.next_track_id = 1
        self.tracked_tracks: list[Track] = []
        self.lost_tracks: list[Track] = []
        self.removed_tracks: list[Track] = []
        self.kalman_filter = KalmanFilter()
        self.last_assignment_timings: list[dict[str, float]] = []

    def _assign(self, costs: np.ndarray, threshold: float):
        if costs.shape[0] > MAX_ROWS or costs.shape[1] > MAX_COLUMNS:
            raise ValueError(
                f"ByteTrack association shape {costs.shape!r} exceeds NPU capacity "
                f"{MAX_ROWS}x{MAX_COLUMNS}"
            )
        result = self.assignment.assign(costs, threshold)
        self.last_assignment_timings.append(result.timings)
        return result

    def update(self, detections: list[Detection]) -> list[Track]:
        started = perf_counter()
        self.frame_id += 1
        self.last_assignment_timings = []
        activated: list[Track] = []
        refound: list[Track] = []
        newly_lost: list[Track] = []
        newly_removed: list[Track] = []

        high = [Track(detection) for detection in detections if detection.score > self.track_high_threshold]
        low = [
            Track(detection)
            for detection in detections
            if self.track_low_threshold < detection.score < self.track_high_threshold
        ]
        if len(high) > MAX_COLUMNS or len(low) > MAX_COLUMNS:
            raise ValueError("detection count exceeds the 16-column assignment capacity")

        unconfirmed = [track for track in self.tracked_tracks if not track.is_activated]
        confirmed = [track for track in self.tracked_tracks if track.is_activated]
        pool = _join(confirmed, self.lost_tracks)
        for track in pool:
            track.predict()

        first_costs = iou_distance(pool, high)
        if self.fuse_score:
            first_costs = fuse_detection_scores(first_costs, high)
        first = self._assign(first_costs, self.first_match_threshold)
        for track_index, detection_index in first.matches.tolist():
            track = pool[track_index]
            detection = high[detection_index]
            if track.state == TrackState.TRACKED:
                track.update(detection, self.frame_id)
                activated.append(track)
            else:
                track.reactivate(detection, self.frame_id)
                refound.append(track)

        remaining_tracked = [
            pool[index]
            for index in first.unmatched_rows.tolist()
            if pool[index].state == TrackState.TRACKED
        ]
        second = self._assign(iou_distance(remaining_tracked, low), self.second_match_threshold)
        for track_index, detection_index in second.matches.tolist():
            track = remaining_tracked[track_index]
            track.update(low[detection_index], self.frame_id)
            activated.append(track)
        for track_index in second.unmatched_rows.tolist():
            track = remaining_tracked[track_index]
            if track.state != TrackState.LOST:
                track.mark_lost()
                newly_lost.append(track)

        remaining_high = [high[index] for index in first.unmatched_columns.tolist()]
        unconfirmed_costs = iou_distance(unconfirmed, remaining_high)
        if self.fuse_score:
            unconfirmed_costs = fuse_detection_scores(unconfirmed_costs, remaining_high)
        third = self._assign(unconfirmed_costs, self.unconfirmed_match_threshold)
        for track_index, detection_index in third.matches.tolist():
            track = unconfirmed[track_index]
            track.update(remaining_high[detection_index], self.frame_id)
            activated.append(track)
        for track_index in third.unmatched_rows.tolist():
            track = unconfirmed[track_index]
            track.mark_removed()
            newly_removed.append(track)

        for detection_index in third.unmatched_columns.tolist():
            track = remaining_high[detection_index]
            if track.score < self.new_track_threshold:
                continue
            track.activate(self.kalman_filter, self.frame_id, self.next_track_id)
            self.next_track_id += 1
            activated.append(track)

        for track in self.lost_tracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                newly_removed.append(track)

        self.tracked_tracks = [track for track in self.tracked_tracks if track.state == TrackState.TRACKED]
        self.tracked_tracks = _join(self.tracked_tracks, activated)
        self.tracked_tracks = _join(self.tracked_tracks, refound)
        self.lost_tracks = _subtract(self.lost_tracks, self.tracked_tracks)
        self.lost_tracks.extend(newly_lost)
        self.lost_tracks = _subtract(self.lost_tracks, newly_removed)
        self.removed_tracks.extend(newly_removed)
        self.tracked_tracks, self.lost_tracks = _remove_duplicates(self.tracked_tracks, self.lost_tracks)
        self.last_update_ms = (perf_counter() - started) * 1000.0
        return [track for track in self.tracked_tracks if track.is_activated]
