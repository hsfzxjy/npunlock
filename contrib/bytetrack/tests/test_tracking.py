from __future__ import annotations

import unittest

import numpy as np

from bytetrack_npu.assignment import scipy_assignment
from bytetrack_npu.kalman import KalmanFilter
from bytetrack_npu.tracker import ByteTracker, Detection, TrackState


class RecordingAssignment:
    def __init__(self) -> None:
        self.calls: list[tuple[tuple[int, int], float]] = []

    def assign(self, costs: object, threshold: float, *, forbidden: object | None = None):
        matrix = np.asarray(costs, dtype=np.float32)
        self.calls.append((matrix.shape, threshold))
        return scipy_assignment(matrix, threshold, forbidden=forbidden)


def detection(x: float, score: float = 0.9, class_id: int = 0) -> Detection:
    return Detection(np.asarray([x, 10.0, x + 20.0, 50.0]), score, class_id)


class TrackingTests(unittest.TestCase):
    def test_kalman_cholesky_update_matches_reference_solve(self) -> None:
        kalman = KalmanFilter()
        mean, covariance = kalman.initiate(np.asarray([40.0, 30.0, 0.5, 80.0]))
        mean, covariance = kalman.predict(mean, covariance)
        measurement = np.asarray([41.0, 29.0, 0.52, 79.0])
        projected_mean, projected_covariance = kalman.project(mean, covariance)
        reference_gain = np.linalg.solve(projected_covariance, covariance[:, :4].T).T
        reference_mean = mean + reference_gain @ (measurement - projected_mean)
        reference_covariance = (
            covariance - reference_gain @ projected_covariance @ reference_gain.T
        )

        updated_mean, updated_covariance = kalman.update(mean, covariance, measurement)

        np.testing.assert_allclose(updated_mean, reference_mean, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(updated_covariance, reference_covariance, rtol=1e-12, atol=1e-12)

    def test_persistent_ids_and_all_three_assignment_stages(self) -> None:
        assignment = RecordingAssignment()
        tracker = ByteTracker(assignment)
        first = tracker.update([detection(10), detection(100)])
        self.assertEqual([track.track_id for track in first], [1, 2])
        self.assertEqual(len(assignment.calls), 3)
        second = tracker.update([detection(12), detection(98)])
        self.assertEqual([track.track_id for track in second], [1, 2])
        self.assertEqual(len(assignment.calls), 6)
        self.assertTrue(all(track.state == TrackState.TRACKED for track in second))

    def test_low_confidence_second_association_keeps_identity(self) -> None:
        assignment = RecordingAssignment()
        tracker = ByteTracker(assignment)
        self.assertEqual(tracker.update([detection(10, 0.9)])[0].track_id, 1)
        result = tracker.update([detection(11, 0.2)])
        self.assertEqual([track.track_id for track in result], [1])
        self.assertEqual(assignment.calls[-3][0], (1, 0))
        self.assertEqual(assignment.calls[-2][0], (1, 1))
        self.assertEqual(assignment.calls[-1][0], (0, 0))

    def test_unconfirmed_track_requires_a_followup_match(self) -> None:
        tracker = ByteTracker(RecordingAssignment())
        self.assertEqual(tracker.update([]), [])
        self.assertEqual(tracker.update([detection(20)]), [])
        result = tracker.update([detection(21)])
        self.assertEqual([track.track_id for track in result], [1])
        self.assertTrue(result[0].is_activated)

    def test_lost_track_reactivates_with_same_id(self) -> None:
        tracker = ByteTracker(RecordingAssignment(), track_buffer=3)
        self.assertEqual(tracker.update([detection(30)])[0].track_id, 1)
        self.assertEqual(tracker.update([]), [])
        result = tracker.update([detection(31)])
        self.assertEqual([track.track_id for track in result], [1])

    def test_new_track_threshold_and_removal(self) -> None:
        tracker = ByteTracker(
            RecordingAssignment(),
            track_high_threshold=0.5,
            new_track_threshold=0.7,
            track_buffer=1,
        )
        self.assertEqual(tracker.update([detection(0, 0.6)]), [])
        self.assertEqual(tracker.tracked_tracks, [])
        self.assertEqual(tracker.update([detection(0, 0.9)]), [])
        self.assertEqual(tracker.update([detection(1, 0.9)])[0].track_id, 1)
        tracker.update([])
        tracker.update([])
        self.assertTrue(any(track.track_id == 1 for track in tracker.removed_tracks))

    def test_rejects_invalid_detections_and_capacity(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive area"):
            Detection(np.asarray([1, 1, 1, 2]), 0.5)
        tracker = ByteTracker(RecordingAssignment())
        with self.assertRaisesRegex(ValueError, "16-column"):
            tracker.update([detection(float(index) * 30.0) for index in range(17)])


if __name__ == "__main__":
    unittest.main()
