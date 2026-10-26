from __future__ import annotations

import numpy as np


class KalmanFilter:
    """Eight-dimensional constant-velocity filter used by ByteTrack."""

    def __init__(self) -> None:
        self.motion = np.eye(8, dtype=np.float64)
        for index in range(4):
            self.motion[index, index + 4] = 1.0
        self.observation = np.eye(4, 8, dtype=np.float64)
        self.position_weight = 1.0 / 20.0
        self.velocity_weight = 1.0 / 160.0

    def initiate(self, measurement: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        value = np.asarray(measurement, dtype=np.float64)
        if value.shape != (4,) or value[3] <= 0 or not np.isfinite(value).all():
            raise ValueError("Kalman measurement must be finite XYAH with positive height")
        mean = np.r_[value, np.zeros(4, dtype=np.float64)]
        height = value[3]
        deviations = np.asarray(
            [
                2 * self.position_weight * height,
                2 * self.position_weight * height,
                1e-2,
                2 * self.position_weight * height,
                10 * self.velocity_weight * height,
                10 * self.velocity_weight * height,
                1e-5,
                10 * self.velocity_weight * height,
            ],
            dtype=np.float64,
        )
        return mean, np.diag(deviations * deviations)

    def predict(self, mean: np.ndarray, covariance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        height = mean[3]
        deviations = np.asarray(
            [
                self.position_weight * height,
                self.position_weight * height,
                1e-2,
                self.position_weight * height,
                self.velocity_weight * height,
                self.velocity_weight * height,
                1e-5,
                self.velocity_weight * height,
            ],
            dtype=np.float64,
        )
        motion_covariance = np.diag(deviations * deviations)
        predicted_mean = self.motion @ mean
        predicted_covariance = self.motion @ covariance @ self.motion.T + motion_covariance
        return predicted_mean, predicted_covariance

    def project(self, mean: np.ndarray, covariance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        height = mean[3]
        deviations = np.asarray(
            [
                self.position_weight * height,
                self.position_weight * height,
                1e-1,
                self.position_weight * height,
            ],
            dtype=np.float64,
        )
        innovation = np.diag(deviations * deviations)
        return self.observation @ mean, self.observation @ covariance @ self.observation.T + innovation

    def update(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
        measurement: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        projected_mean, projected_covariance = self.project(mean, covariance)
        gain = np.linalg.solve(projected_covariance, covariance[:, :4].T).T
        innovation = np.asarray(measurement, dtype=np.float64) - projected_mean
        updated_mean = mean + gain @ innovation
        updated_covariance = covariance - gain @ projected_covariance @ gain.T
        return updated_mean, updated_covariance
