"""Object detection pipeline for robot perception."""

import numpy as np


class ObstacleDetector:
    """Detects obstacles from sensor data (camera, lidar)."""

    def __init__(self, confidence_threshold: float = 0.5):
        self.confidence_threshold = confidence_threshold

    def detect(self, frame: np.ndarray) -> list[dict]:
        """Run detection on a single sensor frame.

        Returns a list of detections, each with keys:
          - bbox: (x1, y1, x2, y2)
          - label: str
          - confidence: float
        """
        raise NotImplementedError("Plug in your detection model here")
