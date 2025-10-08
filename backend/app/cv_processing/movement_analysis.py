"""Movement analysis utilities for adaptive frame sampling.

This module provides utilities for calculating movement metrics between frames,
which are used to decide when to sample frames for OCR processing.
"""

import math
from typing import List, Tuple

from app.cv_models import BoundingBox, PalletTrack


class MovementAnalyzer:
    """Calculate movement metrics between frames.

    Provides static methods for analyzing pallet movement, size changes,
    and velocity estimation to support intelligent frame sampling.

    Example:
        >>> movement = MovementAnalyzer.calculate_movement(bbox1, bbox2)
        >>> size_change = MovementAnalyzer.calculate_size_change(bbox1, bbox2)
        >>> velocity = MovementAnalyzer.estimate_velocity(track, n_recent=5)
    """

    @staticmethod
    def calculate_movement(bbox1: BoundingBox, bbox2: BoundingBox) -> float:
        """Calculate Euclidean distance between bbox centers.

        Args:
            bbox1: First bounding box
            bbox2: Second bounding box

        Returns:
            Distance in pixels between centers

        Example:
            >>> bbox1 = BoundingBox(x1=100, y1=200, x2=300, y2=400, confidence=0.9)
            >>> bbox2 = BoundingBox(x1=150, y1=250, x2=350, y2=450, confidence=0.9)
            >>> distance = MovementAnalyzer.calculate_movement(bbox1, bbox2)
            >>> print(f"Moved {distance:.1f} pixels")
        """
        center1_x, center1_y = bbox1.center()
        center2_x, center2_y = bbox2.center()

        distance = math.sqrt(
            (center2_x - center1_x) ** 2 +
            (center2_y - center1_y) ** 2
        )

        return distance

    @staticmethod
    def calculate_size_change(bbox1: BoundingBox, bbox2: BoundingBox) -> float:
        """Calculate relative size change between bboxes.

        Measures how much the bounding box area has changed, normalized
        by the original size. Useful for detecting when pallet is approaching
        or retreating from camera.

        Args:
            bbox1: First bounding box (reference)
            bbox2: Second bounding box (comparison)

        Returns:
            Relative size change (0-1 scale, where 0.2 = 20% change)

        Example:
            >>> bbox1 = BoundingBox(x1=100, y1=200, x2=300, y2=400, confidence=0.9)
            >>> bbox2 = BoundingBox(x1=100, y1=200, x2=350, y2=450, confidence=0.9)
            >>> change = MovementAnalyzer.calculate_size_change(bbox1, bbox2)
            >>> print(f"Size changed by {change*100:.1f}%")
        """
        area1 = bbox1.area()
        area2 = bbox2.area()

        if area1 == 0:
            return 1.0 if area2 > 0 else 0.0

        # Calculate absolute relative change
        size_change = abs(area2 - area1) / area1

        return size_change

    @staticmethod
    def estimate_velocity(track: PalletTrack, n_recent: int = 5) -> float:
        """Estimate track velocity from recent detections.

        Higher velocity indicates the pallet is moving quickly, which means
        we should sample more frequently to capture different angles and
        perspectives.

        Args:
            track: PalletTrack object with detection history
            n_recent: Number of recent detections to use (default: 5)

        Returns:
            Average movement in pixels per frame

        Example:
            >>> velocity = MovementAnalyzer.estimate_velocity(track, n_recent=5)
            >>> if velocity > 20:
            ...     print("Fast moving pallet - sample more frequently")
        """
        if len(track.detections) < 2:
            return 0.0

        # Use only the most recent n detections
        recent_detections = track.detections[-n_recent:]

        if len(recent_detections) < 2:
            return 0.0

        # Calculate total distance traveled
        total_distance = 0.0
        for i in range(1, len(recent_detections)):
            bbox1 = recent_detections[i - 1].bbox
            bbox2 = recent_detections[i].bbox
            distance = MovementAnalyzer.calculate_movement(bbox1, bbox2)
            total_distance += distance

        # Calculate average velocity (pixels per frame)
        num_transitions = len(recent_detections) - 1
        avg_velocity = total_distance / num_transitions if num_transitions > 0 else 0.0

        return avg_velocity

    @staticmethod
    def calculate_movement_vector(bbox1: BoundingBox, bbox2: BoundingBox) -> Tuple[float, float]:
        """Calculate movement vector (dx, dy) between bbox centers.

        Useful for understanding direction of movement, not just magnitude.

        Args:
            bbox1: First bounding box
            bbox2: Second bounding box

        Returns:
            Tuple of (dx, dy) in pixels

        Example:
            >>> dx, dy = MovementAnalyzer.calculate_movement_vector(bbox1, bbox2)
            >>> if abs(dx) > abs(dy):
            ...     print("Mostly horizontal movement")
        """
        center1_x, center1_y = bbox1.center()
        center2_x, center2_y = bbox2.center()

        dx = center2_x - center1_x
        dy = center2_y - center1_y

        return (dx, dy)

    @staticmethod
    def is_significant_movement(
        bbox1: BoundingBox,
        bbox2: BoundingBox,
        threshold: float = 50.0
    ) -> bool:
        """Check if movement between bboxes exceeds threshold.

        Convenience method for quick movement significance checks.

        Args:
            bbox1: First bounding box
            bbox2: Second bounding box
            threshold: Movement threshold in pixels (default: 50)

        Returns:
            True if movement exceeds threshold

        Example:
            >>> if MovementAnalyzer.is_significant_movement(bbox1, bbox2, threshold=50):
            ...     print("Pallet has moved significantly")
        """
        distance = MovementAnalyzer.calculate_movement(bbox1, bbox2)
        return distance >= threshold

    @staticmethod
    def is_significant_size_change(
        bbox1: BoundingBox,
        bbox2: BoundingBox,
        threshold: float = 0.2
    ) -> bool:
        """Check if size change between bboxes exceeds threshold.

        Convenience method for quick size change significance checks.

        Args:
            bbox1: First bounding box
            bbox2: Second bounding box
            threshold: Relative size change threshold (default: 0.2 = 20%)

        Returns:
            True if size change exceeds threshold

        Example:
            >>> if MovementAnalyzer.is_significant_size_change(bbox1, bbox2, threshold=0.2):
            ...     print("Pallet size changed significantly (approaching/retreating)")
        """
        change = MovementAnalyzer.calculate_size_change(bbox1, bbox2)
        return change >= threshold

    @staticmethod
    def calculate_track_stability(track: PalletTrack, n_recent: int = 10) -> float:
        """Calculate track stability score.

        Stable tracks (low variance in position) might need less frequent
        sampling than erratic tracks.

        Args:
            track: PalletTrack object with detection history
            n_recent: Number of recent detections to analyze (default: 10)

        Returns:
            Stability score (0-1, where 1.0 = perfectly stable)

        Example:
            >>> stability = MovementAnalyzer.calculate_track_stability(track)
            >>> if stability > 0.8:
            ...     print("Very stable track - can sample less frequently")
        """
        if len(track.detections) < 3:
            return 0.0  # Not enough data

        recent_detections = track.detections[-n_recent:]

        if len(recent_detections) < 3:
            return 0.0

        # Calculate variance in movement
        movements = []
        for i in range(1, len(recent_detections)):
            bbox1 = recent_detections[i - 1].bbox
            bbox2 = recent_detections[i].bbox
            distance = MovementAnalyzer.calculate_movement(bbox1, bbox2)
            movements.append(distance)

        if not movements:
            return 1.0

        # Calculate standard deviation
        mean_movement = sum(movements) / len(movements)
        variance = sum((m - mean_movement) ** 2 for m in movements) / len(movements)
        std_dev = math.sqrt(variance)

        # Normalize to 0-1 scale (inverse of coefficient of variation)
        # High std_dev relative to mean = low stability
        if mean_movement == 0:
            return 1.0 if std_dev == 0 else 0.0

        coefficient_of_variation = std_dev / mean_movement
        stability = 1.0 / (1.0 + coefficient_of_variation)

        return min(max(stability, 0.0), 1.0)
