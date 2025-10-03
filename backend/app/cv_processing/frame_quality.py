"""Frame quality assessment for OCR optimization.

This module provides functions to assess the quality of video frames
for OCR processing by measuring sharpness, size, and viewing angle.
"""

from typing import Dict, Tuple

import cv2
import numpy as np

from app.cv_models import BoundingBox


def calculate_sharpness(image_region: np.ndarray) -> float:
    """Calculate sharpness score using Laplacian variance.

    The Laplacian operator detects edges in the image. Blurry images have
    fewer strong edges, resulting in lower variance values.

    Args:
        image_region: Grayscale or color image region (numpy array)

    Returns:
        Sharpness score (higher = sharper)
        Typical values:
            < 100: Very blurry
            100-300: Moderate blur
            > 300: Sharp

    Example:
        >>> region = cv2.imread("document.jpg", cv2.IMREAD_GRAYSCALE)
        >>> sharpness = calculate_sharpness(region)
        >>> if sharpness > 100:
        ...     print("Frame is acceptable for OCR")
    """
    # Convert to grayscale if needed
    if len(image_region.shape) == 3:
        gray = cv2.cvtColor(image_region, cv2.COLOR_BGR2GRAY)
    else:
        gray = image_region

    # Calculate Laplacian and return variance
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    variance = laplacian.var()

    return float(variance)


def calculate_size_score(bbox: BoundingBox, frame_shape: Tuple[int, int]) -> float:
    """Calculate size score based on bbox area relative to frame.

    Larger regions (closer to camera) generally provide better OCR results.
    Score is normalized between 0 and 1.

    Args:
        bbox: Bounding box of the detected region
        frame_shape: Shape of the frame (height, width)

    Returns:
        Size score between 0 and 1 (larger = better)
        0.0: Very small region (< 1% of frame)
        0.5: Moderate size region (~ 10% of frame)
        1.0: Large region (> 30% of frame)

    Example:
        >>> bbox = BoundingBox(x1=100, y1=100, x2=300, y2=200, confidence=0.9)
        >>> score = calculate_size_score(bbox, (1080, 1920))
        >>> print(f"Size score: {score:.2f}")
    """
    frame_height, frame_width = frame_shape
    frame_area = frame_height * frame_width

    bbox_area = bbox.area()
    area_ratio = bbox_area / frame_area

    # Apply sigmoid-like transformation to map ratio to 0-1 score
    # Areas > 30% of frame get score close to 1.0
    # Areas < 1% of frame get score close to 0.0
    if area_ratio >= 0.3:
        return 1.0
    elif area_ratio <= 0.01:
        return 0.0
    else:
        # Linear interpolation between 0.01 and 0.3
        return (area_ratio - 0.01) / (0.3 - 0.01)


def calculate_angle_score(bbox: BoundingBox) -> float:
    """Estimate if bbox represents a frontal view using aspect ratio heuristic.

    Documents viewed at an angle appear compressed. This function uses a simple
    heuristic based on aspect ratio to estimate viewing angle quality.

    Standard document aspect ratios:
        - US Letter: ~1.29 (8.5 x 11 inches)
        - A4: ~1.41
        - Shipping labels: typically 1.0-1.5

    Args:
        bbox: Bounding box of the detected document

    Returns:
        Angle score between 0 and 1 (higher = more frontal)
        1.0: Aspect ratio matches typical document (1.2-1.5)
        0.5: Somewhat compressed/stretched
        0.0: Extremely distorted (< 0.5 or > 3.0)

    Example:
        >>> bbox = BoundingBox(x1=100, y1=100, x2=300, y2=400, confidence=0.9)
        >>> score = calculate_angle_score(bbox)
        >>> if score > 0.7:
        ...     print("Document appears to be frontal-facing")
    """
    width = bbox.width()
    height = bbox.height()

    # Avoid division by zero
    if height == 0:
        return 0.0

    aspect_ratio = width / height

    # Ideal range for documents: 1.2 to 1.5 (portrait orientation)
    # Also check inverse for landscape orientation
    ideal_min = 1.2
    ideal_max = 1.5

    # Check both portrait and landscape
    if ideal_min <= aspect_ratio <= ideal_max:
        return 1.0
    elif (1 / ideal_max) <= aspect_ratio <= (1 / ideal_min):
        return 1.0
    # Square-ish documents (shipping labels)
    elif 0.8 <= aspect_ratio <= 1.2:
        return 0.9
    # Moderately distorted
    elif 0.5 <= aspect_ratio <= 3.0:
        # Linear falloff from ideal range
        if aspect_ratio < ideal_min:
            return 0.5 + 0.5 * (aspect_ratio - 0.5) / (ideal_min - 0.5)
        else:
            return 0.5 + 0.5 * (3.0 - aspect_ratio) / (3.0 - ideal_max)
    else:
        # Extremely distorted
        return 0.0


class FrameQualityScorer:
    """Composite frame quality scorer for OCR optimization.

    Combines multiple quality metrics (sharpness, size, angle) into a
    single composite score for ranking frames.

    Attributes:
        sharpness_weight: Weight for sharpness metric (default: 0.5)
        size_weight: Weight for size metric (default: 0.3)
        angle_weight: Weight for angle metric (default: 0.2)
        min_sharpness: Minimum acceptable sharpness value (default: 100.0)

    Example:
        >>> config = {
        ...     'sharpness_weight': 0.5,
        ...     'size_weight': 0.3,
        ...     'angle_weight': 0.2,
        ...     'min_sharpness': 100.0
        ... }
        >>> scorer = FrameQualityScorer(config)
        >>> result = scorer.score_frame(frame, bbox)
        >>> if result['composite_score'] > 0.7:
        ...     print("High quality frame for OCR")
    """

    def __init__(self, config: Dict):
        """Initialize scorer with configuration.

        Args:
            config: Configuration dict with weights and thresholds
                - sharpness_weight (float): Weight for sharpness (default: 0.5)
                - size_weight (float): Weight for size (default: 0.3)
                - angle_weight (float): Weight for angle (default: 0.2)
                - min_sharpness (float): Minimum sharpness threshold (default: 100.0)
        """
        self.sharpness_weight = config.get("sharpness_weight", 0.5)
        self.size_weight = config.get("size_weight", 0.3)
        self.angle_weight = config.get("angle_weight", 0.2)
        self.min_sharpness = config.get("min_sharpness", 100.0)

        # Normalize weights
        total_weight = self.sharpness_weight + self.size_weight + self.angle_weight
        if total_weight > 0:
            self.sharpness_weight /= total_weight
            self.size_weight /= total_weight
            self.angle_weight /= total_weight

    def score_frame(self, frame: np.ndarray, bbox: BoundingBox) -> Dict[str, float]:
        """Score a single frame region for OCR quality.

        Args:
            frame: Full frame image (BGR or grayscale)
            bbox: Bounding box of region to score

        Returns:
            Dictionary containing:
                - sharpness_raw: Raw sharpness value (Laplacian variance)
                - sharpness_score: Normalized sharpness score (0-1)
                - size_score: Size score (0-1)
                - angle_score: Angle score (0-1)
                - composite_score: Weighted composite score (0-1)
                - acceptable: Boolean indicating if frame meets minimum quality

        Example:
            >>> scorer = FrameQualityScorer({'min_sharpness': 100.0})
            >>> scores = scorer.score_frame(frame, bbox)
            >>> print(f"Composite: {scores['composite_score']:.2f}")
            >>> print(f"Acceptable: {scores['acceptable']}")
        """
        frame_height, frame_width = frame.shape[:2]

        # Extract region of interest
        x1, y1, x2, y2 = int(bbox.x1), int(bbox.y1), int(bbox.x2), int(bbox.y2)

        # Ensure coordinates are within frame bounds
        x1 = max(0, min(x1, frame_width - 1))
        y1 = max(0, min(y1, frame_height - 1))
        x2 = max(x1 + 1, min(x2, frame_width))
        y2 = max(y1 + 1, min(y2, frame_height))

        region = frame[y1:y2, x1:x2]

        # Calculate individual metrics
        sharpness_raw = calculate_sharpness(region)
        size_score = calculate_size_score(bbox, (frame_height, frame_width))
        angle_score = calculate_angle_score(bbox)

        # Normalize sharpness score (using sigmoid-like transformation)
        # Values around min_sharpness get 0.5, higher values approach 1.0
        sharpness_score = self._normalize_sharpness(sharpness_raw)

        # Calculate weighted composite score
        composite_score = (
            self.sharpness_weight * sharpness_score
            + self.size_weight * size_score
            + self.angle_weight * angle_score
        )

        # Frame is acceptable if it meets minimum sharpness threshold
        acceptable = sharpness_raw >= self.min_sharpness

        return {
            "sharpness_raw": sharpness_raw,
            "sharpness_score": sharpness_score,
            "size_score": size_score,
            "angle_score": angle_score,
            "composite_score": composite_score,
            "acceptable": acceptable,
        }

    def _normalize_sharpness(self, sharpness_raw: float) -> float:
        """Normalize raw sharpness value to 0-1 score.

        Uses sigmoid-like transformation centered at min_sharpness threshold.

        Args:
            sharpness_raw: Raw Laplacian variance value

        Returns:
            Normalized score between 0 and 1
        """
        if sharpness_raw >= self.min_sharpness * 3:
            return 1.0
        elif sharpness_raw <= self.min_sharpness * 0.5:
            return 0.0
        else:
            # Linear interpolation between min_sharpness*0.5 and min_sharpness*3
            return (sharpness_raw - self.min_sharpness * 0.5) / (
                self.min_sharpness * 3 - self.min_sharpness * 0.5
            )
