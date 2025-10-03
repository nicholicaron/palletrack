"""Best frame selection for OCR processing.

This module provides intelligent frame selection to choose the optimal frames
from a pallet track for OCR processing.
"""

from typing import Dict, List, Tuple

import numpy as np

from app.cv_models import BoundingBox

from .frame_quality import FrameQualityScorer


class BestFrameSelector:
    """Select the best frames from a pallet track for OCR processing.

    Uses composite quality scoring to rank frames and select the N best
    candidates for OCR processing.

    Attributes:
        scorer: FrameQualityScorer instance
        n_frames: Number of frames to select (default: 5)

    Example:
        >>> config = {
        ...     'sharpness_weight': 0.5,
        ...     'size_weight': 0.3,
        ...     'angle_weight': 0.2,
        ...     'min_sharpness': 100.0,
        ...     'frames_to_select': 5
        ... }
        >>> selector = BestFrameSelector(config)
        >>> track_frames = [(42, frame1, bbox1), (43, frame2, bbox2), ...]
        >>> best_ids = selector.select_best_frames(track_frames)
        >>> print(f"Selected frames: {best_ids}")
    """

    def __init__(self, config: Dict):
        """Initialize frame selector with configuration.

        Args:
            config: Configuration dict with quality weights and thresholds
                - frames_to_select (int): Number of frames to select (default: 5)
                - sharpness_weight (float): Weight for sharpness (default: 0.5)
                - size_weight (float): Weight for size (default: 0.3)
                - angle_weight (float): Weight for angle (default: 0.2)
                - min_sharpness (float): Minimum sharpness threshold (default: 100.0)
                - min_size_score (float): Minimum size score (default: 0.0)
        """
        self.scorer = FrameQualityScorer(config)
        self.n_frames = config.get("frames_to_select", 5)
        self.min_size_score = config.get("min_size_score", 0.0)

    def select_best_frames(
        self, track_frames: List[Tuple[int, np.ndarray, BoundingBox]], n_frames: int = None
    ) -> List[int]:
        """Select the N best frames from a track for OCR processing.

        Strategy:
        1. Score all frames using composite quality metric
        2. Filter out frames below minimum thresholds
        3. Sort by composite score (descending)
        4. Return top N frame IDs

        Args:
            track_frames: List of (frame_id, frame_image, bbox) tuples
            n_frames: Number of frames to select (overrides config if provided)

        Returns:
            List of frame IDs for the best frames, sorted by quality (best first)

        Example:
            >>> track_frames = [
            ...     (42, frame1, bbox1),
            ...     (43, frame2, bbox2),
            ...     (44, frame3, bbox3),
            ... ]
            >>> best_ids = selector.select_best_frames(track_frames, n_frames=2)
            >>> print(best_ids)  # [43, 42]
        """
        if n_frames is None:
            n_frames = self.n_frames

        # Score all frames
        frame_scores = []
        for frame_id, frame, bbox in track_frames:
            scores = self.scorer.score_frame(frame, bbox)

            # Filter by minimum thresholds
            if scores["acceptable"] and scores["size_score"] >= self.min_size_score:
                frame_scores.append((frame_id, scores["composite_score"], scores))

        # If no frames meet thresholds, return empty list
        if not frame_scores:
            return []

        # Sort by composite score (descending)
        frame_scores.sort(key=lambda x: x[1], reverse=True)

        # Return top N frame IDs (or all if fewer than N meet criteria)
        return [frame_id for frame_id, _, _ in frame_scores[:n_frames]]

    def select_best_frames_with_scores(
        self, track_frames: List[Tuple[int, np.ndarray, BoundingBox]], n_frames: int = None
    ) -> List[Tuple[int, Dict[str, float]]]:
        """Select best frames and return with detailed scores.

        Similar to select_best_frames but returns detailed quality metrics
        for each selected frame.

        Args:
            track_frames: List of (frame_id, frame_image, bbox) tuples
            n_frames: Number of frames to select (overrides config if provided)

        Returns:
            List of (frame_id, scores_dict) tuples, sorted by quality (best first)
            Each scores_dict contains:
                - sharpness_raw: Raw sharpness value
                - sharpness_score: Normalized sharpness score
                - size_score: Size score
                - angle_score: Angle score
                - composite_score: Weighted composite score
                - acceptable: Boolean indicating if frame meets minimum quality

        Example:
            >>> results = selector.select_best_frames_with_scores(track_frames)
            >>> for frame_id, scores in results:
            ...     print(f"Frame {frame_id}: {scores['composite_score']:.2f}")
        """
        if n_frames is None:
            n_frames = self.n_frames

        # Score all frames
        frame_scores = []
        for frame_id, frame, bbox in track_frames:
            scores = self.scorer.score_frame(frame, bbox)

            # Filter by minimum thresholds
            if scores["acceptable"] and scores["size_score"] >= self.min_size_score:
                frame_scores.append((frame_id, scores["composite_score"], scores))

        # Sort by composite score (descending)
        frame_scores.sort(key=lambda x: x[1], reverse=True)

        # Return top N with scores
        return [(frame_id, scores) for frame_id, _, scores in frame_scores[:n_frames]]

    def get_frame_diversity_selection(
        self,
        track_frames: List[Tuple[int, np.ndarray, BoundingBox]],
        n_frames: int = None,
        min_frame_gap: int = 5,
    ) -> List[int]:
        """Select best frames with temporal diversity constraint.

        This method ensures selected frames are spread across the track timeline,
        avoiding clustering of similar consecutive frames.

        Strategy:
        1. Score all frames
        2. Sort by composite score
        3. Greedily select top frames with minimum frame gap constraint

        Args:
            track_frames: List of (frame_id, frame_image, bbox) tuples
            n_frames: Number of frames to select (overrides config if provided)
            min_frame_gap: Minimum frame gap between selected frames (default: 5)

        Returns:
            List of frame IDs, sorted by frame_id (not quality)

        Example:
            >>> # Select 3 frames with at least 10 frames between them
            >>> best_ids = selector.get_frame_diversity_selection(
            ...     track_frames, n_frames=3, min_frame_gap=10
            ... )
        """
        if n_frames is None:
            n_frames = self.n_frames

        # Score all frames
        frame_scores = []
        for frame_id, frame, bbox in track_frames:
            scores = self.scorer.score_frame(frame, bbox)

            # Filter by minimum thresholds
            if scores["acceptable"] and scores["size_score"] >= self.min_size_score:
                frame_scores.append((frame_id, scores["composite_score"]))

        # If no frames meet thresholds, return empty list
        if not frame_scores:
            return []

        # Sort by composite score (descending)
        frame_scores.sort(key=lambda x: x[1], reverse=True)

        # Greedily select frames with diversity constraint
        selected = []
        for frame_id, score in frame_scores:
            # Check if frame_id is far enough from all selected frames
            if all(abs(frame_id - sel_id) >= min_frame_gap for sel_id in selected):
                selected.append(frame_id)

                if len(selected) >= n_frames:
                    break

        # Return sorted by frame_id
        return sorted(selected)
