"""Higher-level frame selection strategies for OCR processing.

This module provides strategies for selecting optimal frames from completed
tracks for OCR processing. It combines quality scoring, movement analysis,
and temporal diversity to select the best subset of frames.
"""

from typing import Dict, List

import numpy as np

from app.cv_models import PalletTrack
from app.cv_processing.frame_quality import FrameQualityScorer
from app.cv_processing.frame_sampler import AdaptiveFrameSampler
from app.cv_processing.movement_analysis import MovementAnalyzer


class FrameSelectionStrategy:
    """Higher-level logic for frame selection.

    Combines multiple strategies to select optimal frames for OCR processing
    from completed pallet tracks. Ensures quality, diversity, and efficient
    use of computational resources.

    Attributes:
        sampler: AdaptiveFrameSampler instance
        quality_scorer: FrameQualityScorer instance
        movement_analyzer: MovementAnalyzer instance
        config: Configuration dictionary

    Example:
        >>> strategy = FrameSelectionStrategy(config)
        >>> selected_frames = strategy.select_frames_for_track(track, frames)
    """

    def __init__(self, config: Dict):
        """Initialize frame selection strategy.

        Args:
            config: Configuration dict with frame_sampling and frame_quality settings
        """
        self.config = config
        self.sampler = AdaptiveFrameSampler(config)
        self.quality_scorer = FrameQualityScorer(config)
        self.movement_analyzer = MovementAnalyzer()

    def select_frames_for_track(
        self,
        track: PalletTrack,
        frames: Dict[int, np.ndarray]
    ) -> List[int]:
        """Select optimal subset of frames for OCR processing.

        Given a completed track and its associated frames, selects the best
        frames based on quality, movement diversity, and temporal spacing.

        Process:
        1. Filter to frames where track was detected
        2. Filter to frames where documents were detected (if available)
        3. Score each frame (quality + movement diversity)
        4. Select up to max_samples frames
        5. Ensure temporal diversity (don't cluster)

        Args:
            track: PalletTrack object (should be completed)
            frames: Dictionary mapping frame_number -> frame image

        Returns:
            List of frame_numbers to process for OCR (sorted)

        Example:
            >>> selected = strategy.select_frames_for_track(completed_track, all_frames)
            >>> for frame_num in selected:
            ...     frame = all_frames[frame_num]
            ...     # Perform OCR on selected frames
        """
        if not track.detections:
            return []

        # Step 1: Get all frames where this track was detected
        candidate_frames = []
        for detection in track.detections:
            frame_num = detection.frame_number
            if frame_num in frames:
                candidate_frames.append(frame_num)

        if not candidate_frames:
            return []

        # Step 2: Filter to frames with document detections (if available)
        frames_with_documents = set()
        if track.document_regions:
            for doc in track.document_regions:
                frames_with_documents.add(doc.frame_number)

        # Prefer frames with documents, but fall back to all frames if none
        if frames_with_documents:
            candidate_frames = [f for f in candidate_frames if f in frames_with_documents]

        if not candidate_frames:
            return []

        # Step 3: Score each frame
        frame_scores = self._score_frames(track, frames, candidate_frames)

        # Step 4: Select top frames
        max_samples = self.config.get('frame_sampling', {}).get('max_samples_per_track', 10)
        selected = self._select_top_frames(frame_scores, max_samples)

        # Step 5: Ensure temporal diversity
        min_gap = self.config.get('frame_sampling', {}).get('min_temporal_gap', 10)
        diverse_frames = self.ensure_temporal_diversity(selected, min_gap)

        return sorted(diverse_frames)

    def _score_frames(
        self,
        track: PalletTrack,
        frames: Dict[int, np.ndarray],
        candidate_frames: List[int]
    ) -> Dict[int, float]:
        """Score each candidate frame.

        Combines quality score with movement diversity score.

        Args:
            track: PalletTrack object
            frames: Dictionary of frame images
            candidate_frames: List of candidate frame numbers

        Returns:
            Dictionary mapping frame_number to composite score
        """
        frame_scores = {}

        # Create mapping of frame_number to detection
        detection_map = {det.frame_number: det for det in track.detections}

        for frame_num in candidate_frames:
            if frame_num not in frames or frame_num not in detection_map:
                continue

            frame = frames[frame_num]
            detection = detection_map[frame_num]
            bbox = detection.bbox

            # Quality score
            quality_result = self.quality_scorer.score_frame(frame, bbox)
            quality_score = quality_result['composite_score']

            # Movement diversity score (how different is this frame from others?)
            diversity_score = self._calculate_diversity_score(
                detection, track, candidate_frames
            )

            # Composite score (weighted combination)
            quality_weight = 0.7
            diversity_weight = 0.3
            composite_score = (
                quality_weight * quality_score +
                diversity_weight * diversity_score
            )

            frame_scores[frame_num] = composite_score

        return frame_scores

    def _calculate_diversity_score(
        self,
        detection,
        track: PalletTrack,
        candidate_frames: List[int]
    ) -> float:
        """Calculate how diverse this frame is from other candidates.

        Frames that are very different in position/size from others get
        higher diversity scores.

        Args:
            detection: PalletDetection for this frame
            track: PalletTrack object
            candidate_frames: List of candidate frame numbers

        Returns:
            Diversity score (0-1)
        """
        if len(candidate_frames) <= 1:
            return 1.0

        bbox = detection.bbox
        frame_num = detection.frame_number

        # Calculate average distance from other candidate frames
        total_distance = 0.0
        count = 0

        detection_map = {det.frame_number: det for det in track.detections}

        for other_frame_num in candidate_frames:
            if other_frame_num == frame_num:
                continue

            if other_frame_num not in detection_map:
                continue

            other_bbox = detection_map[other_frame_num].bbox
            distance = MovementAnalyzer.calculate_movement(bbox, other_bbox)
            total_distance += distance
            count += 1

        if count == 0:
            return 1.0

        avg_distance = total_distance / count

        # Normalize to 0-1 scale (higher distance = higher diversity)
        # Assume max useful distance is ~200 pixels
        max_distance = 200.0
        diversity_score = min(avg_distance / max_distance, 1.0)

        return diversity_score

    def _select_top_frames(
        self,
        frame_scores: Dict[int, float],
        max_samples: int
    ) -> List[int]:
        """Select top N frames by score.

        Args:
            frame_scores: Dictionary mapping frame_number to score
            max_samples: Maximum number of frames to select

        Returns:
            List of selected frame numbers
        """
        # Sort by score (descending)
        sorted_frames = sorted(
            frame_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        # Take top N
        selected = [frame_num for frame_num, _ in sorted_frames[:max_samples]]

        return selected

    def ensure_temporal_diversity(
        self,
        selected_frames: List[int],
        min_gap: int = 10
    ) -> List[int]:
        """Remove frames that are too close together temporally.

        Prevents processing nearly identical consecutive frames.

        Args:
            selected_frames: List of selected frame numbers
            min_gap: Minimum frame gap required (default: 10)

        Returns:
            Filtered list with temporal diversity ensured

        Example:
            >>> frames = [10, 15, 20, 50, 52, 100]
            >>> diverse = strategy.ensure_temporal_diversity(frames, min_gap=10)
            >>> # Result: [10, 20, 50, 100] (15 removed as too close to 10/20, 52 too close to 50)
        """
        if not selected_frames:
            return []

        # Sort frames
        sorted_frames = sorted(selected_frames)

        # Greedy selection: keep frames that are far enough apart
        diverse_frames = [sorted_frames[0]]

        for frame_num in sorted_frames[1:]:
            last_selected = diverse_frames[-1]
            gap = frame_num - last_selected

            if gap >= min_gap:
                diverse_frames.append(frame_num)

        return diverse_frames

    def select_best_single_frame(
        self,
        track: PalletTrack,
        frames: Dict[int, np.ndarray]
    ) -> int:
        """Select single best frame for OCR.

        Useful when you only want to process one frame per track.

        Args:
            track: PalletTrack object
            frames: Dictionary of frame images

        Returns:
            Frame number of best frame, or -1 if none found

        Example:
            >>> best_frame_num = strategy.select_best_single_frame(track, frames)
            >>> if best_frame_num >= 0:
            ...     frame = frames[best_frame_num]
        """
        selected = self.select_frames_for_track(track, frames)

        if not selected:
            return -1

        # Return first (highest scoring) frame
        return selected[0]

    def select_frames_adaptive(
        self,
        track: PalletTrack,
        frames: Dict[int, np.ndarray],
        base_samples: int = 5,
        velocity_multiplier: float = 1.5
    ) -> List[int]:
        """Adaptively select frames based on track characteristics.

        Fast-moving tracks get more samples than slow-moving tracks.

        Args:
            track: PalletTrack object
            frames: Dictionary of frame images
            base_samples: Base number of samples for slow tracks (default: 5)
            velocity_multiplier: Multiplier for fast tracks (default: 1.5)

        Returns:
            List of selected frame numbers

        Example:
            >>> # Fast track might get 7-8 samples, slow track gets 5
            >>> selected = strategy.select_frames_adaptive(track, frames)
        """
        # Estimate track velocity
        velocity = MovementAnalyzer.estimate_velocity(track, n_recent=10)

        # Determine number of samples based on velocity
        high_velocity_threshold = self.config.get(
            'frame_sampling', {}
        ).get('high_velocity_threshold', 20.0)

        if velocity >= high_velocity_threshold:
            # Fast track - use more samples
            max_samples = int(base_samples * velocity_multiplier)
        else:
            # Slow track - use base samples
            max_samples = base_samples

        # Override config temporarily
        original_max = self.config.get('frame_sampling', {}).get('max_samples_per_track')
        self.config.setdefault('frame_sampling', {})['max_samples_per_track'] = max_samples

        # Select frames
        selected = self.select_frames_for_track(track, frames)

        # Restore original config
        if original_max is not None:
            self.config['frame_sampling']['max_samples_per_track'] = original_max

        return selected
