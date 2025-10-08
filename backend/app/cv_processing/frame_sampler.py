"""Adaptive frame sampling for intelligent OCR processing.

This module implements intelligent frame selection to avoid processing every
frame. Instead, it samples frames based on movement, quality, and timing to
capture the most useful frames for OCR while minimizing computational cost.
"""

from typing import Dict, Tuple

from app.cv_models import BoundingBox
from app.cv_processing.movement_analysis import MovementAnalyzer


class AdaptiveFrameSampler:
    """Decide which frames to process for OCR based on track movement.

    Maintains state for each track to determine when significant changes
    warrant processing a new frame. Uses multiple criteria including movement,
    size changes, quality scores, and timing constraints.

    Attributes:
        config: Configuration dictionary
        last_processed: Maps track_id to (bbox, frame_number) of last processed frame
        samples_per_track: Maps track_id to count of frames processed

    Example:
        >>> config = {'frame_sampling': {...}}
        >>> sampler = AdaptiveFrameSampler(config)
        >>> if sampler.should_process_frame(track_id, bbox, frame_num, quality):
        ...     # Process this frame for OCR
        ...     sampler.mark_processed(track_id, bbox, frame_num)
    """

    def __init__(self, config: Dict):
        """Initialize adaptive frame sampler.

        Args:
            config: Configuration dict with frame_sampling settings
                - frame_sampling.movement_threshold: Pixels of movement to trigger sampling
                - frame_sampling.size_change_threshold: Relative size change to trigger sampling
                - frame_sampling.max_frame_gap: Maximum frames between samples
                - frame_sampling.max_samples_per_track: Maximum samples per track
                - frame_sampling.min_quality_score: Minimum quality score to consider
                - frame_sampling.high_velocity_threshold: Velocity for increased sampling
                - frame_sampling.high_velocity_sample_rate: Sample rate for fast tracks
        """
        self.config = config
        self.sampling_config = config.get('frame_sampling', {})

        # Track state: track_id -> (last_bbox, last_frame_number)
        self.last_processed: Dict[int, Tuple[BoundingBox, int]] = {}

        # Sample counts: track_id -> count
        self.samples_per_track: Dict[int, int] = {}

    def should_process_frame(
        self,
        track_id: int,
        current_bbox: BoundingBox,
        frame_number: int,
        frame_quality_score: float = 1.0,
        track_velocity: float = 0.0
    ) -> bool:
        """Determine if current frame should be processed for OCR.

        Decision factors (in order of priority):
        1. First detection of track → always process
        2. Quality score below minimum → never process
        3. Max samples exceeded → never process
        4. Significant movement since last processed frame → process
        5. Significant size change (approaching/retreating) → process
        6. Maximum frame gap exceeded (backup condition) → process
        7. High velocity and sample rate interval → process
        8. High quality score and some movement → process

        Args:
            track_id: ID of the track being evaluated
            current_bbox: Current bounding box
            frame_number: Current frame number
            frame_quality_score: Quality score for this frame (0-1, default: 1.0)
            track_velocity: Current track velocity in pixels/frame (default: 0.0)

        Returns:
            True if frame should be processed for OCR

        Example:
            >>> should_process = sampler.should_process_frame(
            ...     track_id=1,
            ...     current_bbox=bbox,
            ...     frame_number=42,
            ...     frame_quality_score=0.85,
            ...     track_velocity=15.0
            ... )
        """
        # Get configuration parameters
        movement_threshold = self.sampling_config.get('movement_threshold', 50.0)
        size_change_threshold = self.sampling_config.get('size_change_threshold', 0.2)
        max_frame_gap = self.sampling_config.get('max_frame_gap', 30)
        max_samples = self.sampling_config.get('max_samples_per_track', 10)
        min_quality = self.sampling_config.get('min_quality_score', 0.5)
        high_velocity_threshold = self.sampling_config.get('high_velocity_threshold', 20.0)
        high_velocity_sample_rate = self.sampling_config.get('high_velocity_sample_rate', 5)

        # Factor 1: First detection → always process
        if track_id not in self.last_processed:
            return True

        # Factor 2: Quality too low → never process
        if frame_quality_score < min_quality:
            return False

        # Factor 3: Max samples exceeded → never process
        sample_count = self.samples_per_track.get(track_id, 0)
        if sample_count >= max_samples:
            return False

        # Get last processed state
        last_bbox, last_frame_number = self.last_processed[track_id]

        # Calculate frame gap
        frame_gap = frame_number - last_frame_number

        # Factor 4: Significant movement → process
        movement = MovementAnalyzer.calculate_movement(last_bbox, current_bbox)
        if movement >= movement_threshold:
            return True

        # Factor 5: Significant size change → process
        size_change = MovementAnalyzer.calculate_size_change(last_bbox, current_bbox)
        if size_change >= size_change_threshold:
            return True

        # Factor 6: Max frame gap exceeded → process (backup condition)
        if frame_gap >= max_frame_gap:
            return True

        # Factor 7: High velocity → sample more frequently
        if track_velocity >= high_velocity_threshold:
            if frame_gap >= high_velocity_sample_rate:
                return True

        # Factor 8: High quality + some movement → process
        # This catches frames that are very high quality even if movement is moderate
        if frame_quality_score >= 0.8:
            # Lower threshold for movement if quality is high
            if movement >= movement_threshold * 0.5:
                return True

        # Default: don't process
        return False

    def mark_processed(self, track_id: int, bbox: BoundingBox, frame_number: int):
        """Record that this frame was processed.

        Updates internal state to track the last processed frame for this track.

        Args:
            track_id: ID of the track
            bbox: Bounding box of the processed frame
            frame_number: Frame number that was processed

        Example:
            >>> sampler.mark_processed(track_id=1, bbox=bbox, frame_number=42)
        """
        self.last_processed[track_id] = (bbox, frame_number)

        # Increment sample count
        if track_id not in self.samples_per_track:
            self.samples_per_track[track_id] = 0
        self.samples_per_track[track_id] += 1

    def get_sample_count(self, track_id: int) -> int:
        """Get number of frames processed for this track.

        Args:
            track_id: ID of the track

        Returns:
            Number of frames processed for this track

        Example:
            >>> count = sampler.get_sample_count(track_id=1)
            >>> print(f"Processed {count} frames for track 1")
        """
        return self.samples_per_track.get(track_id, 0)

    def reset_track(self, track_id: int):
        """Reset sampling state for a track.

        Call this when a track is finalized/completed to free memory.

        Args:
            track_id: ID of the track to reset

        Example:
            >>> sampler.reset_track(track_id=1)
        """
        if track_id in self.last_processed:
            del self.last_processed[track_id]
        if track_id in self.samples_per_track:
            del self.samples_per_track[track_id]

    def reset_all(self):
        """Reset all sampling state.

        Useful for restarting processing or freeing memory.

        Example:
            >>> sampler.reset_all()
        """
        self.last_processed.clear()
        self.samples_per_track.clear()

    def get_last_processed_frame(self, track_id: int) -> int:
        """Get frame number of last processed frame for track.

        Args:
            track_id: ID of the track

        Returns:
            Frame number of last processed frame, or -1 if none

        Example:
            >>> last_frame = sampler.get_last_processed_frame(track_id=1)
        """
        if track_id not in self.last_processed:
            return -1
        _, frame_number = self.last_processed[track_id]
        return frame_number

    def get_statistics(self) -> Dict:
        """Get sampling statistics across all tracks.

        Returns:
            Dictionary with statistics:
                - total_tracks: Number of tracks seen
                - total_samples: Total frames processed
                - avg_samples_per_track: Average samples per track
                - tracks_at_max: Number of tracks at max sample limit

        Example:
            >>> stats = sampler.get_statistics()
            >>> print(f"Processed {stats['total_samples']} frames across {stats['total_tracks']} tracks")
        """
        total_tracks = len(self.samples_per_track)
        total_samples = sum(self.samples_per_track.values())
        max_samples = self.sampling_config.get('max_samples_per_track', 10)

        tracks_at_max = sum(
            1 for count in self.samples_per_track.values()
            if count >= max_samples
        )

        avg_samples = total_samples / total_tracks if total_tracks > 0 else 0.0

        return {
            'total_tracks': total_tracks,
            'total_samples': total_samples,
            'avg_samples_per_track': avg_samples,
            'tracks_at_max': tracks_at_max
        }
