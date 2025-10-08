"""ByteTrack-based pallet tracking module.

This module provides wrappers around ByteTrack for tracking pallets across
video frames, maintaining their identity through occlusions and managing
track lifecycle.
"""

from typing import Dict, List, Optional

import numpy as np
import supervision as sv

from app.cv_models import PalletDetection, PalletTrack, TrackStatus
from app.cv_processing.frame_quality import FrameQualityScorer


class PalletTracker:
    """ByteTrack wrapper for pallet tracking.

    Manages pallet tracking using supervision's ByteTrack implementation,
    maintaining active and completed tracks with full detection history.

    Attributes:
        tracker: ByteTrack tracker instance
        active_tracks: Dictionary of currently active PalletTrack objects
        completed_tracks: Dictionary of completed PalletTrack objects

    Example:
        >>> config = {
        ...     'tracking': {
        ...         'max_age': 30,
        ...         'min_hits': 3,
        ...         'iou_threshold': 0.3
        ...     }
        ... }
        >>> tracker = PalletTracker(config)
        >>> active = tracker.update(detections, frame_number=0)
    """

    def __init__(self, config: Dict):
        """Initialize ByteTrack tracker.

        Args:
            config: Configuration dict with tracking settings
                - tracking.max_age: Frames to keep track alive without detection
                - tracking.min_hits: Minimum detections before confirming track
                - tracking.iou_threshold: IoU threshold for matching
        """
        tracking_config = config['tracking']

        # Initialize ByteTrack with supervision
        self.tracker = sv.ByteTrack(
            track_activation_threshold=tracking_config['min_hits'],
            lost_track_buffer=tracking_config['max_age'],
            minimum_matching_threshold=tracking_config['iou_threshold']
        )

        self.active_tracks: Dict[int, PalletTrack] = {}
        self.completed_tracks: Dict[int, PalletTrack] = {}
        self.config = config

    def update(self,
               detections: List[PalletDetection],
               frame_number: int) -> Dict[int, PalletTrack]:
        """Update tracker with new detections.

        Process:
        1. Convert PalletDetection to supervision Detections format
        2. Update ByteTrack tracker
        3. Update PalletTrack objects with track IDs
        4. Manage track lifecycle (active → completed)

        Args:
            detections: List of PalletDetection objects from current frame
            frame_number: Current frame number

        Returns:
            Dictionary of active PalletTrack objects keyed by track_id
        """
        if not detections:
            # Still update tracker with empty detections to age out tracks
            empty_detections = sv.Detections.empty()
            self.tracker.update_with_detections(empty_detections)
            return self.active_tracks

        # Convert PalletDetection to supervision Detections format
        sv_detections = self._convert_to_sv_detections(detections)

        # Update ByteTrack
        tracked_detections = self.tracker.update_with_detections(sv_detections)

        # Get current track IDs from this frame
        current_track_ids = set()

        # Update PalletTrack objects with new detections
        if len(tracked_detections) > 0:
            for i, track_id in enumerate(tracked_detections.tracker_id):
                track_id = int(track_id)
                current_track_ids.add(track_id)

                # Get corresponding detection
                detection = detections[i]
                detection.track_id = track_id

                # Create or update PalletTrack
                if track_id not in self.active_tracks:
                    # New track
                    self.active_tracks[track_id] = PalletTrack(
                        track_id=track_id,
                        detections=[detection],
                        first_seen_frame=frame_number,
                        last_seen_frame=frame_number,
                        status=TrackStatus.ACTIVE
                    )
                else:
                    # Update existing track
                    track = self.active_tracks[track_id]
                    track.detections.append(detection)
                    track.last_seen_frame = frame_number

        # Check for lost tracks (tracks that were active but not in current frame)
        lost_track_ids = set(self.active_tracks.keys()) - current_track_ids
        for track_id in lost_track_ids:
            # Check if track has been lost for too long
            frames_since_last_seen = frame_number - self.active_tracks[track_id].last_seen_frame
            if frames_since_last_seen > self.config['tracking']['max_age']:
                self.finalize_track(track_id)

        return self.active_tracks

    def _convert_to_sv_detections(self, detections: List[PalletDetection]) -> sv.Detections:
        """Convert PalletDetection list to supervision Detections format.

        Args:
            detections: List of PalletDetection objects

        Returns:
            supervision Detections object
        """
        if not detections:
            return sv.Detections.empty()

        # Extract bounding boxes in [x1, y1, x2, y2] format
        xyxy = np.array([
            [det.bbox.x1, det.bbox.y1, det.bbox.x2, det.bbox.y2]
            for det in detections
        ], dtype=np.float32)

        # Extract confidence scores
        confidence = np.array([det.bbox.confidence for det in detections], dtype=np.float32)

        # Create supervision Detections object
        # class_id is not needed for tracking, but required by supervision
        class_id = np.zeros(len(detections), dtype=int)

        return sv.Detections(
            xyxy=xyxy,
            confidence=confidence,
            class_id=class_id
        )

    def get_track(self, track_id: int) -> Optional[PalletTrack]:
        """Retrieve track by ID (active or completed).

        Args:
            track_id: Track ID to retrieve

        Returns:
            PalletTrack object if found, None otherwise
        """
        if track_id in self.active_tracks:
            return self.active_tracks[track_id]
        if track_id in self.completed_tracks:
            return self.completed_tracks[track_id]
        return None

    def finalize_track(self, track_id: int):
        """Move track from active to completed.

        Called when track is lost or exits frame.

        Args:
            track_id: Track ID to finalize
        """
        if track_id in self.active_tracks:
            track = self.active_tracks.pop(track_id)
            track.status = TrackStatus.COMPLETED
            self.completed_tracks[track_id] = track

    def finalize_all_tracks(self):
        """Finalize all active tracks.

        Useful for end of video processing.
        """
        track_ids = list(self.active_tracks.keys())
        for track_id in track_ids:
            self.finalize_track(track_id)


class TrackManager:
    """Higher-level track management logic.

    Manages tracking decisions like when to process tracks for OCR,
    which tracks are ready for final extraction, and quality assessment.

    Attributes:
        tracker: PalletTracker instance
        config: Configuration dictionary
        quality_assessor: FrameQualityScorer instance
        last_ocr_frame: Dictionary tracking last OCR attempt per track

    Example:
        >>> config = {...}  # Full configuration
        >>> manager = TrackManager(config)
        >>> active_tracks = manager.update(detections, frame_number=0)
        >>> if manager.should_process_track(track_id, frame_number):
        ...     # Perform OCR on this track
    """

    def __init__(self, config: Dict):
        """Initialize track manager.

        Args:
            config: Configuration dict with tracking and frame_quality settings
        """
        self.tracker = PalletTracker(config)
        self.config = config
        self.quality_assessor = FrameQualityScorer(config)
        self.last_ocr_frame: Dict[int, int] = {}

    def update(self,
               detections: List[PalletDetection],
               frame_number: int) -> Dict[int, PalletTrack]:
        """Update tracker with new detections.

        Args:
            detections: List of PalletDetection objects
            frame_number: Current frame number

        Returns:
            Dictionary of active PalletTrack objects
        """
        return self.tracker.update(detections, frame_number)

    def should_process_track(self, track_id: int, frame_number: int) -> bool:
        """Determine if track should be processed for OCR in this frame.

        Logic:
        - Has track been alive for minimum frames?
        - Has enough time passed since last OCR attempt?

        Args:
            track_id: Track ID to check
            frame_number: Current frame number

        Returns:
            True if track should be processed for OCR
        """
        track = self.tracker.get_track(track_id)
        if track is None:
            return False

        # Check if track has minimum length
        track_length = len(track.detections)
        min_track_length = self.config.get('tracking', {}).get('min_track_length', 10)
        if track_length < min_track_length:
            return False

        # Check if enough frames have passed since last OCR
        frames_between_ocr = self.config.get('tracking', {}).get('frames_between_ocr', 15)
        if track_id in self.last_ocr_frame:
            frames_since_last_ocr = frame_number - self.last_ocr_frame[track_id]
            if frames_since_last_ocr < frames_between_ocr:
                return False

        # Update last OCR frame
        self.last_ocr_frame[track_id] = frame_number
        return True

    def assess_frame_quality(self, frame: np.ndarray, bbox) -> float:
        """Assess quality of a frame region for OCR.

        Args:
            frame: Frame image
            bbox: BoundingBox object

        Returns:
            Quality score (0-1)
        """
        result = self.quality_assessor.score_frame(frame, bbox)
        return result['composite_score']

    def get_tracks_ready_for_extraction(self) -> List[PalletTrack]:
        """Return tracks ready for final data extraction.

        Criteria:
        - Track completed (exited frame or lost)
        - Has minimum number of detections
        - Has document detections (optional - may be added later in pipeline)

        Returns:
            List of completed PalletTrack objects ready for extraction
        """
        ready_tracks = []

        min_track_length = self.config.get('tracking', {}).get('min_track_length', 10)

        for track in self.tracker.completed_tracks.values():
            # Check minimum track length
            if len(track.detections) < min_track_length:
                continue

            # Track is ready
            ready_tracks.append(track)

        return ready_tracks

    def finalize_all_tracks(self):
        """Finalize all active tracks.

        Useful for end of video processing.
        """
        self.tracker.finalize_all_tracks()

    def get_track(self, track_id: int) -> Optional[PalletTrack]:
        """Retrieve track by ID.

        Args:
            track_id: Track ID to retrieve

        Returns:
            PalletTrack object if found, None otherwise
        """
        return self.tracker.get_track(track_id)
