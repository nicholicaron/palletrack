"""Unit tests for pallet tracking module."""

import numpy as np
import pytest

from app.cv_models import BoundingBox, PalletDetection, PalletTrack, TrackStatus
from app.cv_processing import PalletTracker, TrackManager
from app.cv_processing.track_utils import (
    TrackVisualizer,
    calculate_track_size_change,
    calculate_track_velocity,
    get_track_summary,
)


@pytest.fixture
def tracking_config():
    """Fixture for tracking configuration."""
    return {
        'tracking': {
            'max_age': 30,
            'min_hits': 3,
            'iou_threshold': 0.3,
            'min_track_length': 10,
            'frames_between_ocr': 15
        },
        'frame_quality': {
            'min_sharpness': 100.0,
            'min_size_score': 0.3,
            'sharpness_weight': 0.5,
            'size_weight': 0.3,
            'angle_weight': 0.2,
            'frames_to_select': 5
        }
    }


@pytest.fixture
def sample_detections():
    """Fixture for sample pallet detections."""
    detections = []
    for i in range(5):
        bbox = BoundingBox(
            x1=100 + i * 10,
            y1=200 + i * 10,
            x2=400 + i * 10,
            y2=600 + i * 10,
            confidence=0.9
        )
        detection = PalletDetection(
            bbox=bbox,
            frame_number=i,
            timestamp=i * 0.033
        )
        detections.append(detection)
    return detections


class TestPalletTracker:
    """Tests for PalletTracker class."""

    def test_initialization(self, tracking_config):
        """Test tracker initialization."""
        tracker = PalletTracker(tracking_config)
        assert tracker.tracker is not None
        assert len(tracker.active_tracks) == 0
        assert len(tracker.completed_tracks) == 0

    def test_single_track_creation(self, tracking_config, sample_detections):
        """Test creating a single track from detections."""
        tracker = PalletTracker(tracking_config)

        # Process first detection
        active = tracker.update([sample_detections[0]], frame_number=0)

        # Track should be created
        assert len(active) == 1
        track_id = list(active.keys())[0]
        assert track_id >= 0
        assert len(active[track_id].detections) == 1

    def test_track_continuation(self, tracking_config, sample_detections):
        """Test track continues across multiple frames."""
        tracker = PalletTracker(tracking_config)

        # Process multiple detections
        track_id = None
        for i, detection in enumerate(sample_detections):
            active = tracker.update([detection], frame_number=i)
            if track_id is None:
                track_id = list(active.keys())[0]

        # Should still have same track
        assert track_id in tracker.active_tracks
        assert len(tracker.active_tracks[track_id].detections) == len(sample_detections)

    def test_empty_frame_handling(self, tracking_config, sample_detections):
        """Test tracker handles frames with no detections."""
        tracker = PalletTracker(tracking_config)

        # Create track
        active = tracker.update([sample_detections[0]], frame_number=0)
        track_id = list(active.keys())[0]

        # Process empty frame
        active = tracker.update([], frame_number=1)

        # Track should still be active (within max_age)
        assert track_id in tracker.active_tracks

    def test_track_finalization(self, tracking_config, sample_detections):
        """Test moving track from active to completed."""
        tracker = PalletTracker(tracking_config)

        # Create track
        active = tracker.update([sample_detections[0]], frame_number=0)
        track_id = list(active.keys())[0]

        # Finalize track
        tracker.finalize_track(track_id)

        # Should be in completed tracks
        assert track_id not in tracker.active_tracks
        assert track_id in tracker.completed_tracks
        assert tracker.completed_tracks[track_id].status == TrackStatus.COMPLETED

    def test_get_track(self, tracking_config, sample_detections):
        """Test retrieving track by ID."""
        tracker = PalletTracker(tracking_config)

        # Create track
        active = tracker.update([sample_detections[0]], frame_number=0)
        track_id = list(active.keys())[0]

        # Get active track
        track = tracker.get_track(track_id)
        assert track is not None
        assert track.track_id == track_id

        # Finalize and get completed track
        tracker.finalize_track(track_id)
        track = tracker.get_track(track_id)
        assert track is not None
        assert track.status == TrackStatus.COMPLETED

    def test_get_nonexistent_track(self, tracking_config):
        """Test retrieving non-existent track returns None."""
        tracker = PalletTracker(tracking_config)
        track = tracker.get_track(999)
        assert track is None

    def test_multiple_tracks(self, tracking_config):
        """Test tracking multiple pallets simultaneously."""
        tracker = PalletTracker(tracking_config)

        # Create two detections in different locations
        bbox1 = BoundingBox(x1=100, y1=200, x2=400, y2=600, confidence=0.9)
        bbox2 = BoundingBox(x1=500, y1=200, x2=800, y2=600, confidence=0.85)

        det1 = PalletDetection(bbox=bbox1, frame_number=0, timestamp=0.0)
        det2 = PalletDetection(bbox=bbox2, frame_number=0, timestamp=0.0)

        # Update tracker
        active = tracker.update([det1, det2], frame_number=0)

        # Should have two tracks
        assert len(active) == 2

    def test_finalize_all_tracks(self, tracking_config, sample_detections):
        """Test finalizing all active tracks."""
        tracker = PalletTracker(tracking_config)

        # Create tracks
        tracker.update([sample_detections[0]], frame_number=0)
        tracker.update([sample_detections[1]], frame_number=1)

        initial_active_count = len(tracker.active_tracks)
        assert initial_active_count > 0

        # Finalize all
        tracker.finalize_all_tracks()

        # All should be completed
        assert len(tracker.active_tracks) == 0
        assert len(tracker.completed_tracks) == initial_active_count


class TestTrackManager:
    """Tests for TrackManager class."""

    def test_initialization(self, tracking_config):
        """Test manager initialization."""
        manager = TrackManager(tracking_config)
        assert manager.tracker is not None
        assert manager.config is not None
        assert manager.quality_assessor is not None

    def test_update(self, tracking_config, sample_detections):
        """Test updating tracks through manager."""
        manager = TrackManager(tracking_config)

        active = manager.update([sample_detections[0]], frame_number=0)
        assert len(active) == 1

    def test_should_process_track_min_length(self, tracking_config, sample_detections):
        """Test should_process_track respects minimum track length."""
        manager = TrackManager(tracking_config)

        # Create track with few detections
        for i in range(5):
            manager.update([sample_detections[i]], frame_number=i)

        track_id = list(manager.tracker.active_tracks.keys())[0]

        # Should not process (less than min_track_length)
        should_process = manager.should_process_track(track_id, frame_number=5)
        assert not should_process

    def test_should_process_track_sufficient_length(self, tracking_config, sample_detections):
        """Test should_process_track returns True for sufficient track length."""
        manager = TrackManager(tracking_config)

        # Create track with enough detections
        for i in range(12):
            bbox = BoundingBox(
                x1=100 + i * 5,
                y1=200 + i * 5,
                x2=400 + i * 5,
                y2=600 + i * 5,
                confidence=0.9
            )
            detection = PalletDetection(bbox=bbox, frame_number=i, timestamp=i * 0.033)
            manager.update([detection], frame_number=i)

        track_id = list(manager.tracker.active_tracks.keys())[0]

        # Should process (meets min_track_length)
        should_process = manager.should_process_track(track_id, frame_number=12)
        assert should_process

    def test_should_process_track_frames_between_ocr(self, tracking_config, sample_detections):
        """Test should_process_track respects frames_between_ocr."""
        manager = TrackManager(tracking_config)

        # Create track
        for i in range(15):
            bbox = BoundingBox(
                x1=100 + i * 5,
                y1=200 + i * 5,
                x2=400 + i * 5,
                y2=600 + i * 5,
                confidence=0.9
            )
            detection = PalletDetection(bbox=bbox, frame_number=i, timestamp=i * 0.033)
            manager.update([detection], frame_number=i)

        track_id = list(manager.tracker.active_tracks.keys())[0]

        # First OCR should succeed
        should_process_1 = manager.should_process_track(track_id, frame_number=15)
        assert should_process_1

        # Second OCR too soon should fail
        should_process_2 = manager.should_process_track(track_id, frame_number=20)
        assert not should_process_2

        # Third OCR after enough frames should succeed
        should_process_3 = manager.should_process_track(track_id, frame_number=31)
        assert should_process_3

    def test_get_tracks_ready_for_extraction(self, tracking_config, sample_detections):
        """Test getting tracks ready for extraction."""
        manager = TrackManager(tracking_config)

        # Create tracks
        for i in range(15):
            bbox = BoundingBox(
                x1=100 + i * 5,
                y1=200 + i * 5,
                x2=400 + i * 5,
                y2=600 + i * 5,
                confidence=0.9
            )
            detection = PalletDetection(bbox=bbox, frame_number=i, timestamp=i * 0.033)
            manager.update([detection], frame_number=i)

        # Finalize tracks
        manager.finalize_all_tracks()

        # Get ready tracks
        ready = manager.get_tracks_ready_for_extraction()
        assert len(ready) > 0

    def test_get_track(self, tracking_config, sample_detections):
        """Test getting track through manager."""
        manager = TrackManager(tracking_config)

        active = manager.update([sample_detections[0]], frame_number=0)
        track_id = list(active.keys())[0]

        track = manager.get_track(track_id)
        assert track is not None
        assert track.track_id == track_id


class TestTrackVisualizer:
    """Tests for TrackVisualizer class."""

    def test_initialization(self):
        """Test visualizer initialization."""
        visualizer = TrackVisualizer(max_trail_length=30)
        assert visualizer.max_trail_length == 30
        assert len(visualizer.trail_history) == 0

    def test_draw_tracks_basic(self, sample_detections):
        """Test basic track drawing."""
        # Create test frame
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)

        # Create track
        track = PalletTrack(
            track_id=1,
            detections=[sample_detections[0]],
            first_seen_frame=0,
            last_seen_frame=0,
            status=TrackStatus.ACTIVE
        )

        # Draw tracks
        vis_frame = TrackVisualizer.draw_tracks(
            frame,
            {1: track},
            show_trail=False,
            show_info=True
        )

        # Should return frame with same shape
        assert vis_frame.shape == frame.shape

        # Frame should have been modified (not all zeros)
        assert not np.all(vis_frame == 0)

    def test_draw_tracks_with_trail(self, sample_detections):
        """Test track drawing with trail."""
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)

        # Create track with multiple detections
        track = PalletTrack(
            track_id=1,
            detections=sample_detections,
            first_seen_frame=0,
            last_seen_frame=len(sample_detections) - 1,
            status=TrackStatus.ACTIVE
        )

        # Draw with trail
        vis_frame = TrackVisualizer.draw_tracks(
            frame,
            {1: track},
            show_trail=True,
            show_info=True
        )

        assert vis_frame.shape == frame.shape
        assert not np.all(vis_frame == 0)

    def test_draw_track_stats(self):
        """Test drawing track statistics."""
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)

        vis_frame = TrackVisualizer.draw_track_stats(
            frame,
            active_count=5,
            completed_count=10,
            frame_number=100
        )

        assert vis_frame.shape == frame.shape
        assert not np.all(vis_frame == 0)


class TestTrackUtilities:
    """Tests for track utility functions."""

    def test_calculate_track_velocity(self, sample_detections):
        """Test velocity calculation."""
        track = PalletTrack(
            track_id=1,
            detections=sample_detections,
            first_seen_frame=0,
            last_seen_frame=len(sample_detections) - 1,
            status=TrackStatus.ACTIVE
        )

        velocity_x, velocity_y = calculate_track_velocity(track, num_frames=5)

        # Velocity should be non-zero (track is moving)
        assert velocity_x != 0.0 or velocity_y != 0.0

    def test_calculate_track_velocity_single_detection(self):
        """Test velocity with single detection."""
        bbox = BoundingBox(x1=100, y1=200, x2=400, y2=600, confidence=0.9)
        detection = PalletDetection(bbox=bbox, frame_number=0, timestamp=0.0)
        track = PalletTrack(
            track_id=1,
            detections=[detection],
            first_seen_frame=0,
            last_seen_frame=0,
            status=TrackStatus.ACTIVE
        )

        velocity_x, velocity_y = calculate_track_velocity(track)
        assert velocity_x == 0.0
        assert velocity_y == 0.0

    def test_calculate_track_size_change(self, sample_detections):
        """Test size change calculation."""
        track = PalletTrack(
            track_id=1,
            detections=sample_detections,
            first_seen_frame=0,
            last_seen_frame=len(sample_detections) - 1,
            status=TrackStatus.ACTIVE
        )

        size_change = calculate_track_size_change(track, num_frames=5)
        assert size_change >= 0.0

    def test_get_track_summary(self, sample_detections):
        """Test track summary generation."""
        track = PalletTrack(
            track_id=1,
            detections=sample_detections,
            first_seen_frame=0,
            last_seen_frame=len(sample_detections) - 1,
            status=TrackStatus.ACTIVE
        )

        summary = get_track_summary(track)

        assert summary['track_id'] == 1
        assert summary['num_detections'] == len(sample_detections)
        assert summary['duration_frames'] > 0
        assert 'avg_confidence' in summary
        assert 'velocity_x' in summary
        assert 'velocity_y' in summary

    def test_get_track_summary_empty(self):
        """Test summary with empty track."""
        track = PalletTrack(
            track_id=1,
            detections=[],
            first_seen_frame=0,
            last_seen_frame=0,
            status=TrackStatus.ACTIVE
        )

        summary = get_track_summary(track)
        assert summary['num_detections'] == 0
        assert summary['duration_frames'] == 0
