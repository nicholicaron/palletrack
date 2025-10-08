"""Unit tests for frame sampling module."""

import numpy as np
import pytest

from app.cv_models import BoundingBox, PalletDetection, PalletTrack, TrackStatus, DocumentDetection
from app.cv_processing import AdaptiveFrameSampler, FrameSelectionStrategy, MovementAnalyzer


@pytest.fixture
def sampling_config():
    """Fixture for frame sampling configuration."""
    return {
        'frame_sampling': {
            'movement_threshold': 50.0,
            'size_change_threshold': 0.2,
            'max_frame_gap': 30,
            'max_samples_per_track': 10,
            'min_quality_score': 0.5,
            'min_temporal_gap': 10,
            'high_velocity_threshold': 20.0,
            'high_velocity_sample_rate': 5
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
def sample_bboxes():
    """Fixture for sample bounding boxes."""
    return [
        BoundingBox(x1=100, y1=200, x2=400, y2=600, confidence=0.9),
        BoundingBox(x1=150, y1=250, x2=450, y2=650, confidence=0.9),  # Moved 50px
        BoundingBox(x1=200, y1=300, x2=500, y2=700, confidence=0.9),  # Moved 50px more
        BoundingBox(x1=210, y1=310, x2=510, y2=710, confidence=0.9),  # Moved 10px (small)
    ]


@pytest.fixture
def sample_track():
    """Fixture for sample pallet track with multiple detections."""
    detections = []
    for i in range(20):
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
            timestamp=i * 0.033,
            track_id=1
        )
        detections.append(detection)

    return PalletTrack(
        track_id=1,
        detections=detections,
        first_seen_frame=0,
        last_seen_frame=19,
        status=TrackStatus.COMPLETED
    )


@pytest.fixture
def sample_frames():
    """Fixture for sample video frames."""
    frames = {}
    for i in range(20):
        frames[i] = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
    return frames


class TestMovementAnalyzer:
    """Tests for MovementAnalyzer class."""

    def test_calculate_movement(self, sample_bboxes):
        """Test movement calculation between bboxes."""
        bbox1 = sample_bboxes[0]
        bbox2 = sample_bboxes[1]

        distance = MovementAnalyzer.calculate_movement(bbox1, bbox2)

        # Center of bbox1: (250, 400)
        # Center of bbox2: (300, 450)
        # Expected distance: sqrt(50^2 + 50^2) ≈ 70.71
        assert distance > 70
        assert distance < 71

    def test_calculate_movement_zero(self):
        """Test movement calculation for identical bboxes."""
        bbox = BoundingBox(x1=100, y1=200, x2=300, y2=400, confidence=0.9)
        distance = MovementAnalyzer.calculate_movement(bbox, bbox)
        assert distance == 0.0

    def test_calculate_size_change(self):
        """Test size change calculation."""
        bbox1 = BoundingBox(x1=100, y1=200, x2=300, y2=400, confidence=0.9)
        bbox2 = BoundingBox(x1=100, y1=200, x2=350, y2=450, confidence=0.9)

        size_change = MovementAnalyzer.calculate_size_change(bbox1, bbox2)

        # bbox1 area: 200 * 200 = 40000
        # bbox2 area: 250 * 250 = 62500
        # Change: (62500 - 40000) / 40000 = 0.5625
        assert size_change > 0.5
        assert size_change < 0.6

    def test_calculate_size_change_zero(self):
        """Test size change for identical bboxes."""
        bbox = BoundingBox(x1=100, y1=200, x2=300, y2=400, confidence=0.9)
        size_change = MovementAnalyzer.calculate_size_change(bbox, bbox)
        assert size_change == 0.0

    def test_estimate_velocity(self, sample_track):
        """Test velocity estimation from track."""
        velocity = MovementAnalyzer.estimate_velocity(sample_track, n_recent=5)

        # Track moves ~10px in x and ~10px in y per frame
        # Expected velocity: sqrt(10^2 + 10^2) ≈ 14.14 pixels/frame
        assert velocity > 13
        assert velocity < 15

    def test_estimate_velocity_single_detection(self):
        """Test velocity with single detection."""
        bbox = BoundingBox(x1=100, y1=200, x2=300, y2=400, confidence=0.9)
        detection = PalletDetection(bbox=bbox, frame_number=0, timestamp=0.0, track_id=1)
        track = PalletTrack(
            track_id=1,
            detections=[detection],
            first_seen_frame=0,
            last_seen_frame=0,
            status=TrackStatus.ACTIVE
        )

        velocity = MovementAnalyzer.estimate_velocity(track)
        assert velocity == 0.0

    def test_calculate_movement_vector(self, sample_bboxes):
        """Test movement vector calculation."""
        bbox1 = sample_bboxes[0]
        bbox2 = sample_bboxes[1]

        dx, dy = MovementAnalyzer.calculate_movement_vector(bbox1, bbox2)

        # Center1: (250, 400), Center2: (300, 450)
        # Expected: dx=50, dy=50
        assert dx == 50
        assert dy == 50

    def test_is_significant_movement(self, sample_bboxes):
        """Test significant movement check."""
        bbox1 = sample_bboxes[0]
        bbox2 = sample_bboxes[1]  # 70px movement
        bbox3 = sample_bboxes[3]  # 14px movement

        # 70px movement > 50px threshold
        assert MovementAnalyzer.is_significant_movement(bbox1, bbox2, threshold=50.0)

        # 14px movement < 50px threshold
        assert not MovementAnalyzer.is_significant_movement(bbox2, bbox3, threshold=50.0)

    def test_is_significant_size_change(self):
        """Test significant size change check."""
        bbox1 = BoundingBox(x1=100, y1=200, x2=300, y2=400, confidence=0.9)
        bbox2 = BoundingBox(x1=100, y1=200, x2=350, y2=450, confidence=0.9)  # 56% increase
        bbox3 = BoundingBox(x1=100, y1=200, x2=310, y2=410, confidence=0.9)  # 10% increase

        # 56% change > 20% threshold
        assert MovementAnalyzer.is_significant_size_change(bbox1, bbox2, threshold=0.2)

        # 10% change < 20% threshold
        assert not MovementAnalyzer.is_significant_size_change(bbox1, bbox3, threshold=0.2)

    def test_calculate_track_stability(self, sample_track):
        """Test track stability calculation."""
        stability = MovementAnalyzer.calculate_track_stability(sample_track, n_recent=10)

        # Sample track has constant velocity, should be stable
        assert stability > 0.5


class TestAdaptiveFrameSampler:
    """Tests for AdaptiveFrameSampler class."""

    def test_initialization(self, sampling_config):
        """Test sampler initialization."""
        sampler = AdaptiveFrameSampler(sampling_config)
        assert sampler.config is not None
        assert len(sampler.last_processed) == 0
        assert len(sampler.samples_per_track) == 0

    def test_first_frame_always_processed(self, sampling_config, sample_bboxes):
        """Test first frame of track is always selected."""
        sampler = AdaptiveFrameSampler(sampling_config)

        should_process = sampler.should_process_frame(
            track_id=1,
            current_bbox=sample_bboxes[0],
            frame_number=0,
            frame_quality_score=0.9
        )

        assert should_process

    def test_low_quality_rejected(self, sampling_config, sample_bboxes):
        """Test low quality frames are rejected."""
        sampler = AdaptiveFrameSampler(sampling_config)

        # Process first frame
        sampler.should_process_frame(1, sample_bboxes[0], 0, 0.9)
        sampler.mark_processed(1, sample_bboxes[0], 0)

        # Try second frame with low quality
        should_process = sampler.should_process_frame(
            track_id=1,
            current_bbox=sample_bboxes[1],
            frame_number=10,
            frame_quality_score=0.3  # Below 0.5 threshold
        )

        assert not should_process

    def test_significant_movement_triggers_processing(self, sampling_config, sample_bboxes):
        """Test significant movement triggers frame processing."""
        sampler = AdaptiveFrameSampler(sampling_config)

        # Process first frame
        sampler.should_process_frame(1, sample_bboxes[0], 0, 0.9)
        sampler.mark_processed(1, sample_bboxes[0], 0)

        # Second frame with significant movement (70px > 50px threshold)
        should_process = sampler.should_process_frame(
            track_id=1,
            current_bbox=sample_bboxes[1],
            frame_number=10,
            frame_quality_score=0.9
        )

        assert should_process

    def test_max_frame_gap_triggers_processing(self, sampling_config, sample_bboxes):
        """Test max frame gap forces processing."""
        sampler = AdaptiveFrameSampler(sampling_config)

        # Process first frame
        sampler.should_process_frame(1, sample_bboxes[0], 0, 0.9)
        sampler.mark_processed(1, sample_bboxes[0], 0)

        # Frame at max gap (no movement)
        should_process = sampler.should_process_frame(
            track_id=1,
            current_bbox=sample_bboxes[0],  # Same bbox (no movement)
            frame_number=30,  # Max gap
            frame_quality_score=0.9
        )

        assert should_process

    def test_max_samples_limit(self, sampling_config, sample_bboxes):
        """Test max samples per track is enforced."""
        sampler = AdaptiveFrameSampler(sampling_config)

        # Process max_samples_per_track frames (10)
        for i in range(10):
            sampler.mark_processed(1, sample_bboxes[0], i * 10)

        # Try to process 11th frame (should be rejected)
        should_process = sampler.should_process_frame(
            track_id=1,
            current_bbox=sample_bboxes[1],
            frame_number=100,
            frame_quality_score=0.9
        )

        assert not should_process

    def test_mark_processed(self, sampling_config, sample_bboxes):
        """Test marking frame as processed."""
        sampler = AdaptiveFrameSampler(sampling_config)

        sampler.mark_processed(1, sample_bboxes[0], 0)

        assert 1 in sampler.last_processed
        assert sampler.get_sample_count(1) == 1

    def test_get_sample_count(self, sampling_config, sample_bboxes):
        """Test getting sample count for track."""
        sampler = AdaptiveFrameSampler(sampling_config)

        assert sampler.get_sample_count(1) == 0

        sampler.mark_processed(1, sample_bboxes[0], 0)
        assert sampler.get_sample_count(1) == 1

        sampler.mark_processed(1, sample_bboxes[1], 10)
        assert sampler.get_sample_count(1) == 2

    def test_reset_track(self, sampling_config, sample_bboxes):
        """Test resetting track state."""
        sampler = AdaptiveFrameSampler(sampling_config)

        sampler.mark_processed(1, sample_bboxes[0], 0)
        assert sampler.get_sample_count(1) == 1

        sampler.reset_track(1)
        assert sampler.get_sample_count(1) == 0

    def test_reset_all(self, sampling_config, sample_bboxes):
        """Test resetting all state."""
        sampler = AdaptiveFrameSampler(sampling_config)

        sampler.mark_processed(1, sample_bboxes[0], 0)
        sampler.mark_processed(2, sample_bboxes[1], 0)

        sampler.reset_all()

        assert len(sampler.last_processed) == 0
        assert len(sampler.samples_per_track) == 0

    def test_get_statistics(self, sampling_config, sample_bboxes):
        """Test getting sampling statistics."""
        sampler = AdaptiveFrameSampler(sampling_config)

        sampler.mark_processed(1, sample_bboxes[0], 0)
        sampler.mark_processed(1, sample_bboxes[1], 10)
        sampler.mark_processed(2, sample_bboxes[0], 0)

        stats = sampler.get_statistics()

        assert stats['total_tracks'] == 2
        assert stats['total_samples'] == 3
        assert stats['avg_samples_per_track'] == 1.5
        assert stats['tracks_at_max'] == 0

    def test_high_velocity_sampling(self, sampling_config, sample_bboxes):
        """Test high velocity triggers more frequent sampling."""
        sampler = AdaptiveFrameSampler(sampling_config)

        # Process first frame
        sampler.should_process_frame(1, sample_bboxes[0], 0, 0.9)
        sampler.mark_processed(1, sample_bboxes[0], 0)

        # Small movement but high velocity
        should_process = sampler.should_process_frame(
            track_id=1,
            current_bbox=sample_bboxes[3],  # Small movement
            frame_number=5,  # At high_velocity_sample_rate
            frame_quality_score=0.9,
            track_velocity=25.0  # High velocity
        )

        assert should_process


class TestFrameSelectionStrategy:
    """Tests for FrameSelectionStrategy class."""

    def test_initialization(self, sampling_config):
        """Test strategy initialization."""
        strategy = FrameSelectionStrategy(sampling_config)
        assert strategy.sampler is not None
        assert strategy.quality_scorer is not None
        assert strategy.movement_analyzer is not None

    def test_ensure_temporal_diversity(self, sampling_config):
        """Test temporal diversity enforcement."""
        strategy = FrameSelectionStrategy(sampling_config)

        frames = [10, 15, 20, 50, 52, 100]
        diverse = strategy.ensure_temporal_diversity(frames, min_gap=10)

        # Should remove 15 (too close to 10/20) and 52 (too close to 50)
        assert 10 in diverse
        assert 20 in diverse
        assert 50 in diverse
        assert 100 in diverse
        assert 15 not in diverse
        assert 52 not in diverse

    def test_ensure_temporal_diversity_empty(self, sampling_config):
        """Test temporal diversity with empty input."""
        strategy = FrameSelectionStrategy(sampling_config)
        diverse = strategy.ensure_temporal_diversity([], min_gap=10)
        assert diverse == []

    def test_select_frames_for_track(self, sampling_config, sample_track, sample_frames):
        """Test frame selection for track."""
        strategy = FrameSelectionStrategy(sampling_config)

        selected = strategy.select_frames_for_track(sample_track, sample_frames)

        # Should select some frames
        assert len(selected) > 0

        # Should not exceed max_samples
        max_samples = sampling_config['frame_sampling']['max_samples_per_track']
        assert len(selected) <= max_samples

        # Should be sorted
        assert selected == sorted(selected)

    def test_select_frames_empty_track(self, sampling_config, sample_frames):
        """Test selection with empty track."""
        strategy = FrameSelectionStrategy(sampling_config)

        track = PalletTrack(
            track_id=1,
            detections=[],
            first_seen_frame=0,
            last_seen_frame=0,
            status=TrackStatus.COMPLETED
        )

        selected = strategy.select_frames_for_track(track, sample_frames)
        assert selected == []

    def test_select_frames_with_documents(self, sampling_config, sample_track, sample_frames):
        """Test selection prefers frames with document detections."""
        strategy = FrameSelectionStrategy(sampling_config)

        # Add document detections to specific frames
        doc_bbox = BoundingBox(x1=200, y1=300, x2=300, y2=400, confidence=0.85)
        sample_track.document_regions = [
            DocumentDetection(bbox=doc_bbox, frame_number=5),
            DocumentDetection(bbox=doc_bbox, frame_number=10),
            DocumentDetection(bbox=doc_bbox, frame_number=15),
        ]

        selected = strategy.select_frames_for_track(sample_track, sample_frames)

        # Should select frames with documents
        assert len(selected) > 0
        # All selected should have documents
        for frame_num in selected:
            assert frame_num in [5, 10, 15]

    def test_select_best_single_frame(self, sampling_config, sample_track, sample_frames):
        """Test selecting single best frame."""
        strategy = FrameSelectionStrategy(sampling_config)

        best_frame = strategy.select_best_single_frame(sample_track, sample_frames)

        # Should return a valid frame number
        assert best_frame >= 0
        assert best_frame in sample_frames

    def test_select_frames_adaptive(self, sampling_config, sample_track, sample_frames):
        """Test adaptive frame selection based on velocity."""
        strategy = FrameSelectionStrategy(sampling_config)

        selected = strategy.select_frames_adaptive(
            sample_track, sample_frames, base_samples=5
        )

        # Fast track should get more samples
        assert len(selected) > 0


class TestIntegration:
    """Integration tests for frame sampling."""

    def test_full_sampling_workflow(self, sampling_config, sample_track, sample_frames):
        """Test complete frame sampling workflow."""
        # Create components
        sampler = AdaptiveFrameSampler(sampling_config)
        strategy = FrameSelectionStrategy(sampling_config)

        # Step 1: Online sampling during tracking
        sampled_frames = []
        for detection in sample_track.detections:
            velocity = MovementAnalyzer.estimate_velocity(sample_track, n_recent=5)

            should_process = sampler.should_process_frame(
                track_id=sample_track.track_id,
                current_bbox=detection.bbox,
                frame_number=detection.frame_number,
                frame_quality_score=0.8,
                track_velocity=velocity
            )

            if should_process:
                sampler.mark_processed(
                    sample_track.track_id,
                    detection.bbox,
                    detection.frame_number
                )
                sampled_frames.append(detection.frame_number)

        # Should have sampled some frames
        assert len(sampled_frames) > 0

        # Should not exceed max samples
        max_samples = sampling_config['frame_sampling']['max_samples_per_track']
        assert len(sampled_frames) <= max_samples

        # Step 2: Offline selection from completed track
        selected = strategy.select_frames_for_track(sample_track, sample_frames)

        assert len(selected) > 0
        assert len(selected) <= max_samples
