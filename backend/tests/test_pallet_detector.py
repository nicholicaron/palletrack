"""Unit tests for pallet detection module."""

import numpy as np
import pytest
from unittest.mock import MagicMock, Mock, patch

from app.cv_models import BoundingBox, PalletDetection
from app.cv_processing import DetectionPostProcessor, DetectionVisualizer, PalletDetector


class TestPalletDetector:
    """Tests for PalletDetector class."""

    @patch("app.cv_processing.pallet_detector.YOLO")
    @patch("app.cv_processing.pallet_detector.Path")
    def test_initialization_success(self, mock_path, mock_yolo):
        """Test successful detector initialization."""
        mock_path.return_value.exists.return_value = True
        mock_model = MagicMock()
        mock_yolo.return_value = mock_model

        config = {
            "detection": {
                "pallet_model_path": "models/pallet.pt",
                "pallet_conf_threshold": 0.6,
                "pallet_iou_threshold": 0.5,
                "device": "cpu",
            }
        }

        detector = PalletDetector(config)

        assert detector.conf_threshold == 0.6
        assert detector.iou_threshold == 0.5
        assert detector.device == "cpu"
        mock_yolo.assert_called_once_with("models/pallet.pt")
        mock_model.to.assert_called_once_with("cpu")

    @patch("app.cv_processing.pallet_detector.YOLO")
    @patch("app.cv_processing.pallet_detector.Path")
    def test_initialization_default_config(self, mock_path, mock_yolo):
        """Test initialization with default config values."""
        mock_path.return_value.exists.return_value = True
        mock_model = MagicMock()
        mock_yolo.return_value = mock_model

        config = {"detection": {"pallet_model_path": "models/pallet.pt"}}

        detector = PalletDetector(config)

        assert detector.conf_threshold == 0.5  # default
        assert detector.iou_threshold == 0.45  # default
        assert detector.device == "cuda"  # default

    @patch("app.cv_processing.pallet_detector.Path")
    def test_initialization_model_not_found(self, mock_path):
        """Test that initialization fails when model file doesn't exist."""
        mock_path.return_value.exists.return_value = False

        config = {"detection": {"pallet_model_path": "models/nonexistent.pt"}}

        with pytest.raises(FileNotFoundError, match="Pallet detection model not found"):
            PalletDetector(config)

    @patch("app.cv_processing.pallet_detector.YOLO")
    @patch("app.cv_processing.pallet_detector.Path")
    def test_initialization_model_load_failure(self, mock_path, mock_yolo):
        """Test that initialization fails when model loading fails."""
        mock_path.return_value.exists.return_value = True
        mock_yolo.side_effect = Exception("Model loading error")

        config = {"detection": {"pallet_model_path": "models/pallet.pt"}}

        with pytest.raises(RuntimeError, match="Failed to load YOLO model"):
            PalletDetector(config)

    @patch("app.cv_processing.pallet_detector.YOLO")
    @patch("app.cv_processing.pallet_detector.Path")
    def test_detect_single_frame(self, mock_path, mock_yolo):
        """Test detection on a single frame."""
        mock_path.return_value.exists.return_value = True
        mock_model = MagicMock()
        mock_yolo.return_value = mock_model

        # Mock detection result
        mock_box = MagicMock()
        mock_box.xyxy = [Mock(cpu=Mock(return_value=Mock(numpy=Mock(return_value=[100, 200, 300, 400]))))]
        mock_box.conf = [Mock(cpu=Mock(return_value=Mock(numpy=Mock(return_value=0.85))))]

        mock_result = MagicMock()
        mock_result.boxes = [mock_box]
        mock_model.return_value = [mock_result]

        config = {"detection": {"pallet_model_path": "models/pallet.pt"}}
        detector = PalletDetector(config)

        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        detections = detector.detect(frame, frame_number=42, timestamp=1.4)

        assert len(detections) == 1
        det = detections[0]
        assert isinstance(det, PalletDetection)
        assert det.bbox.x1 == 100
        assert det.bbox.y1 == 200
        assert det.bbox.x2 == 300
        assert det.bbox.y2 == 400
        assert det.bbox.confidence == 0.85
        assert det.frame_number == 42
        assert det.timestamp == 1.4

    @patch("app.cv_processing.pallet_detector.YOLO")
    @patch("app.cv_processing.pallet_detector.Path")
    def test_detect_multiple_detections(self, mock_path, mock_yolo):
        """Test detection with multiple pallets in frame."""
        mock_path.return_value.exists.return_value = True
        mock_model = MagicMock()
        mock_yolo.return_value = mock_model

        # Mock two detections
        mock_box1 = MagicMock()
        mock_box1.xyxy = [Mock(cpu=Mock(return_value=Mock(numpy=Mock(return_value=[100, 100, 200, 200]))))]
        mock_box1.conf = [Mock(cpu=Mock(return_value=Mock(numpy=Mock(return_value=0.9))))]

        mock_box2 = MagicMock()
        mock_box2.xyxy = [Mock(cpu=Mock(return_value=Mock(numpy=Mock(return_value=[300, 300, 400, 400]))))]
        mock_box2.conf = [Mock(cpu=Mock(return_value=Mock(numpy=Mock(return_value=0.75))))]

        mock_result = MagicMock()
        mock_result.boxes = [mock_box1, mock_box2]
        mock_model.return_value = [mock_result]

        config = {"detection": {"pallet_model_path": "models/pallet.pt"}}
        detector = PalletDetector(config)

        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        detections = detector.detect(frame, frame_number=0, timestamp=0.0)

        assert len(detections) == 2
        assert detections[0].bbox.confidence == 0.9
        assert detections[1].bbox.confidence == 0.75

    @patch("app.cv_processing.pallet_detector.YOLO")
    @patch("app.cv_processing.pallet_detector.Path")
    def test_detect_no_detections(self, mock_path, mock_yolo):
        """Test detection when no pallets found."""
        mock_path.return_value.exists.return_value = True
        mock_model = MagicMock()
        mock_yolo.return_value = mock_model

        mock_result = MagicMock()
        mock_result.boxes = []
        mock_model.return_value = [mock_result]

        config = {"detection": {"pallet_model_path": "models/pallet.pt"}}
        detector = PalletDetector(config)

        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        detections = detector.detect(frame, frame_number=0, timestamp=0.0)

        assert len(detections) == 0

    @patch("app.cv_processing.pallet_detector.YOLO")
    @patch("app.cv_processing.pallet_detector.Path")
    def test_detect_batch(self, mock_path, mock_yolo):
        """Test batch detection."""
        mock_path.return_value.exists.return_value = True
        mock_model = MagicMock()
        mock_yolo.return_value = mock_model

        # Mock results for 3 frames
        def create_mock_box(coords, conf):
            mock_box = MagicMock()
            mock_box.xyxy = [Mock(cpu=Mock(return_value=Mock(numpy=Mock(return_value=coords))))]
            mock_box.conf = [Mock(cpu=Mock(return_value=Mock(numpy=Mock(return_value=conf))))]
            return mock_box

        # Frame 1: 1 detection
        mock_result1 = MagicMock()
        mock_result1.boxes = [create_mock_box([100, 100, 200, 200], 0.8)]

        # Frame 2: 2 detections
        mock_result2 = MagicMock()
        mock_result2.boxes = [
            create_mock_box([150, 150, 250, 250], 0.85),
            create_mock_box([300, 300, 400, 400], 0.75),
        ]

        # Frame 3: 0 detections
        mock_result3 = MagicMock()
        mock_result3.boxes = []

        mock_model.return_value = [mock_result1, mock_result2, mock_result3]

        config = {"detection": {"pallet_model_path": "models/pallet.pt"}}
        detector = PalletDetector(config)

        frames = [
            np.zeros((1080, 1920, 3), dtype=np.uint8),
            np.zeros((1080, 1920, 3), dtype=np.uint8),
            np.zeros((1080, 1920, 3), dtype=np.uint8),
        ]
        frame_numbers = [0, 1, 2]
        timestamps = [0.0, 0.033, 0.066]

        batch_detections = detector.detect_batch(frames, frame_numbers, timestamps)

        assert len(batch_detections) == 3
        assert len(batch_detections[0]) == 1  # Frame 0: 1 detection
        assert len(batch_detections[1]) == 2  # Frame 1: 2 detections
        assert len(batch_detections[2]) == 0  # Frame 2: 0 detections

        # Check metadata
        assert batch_detections[0][0].frame_number == 0
        assert batch_detections[1][0].frame_number == 1

    @patch("app.cv_processing.pallet_detector.YOLO")
    @patch("app.cv_processing.pallet_detector.Path")
    def test_detect_batch_length_mismatch(self, mock_path, mock_yolo):
        """Test batch detection with mismatched input lengths."""
        mock_path.return_value.exists.return_value = True
        mock_model = MagicMock()
        mock_yolo.return_value = mock_model

        config = {"detection": {"pallet_model_path": "models/pallet.pt"}}
        detector = PalletDetector(config)

        frames = [np.zeros((1080, 1920, 3), dtype=np.uint8)] * 3
        frame_numbers = [0, 1]  # Wrong length
        timestamps = [0.0, 0.033, 0.066]

        with pytest.raises(ValueError, match="must have same length"):
            detector.detect_batch(frames, frame_numbers, timestamps)

    @patch("app.cv_processing.pallet_detector.YOLO")
    @patch("app.cv_processing.pallet_detector.Path")
    def test_detect_batch_empty(self, mock_path, mock_yolo):
        """Test batch detection with empty input."""
        mock_path.return_value.exists.return_value = True
        mock_model = MagicMock()
        mock_yolo.return_value = mock_model

        config = {"detection": {"pallet_model_path": "models/pallet.pt"}}
        detector = PalletDetector(config)

        batch_detections = detector.detect_batch([], [], [])

        assert batch_detections == []


class TestDetectionPostProcessor:
    """Tests for DetectionPostProcessor class."""

    def test_filter_overlapping_detections(self):
        """Test NMS filtering of overlapping detections."""
        # Create overlapping detections
        det1 = PalletDetection(
            bbox=BoundingBox(x1=100, y1=100, x2=200, y2=200, confidence=0.9),
            frame_number=0,
            timestamp=0.0,
        )
        det2 = PalletDetection(
            bbox=BoundingBox(x1=110, y1=110, x2=210, y2=210, confidence=0.8),  # Overlaps with det1
            frame_number=0,
            timestamp=0.0,
        )
        det3 = PalletDetection(
            bbox=BoundingBox(x1=300, y1=300, x2=400, y2=400, confidence=0.85),  # No overlap
            frame_number=0,
            timestamp=0.0,
        )

        detections = [det1, det2, det3]
        filtered = DetectionPostProcessor.filter_overlapping_detections(detections, iou_threshold=0.5)

        # Should keep det1 (highest conf) and det3 (no overlap)
        assert len(filtered) == 2
        assert det1 in filtered
        assert det3 in filtered
        assert det2 not in filtered

    def test_filter_overlapping_no_overlap(self):
        """Test NMS when no detections overlap."""
        det1 = PalletDetection(
            bbox=BoundingBox(x1=100, y1=100, x2=200, y2=200, confidence=0.9),
            frame_number=0,
            timestamp=0.0,
        )
        det2 = PalletDetection(
            bbox=BoundingBox(x1=300, y1=300, x2=400, y2=400, confidence=0.8),
            frame_number=0,
            timestamp=0.0,
        )

        detections = [det1, det2]
        filtered = DetectionPostProcessor.filter_overlapping_detections(detections)

        assert len(filtered) == 2  # Both kept

    def test_filter_overlapping_empty(self):
        """Test NMS with empty input."""
        filtered = DetectionPostProcessor.filter_overlapping_detections([])
        assert filtered == []

    def test_filter_by_size(self):
        """Test size-based filtering."""
        # Small detection (50x50 = 2500 pixels)
        det1 = PalletDetection(
            bbox=BoundingBox(x1=0, y1=0, x2=50, y2=50, confidence=0.9),
            frame_number=0,
            timestamp=0.0,
        )
        # Medium detection (200x200 = 40000 pixels)
        det2 = PalletDetection(
            bbox=BoundingBox(x1=0, y1=0, x2=200, y2=200, confidence=0.9),
            frame_number=0,
            timestamp=0.0,
        )
        # Large detection (1000x1000 = 1000000 pixels)
        det3 = PalletDetection(
            bbox=BoundingBox(x1=0, y1=0, x2=1000, y2=1000, confidence=0.9),
            frame_number=0,
            timestamp=0.0,
        )

        detections = [det1, det2, det3]
        filtered = DetectionPostProcessor.filter_by_size(
            detections, min_area=10000, max_area=500000
        )

        # Only det2 should pass (40000 pixels)
        assert len(filtered) == 1
        assert filtered[0] == det2

    def test_filter_by_size_all_pass(self):
        """Test size filtering when all detections pass."""
        det1 = PalletDetection(
            bbox=BoundingBox(x1=0, y1=0, x2=200, y2=200, confidence=0.9),
            frame_number=0,
            timestamp=0.0,
        )
        det2 = PalletDetection(
            bbox=BoundingBox(x1=0, y1=0, x2=300, y2=300, confidence=0.9),
            frame_number=0,
            timestamp=0.0,
        )

        detections = [det1, det2]
        filtered = DetectionPostProcessor.filter_by_size(detections, min_area=1000, max_area=1000000)

        assert len(filtered) == 2

    def test_filter_by_confidence(self):
        """Test confidence-based filtering."""
        det1 = PalletDetection(
            bbox=BoundingBox(x1=0, y1=0, x2=100, y2=100, confidence=0.9),
            frame_number=0,
            timestamp=0.0,
        )
        det2 = PalletDetection(
            bbox=BoundingBox(x1=0, y1=0, x2=100, y2=100, confidence=0.6),
            frame_number=0,
            timestamp=0.0,
        )
        det3 = PalletDetection(
            bbox=BoundingBox(x1=0, y1=0, x2=100, y2=100, confidence=0.4),
            frame_number=0,
            timestamp=0.0,
        )

        detections = [det1, det2, det3]
        filtered = DetectionPostProcessor.filter_by_confidence(detections, min_confidence=0.7)

        assert len(filtered) == 1
        assert filtered[0] == det1


class TestDetectionVisualizer:
    """Tests for DetectionVisualizer class."""

    def test_draw_detections_basic(self):
        """Test basic detection visualization."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        det = PalletDetection(
            bbox=BoundingBox(x1=100, y1=100, x2=200, y2=200, confidence=0.85),
            frame_number=0,
            timestamp=0.0,
        )

        annotated = DetectionVisualizer.draw_detections(frame, [det])

        # Check that frame was not modified (returned copy)
        assert not np.array_equal(frame, annotated)

        # Check that original frame is still all zeros
        assert np.all(frame == 0)

        # Check that annotated frame has non-zero pixels (boxes drawn)
        assert np.any(annotated > 0)

    def test_draw_detections_multiple(self):
        """Test visualization with multiple detections."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        det1 = PalletDetection(
            bbox=BoundingBox(x1=50, y1=50, x2=150, y2=150, confidence=0.9),
            frame_number=0,
            timestamp=0.0,
        )
        det2 = PalletDetection(
            bbox=BoundingBox(x1=200, y1=200, x2=300, y2=300, confidence=0.75),
            frame_number=0,
            timestamp=0.0,
        )

        annotated = DetectionVisualizer.draw_detections(frame, [det1, det2])

        assert np.any(annotated > 0)

    def test_draw_detections_no_confidence(self):
        """Test visualization without confidence labels."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        det = PalletDetection(
            bbox=BoundingBox(x1=100, y1=100, x2=200, y2=200, confidence=0.85),
            frame_number=0,
            timestamp=0.0,
        )

        annotated = DetectionVisualizer.draw_detections(frame, [det], show_confidence=False)

        assert np.any(annotated > 0)

    def test_draw_detections_empty(self):
        """Test visualization with no detections."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        annotated = DetectionVisualizer.draw_detections(frame, [])

        # Should return copy of original frame
        assert np.array_equal(frame, annotated)
        assert frame is not annotated  # But not the same object

    def test_draw_detections_custom_color(self):
        """Test visualization with custom color."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        det = PalletDetection(
            bbox=BoundingBox(x1=100, y1=100, x2=200, y2=200, confidence=0.85),
            frame_number=0,
            timestamp=0.0,
        )

        # Draw in red
        annotated = DetectionVisualizer.draw_detections(frame, [det], color=(0, 0, 255))

        # Check that red channel has values
        assert np.any(annotated[:, :, 2] > 0)

    def test_draw_detection_grid(self):
        """Test grid visualization."""
        frames = [np.zeros((100, 100, 3), dtype=np.uint8) for _ in range(4)]

        det = PalletDetection(
            bbox=BoundingBox(x1=20, y1=20, x2=80, y2=80, confidence=0.9),
            frame_number=0,
            timestamp=0.0,
        )
        detections_per_frame = [[det], [det], [], [det]]

        grid = DetectionVisualizer.draw_detection_grid(
            frames, detections_per_frame, grid_cols=2
        )

        # Grid should be 2x2 (4 frames, 2 cols)
        assert grid.shape == (200, 200, 3)  # 2 rows x 2 cols, each 100x100

    def test_draw_detection_grid_mismatch(self):
        """Test grid visualization with mismatched inputs."""
        frames = [np.zeros((100, 100, 3), dtype=np.uint8) for _ in range(3)]
        detections_per_frame = [[], []]  # Wrong length

        with pytest.raises(ValueError, match="must have same length"):
            DetectionVisualizer.draw_detection_grid(frames, detections_per_frame)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
