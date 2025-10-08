"""Integration tests for the complete PalletTrack pipeline."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from app.cv_models import (
    BoundingBox,
    DocumentDetection,
    DocumentType,
    ExtractedShippingData,
    OCRResult,
    PalletDetection,
    PalletTrack,
    TrackStatus,
)
from app.pipeline import (
    FrameAnnotator,
    FPSTracker,
    PalletScannerPipeline,
    PipelineMonitor,
    ResultsExporter,
    VideoStreamProcessor,
)


class TestPipelineIntegration:
    """Test complete pipeline integration."""

    @pytest.fixture
    def config(self):
        """Minimal test configuration."""
        return {
            'detection': {
                'pallet_model_path': 'models/pallet_yolov8n.pt',
                'pallet_conf_threshold': 0.5,
                'pallet_iou_threshold': 0.45,
                'document_model_path': 'models/document_yolov8n.pt',
                'document_conf_threshold': 0.6,
                'device': 'cpu',
                'min_pallet_area': 10000,
                'max_pallet_area': 500000,
                'max_association_distance': 100,
                'containment_confidence': 0.9,
                'proximity_confidence': 0.7,
            },
            'tracking': {
                'max_age': 30,
                'min_hits': 3,
                'iou_threshold': 0.3,
                'min_track_length': 2,
                'frames_between_ocr': 5,
            },
            'frame_sampling': {
                'movement_threshold': 50.0,
                'size_change_threshold': 0.2,
                'max_frame_gap': 30,
                'max_samples_per_track': 10,
                'min_quality_score': 0.5,
                'min_temporal_gap': 10,
            },
            'frame_quality': {
                'min_sharpness': 100.0,
                'min_size_score': 0.3,
                'sharpness_weight': 0.5,
                'size_weight': 0.3,
                'angle_weight': 0.2,
                'frames_to_select': 5,
            },
            'ocr': {
                'language': 'en',
                'use_gpu': False,
                'min_text_confidence': 0.6,
                'preprocessing': {
                    'apply_clahe': True,
                    'clahe_clip_limit': 2.0,
                    'clahe_grid_size': 8,
                    'bilateral_filter': True,
                    'remove_glare': True,
                    'glare_threshold': 240,
                },
                'aggregation': {
                    'method': 'voting',
                    'min_frames_for_consensus': 2,
                },
            },
            'data_extraction': {
                'classification': {'min_confidence': 0.6},
                'validation': {
                    'require_tracking_number': True,
                    'min_weight': 1.0,
                    'max_weight': 5000.0,
                    'validate_addresses': True,
                },
            },
            'confidence': {
                'auto_accept': 0.85,
                'needs_review': 0.60,
                'auto_reject': 0.40,
                'weights': {
                    'detection_conf': 0.15,
                    'ocr_conf': 0.25,
                    'field_completeness': 0.25,
                    'cross_frame_consistency': 0.20,
                    'data_validation': 0.15,
                },
            },
            'review_queue': {
                'priority_order': 'confidence',
                'max_queue_size': 1000,
                'save_frames': False,
                'frame_save_path': 'review_frames',
            },
        }

    @pytest.fixture
    def test_frame(self):
        """Create a test frame."""
        # Create 640x480 BGR image
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Add some content
        cv2.rectangle(frame, (100, 100), (300, 300), (0, 255, 0), -1)
        return frame

    def test_pipeline_initialization(self, tmp_path):
        """Test pipeline initialization."""
        # Create temp config file
        config_path = tmp_path / "config.yaml"
        import yaml

        with open(config_path, 'w') as f:
            yaml.dump(
                {
                    'detection': {'pallet_model_path': 'models/pallet_yolov8n.pt'},
                    'tracking': {},
                    'frame_sampling': {},
                    'ocr': {},
                    'confidence': {},
                    'review_queue': {},
                },
                f,
            )

        # Mock component initialization to avoid loading real models
        with patch('app.pipeline.main_pipeline.PalletDetector'), \
             patch('app.pipeline.main_pipeline.PalletTracker'), \
             patch('app.pipeline.main_pipeline.DocumentDetector'), \
             patch('app.pipeline.main_pipeline.AdaptiveFrameSampler'), \
             patch('app.pipeline.main_pipeline.DocumentOCR'), \
             patch('app.pipeline.main_pipeline.ShippingDataExtractor'):

            pipeline = PalletScannerPipeline(str(config_path))

            assert pipeline.config is not None
            assert pipeline.stats['total_frames_processed'] == 0
            assert len(pipeline.active_tracks) == 0

    def test_fps_tracker(self):
        """Test FPS tracking."""
        tracker = FPSTracker(window_size=5)

        # Simulate processing frames
        import time

        for _ in range(10):
            tracker.update()
            time.sleep(0.01)  # Small delay

        fps = tracker.get_fps()
        assert fps > 0
        assert fps < 200  # Should be reasonable

    def test_frame_annotator(self, test_frame):
        """Test frame annotation."""
        # Create test detections
        pallet_det = PalletDetection(
            bbox=BoundingBox(x1=100, y1=100, x2=300, y2=300, confidence=0.9),
            track_id=1,
        )

        doc_det = DocumentDetection(
            bbox=BoundingBox(x1=150, y1=150, x2=250, y2=250, confidence=0.85),
            parent_pallet_track_id=1,
        )

        track = PalletTrack(track_id=1, initial_detection=pallet_det, frame_number=0)

        # Annotate frame
        annotated = FrameAnnotator.annotate_complete_frame(
            frame=test_frame,
            pallet_detections=[pallet_det],
            active_tracks={1: track},
            document_detections=[doc_det],
            ocr_processed=True,
        )

        # Check that annotation modified the frame
        assert not np.array_equal(annotated, test_frame)
        assert annotated.shape == test_frame.shape

    def test_results_exporter_json(self, tmp_path):
        """Test JSON export."""
        # Create test extraction
        extraction = ExtractedShippingData(
            track_id=1,
            document_type=DocumentType.BOL.value,
            tracking_number="1Z9999999999999999",
            weight=150.5,
            destination_city="New York",
            destination_state="NY",
            destination_zip="10001",
        )

        output_path = tmp_path / "results.json"
        ResultsExporter.export_to_json([extraction], str(output_path))

        # Check file was created
        assert output_path.exists()

        # Load and verify
        import json

        with open(output_path) as f:
            data = json.load(f)

        assert len(data) == 1
        assert data[0]['tracking_number'] == "1Z9999999999999999"
        assert data[0]['weight'] == 150.5

    def test_results_exporter_csv(self, tmp_path):
        """Test CSV export."""
        # Create test extractions
        extractions = [
            ExtractedShippingData(
                track_id=1,
                document_type=DocumentType.BOL.value,
                tracking_number="1Z9999999999999999",
                weight=150.5,
            ),
            ExtractedShippingData(
                track_id=2,
                document_type=DocumentType.SHIPPING_LABEL.value,
                tracking_number="1Z8888888888888888",
                weight=200.0,
            ),
        ]

        output_path = tmp_path / "results.csv"
        ResultsExporter.export_to_csv(extractions, str(output_path))

        # Check file was created
        assert output_path.exists()

        # Load and verify
        import csv

        with open(output_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 2
        assert rows[0]['tracking_number'] == "1Z9999999999999999"
        assert rows[1]['tracking_number'] == "1Z8888888888888888"

    def test_pipeline_monitor(self, tmp_path):
        """Test pipeline monitoring."""
        monitor = PipelineMonitor(log_dir=str(tmp_path), log_level="INFO")

        # Log some events
        monitor.log_frame_processing(0, {'ocr_processed': True})
        monitor.log_frame_processing(1, {'ocr_processed': False})

        extraction = ExtractedShippingData(
            track_id=1, document_type=DocumentType.BOL.value
        )
        monitor.log_extraction(extraction, 0.85, 'AUTO_ACCEPT')

        # Generate report
        report = monitor.generate_performance_report()

        assert report['frames_processed'] == 2
        assert report['total_extractions'] == 1
        assert report['extraction_routes']['AUTO_ACCEPT'] == 1

    def test_wms_payload_format(self):
        """Test WMS payload formatting."""
        extraction = ExtractedShippingData(
            track_id=1,
            document_type=DocumentType.BOL.value,
            tracking_number="1Z9999999999999999",
            po_number="PO-12345",
            weight=150.5,
            destination_address="123 Main St",
            destination_city="New York",
            destination_state="NY",
            destination_zip="10001",
            carrier="UPS",
            confidence_score=0.92,
        )

        payload = ResultsExporter.create_wms_payload(extraction)

        assert payload['pallet_id'] == 1
        assert payload['shipment_info']['tracking_number'] == "1Z9999999999999999"
        assert payload['shipment_info']['po_number'] == "PO-12345"
        assert payload['weight']['value'] == 150.5
        assert payload['destination']['city'] == "New York"
        assert payload['metadata']['confidence_score'] == 0.92

    def test_info_panel_annotation(self, test_frame):
        """Test info panel annotation."""
        annotated = FrameAnnotator.add_info_panel(
            frame=test_frame,
            fps=30.5,
            active_tracks=3,
            processed_extractions=10,
            frame_number=100,
        )

        # Check that annotation modified the frame
        assert not np.array_equal(annotated, test_frame)
        assert annotated.shape == test_frame.shape

    def test_side_by_side_visualization(self, test_frame):
        """Test side-by-side frame comparison."""
        frame1 = test_frame.copy()
        frame2 = test_frame.copy()

        # Modify frame2
        cv2.rectangle(frame2, (200, 200), (400, 400), (0, 0, 255), -1)

        combined = FrameAnnotator.create_side_by_side(
            frame1, frame2, label1="Original", label2="Processed"
        )

        # Check dimensions
        assert combined.shape[0] == test_frame.shape[0]  # Same height
        assert combined.shape[1] > test_frame.shape[1]  # Wider

    def test_pipeline_finalize(self, tmp_path):
        """Test pipeline finalization."""
        # Create temp config
        config_path = tmp_path / "config.yaml"
        import yaml

        with open(config_path, 'w') as f:
            yaml.dump(
                {
                    'detection': {},
                    'tracking': {},
                    'frame_sampling': {},
                    'ocr': {},
                    'data_extraction': {},
                    'confidence': {},
                    'review_queue': {},
                },
                f,
            )

        # Mock all components
        with patch('app.pipeline.main_pipeline.PalletDetector'), \
             patch('app.pipeline.main_pipeline.PalletTracker'), \
             patch('app.pipeline.main_pipeline.DocumentDetector'), \
             patch('app.pipeline.main_pipeline.AdaptiveFrameSampler'), \
             patch('app.pipeline.main_pipeline.DocumentOCR'), \
             patch('app.pipeline.main_pipeline.ShippingDataExtractor'), \
             patch('app.pipeline.main_pipeline.ConfidenceCalculator'), \
             patch('app.pipeline.main_pipeline.ReviewQueueManager'), \
             patch('app.pipeline.main_pipeline.QualityMetricsTracker'):

            pipeline = PalletScannerPipeline(str(config_path))

            # Add a mock active track
            pallet_det = PalletDetection(
                bbox=BoundingBox(x1=100, y1=100, x2=300, y2=300, confidence=0.9),
                track_id=1,
            )
            track = PalletTrack(track_id=1, initial_detection=pallet_det, frame_number=0)
            pipeline.active_tracks[1] = track

            # Finalize should mark tracks as lost
            pipeline.finalize()

            # Track should be marked as lost
            assert track.status == TrackStatus.LOST


class TestVideoProcessorIntegration:
    """Test video processor integration."""

    def test_create_test_video(self, tmp_path):
        """Helper to create a test video file."""
        video_path = tmp_path / "test_video.mp4"

        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(video_path), fourcc, 30.0, (640, 480))

        # Write 10 test frames
        for i in range(10):
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            # Add frame number
            cv2.putText(
                frame,
                f"Frame {i}",
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
            )
            writer.write(frame)

        writer.release()

        return str(video_path)

    @pytest.mark.skip(reason="Requires full pipeline initialization")
    def test_video_processing_end_to_end(self, tmp_path):
        """Test complete video processing (requires real models)."""
        # This test would require actual model files and is more of an
        # end-to-end test. Skip for unit tests.
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
