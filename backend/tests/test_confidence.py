"""Unit tests for confidence scoring and QA components."""

import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from app.cv_models import (
    BoundingBox,
    DocumentType,
    ExtractedShippingData,
    OCRResult,
    PalletDetection,
    PalletTrack,
    TrackStatus,
)
from app.cv_processing.qa import (
    ConfidenceCalculator,
    DataValidator,
    QualityMetricsTracker,
    ReviewQueueManager,
)


class TestDataValidator:
    """Test cases for DataValidator class."""

    def test_validate_tracking_number_ups(self):
        """Test UPS tracking number validation."""
        # Valid UPS tracking numbers
        assert DataValidator.validate_tracking_number("1Z999AA10123456784")
        assert DataValidator.validate_tracking_number("1z999aa10123456784")  # Case insensitive

        # Invalid UPS tracking numbers
        assert not DataValidator.validate_tracking_number("1Z999AA1012345678")  # Too short
        assert not DataValidator.validate_tracking_number("2Z999AA10123456784")  # Wrong prefix

    def test_validate_tracking_number_fedex(self):
        """Test FedEx tracking number validation."""
        # Valid FedEx tracking numbers
        assert DataValidator.validate_tracking_number("123456789012")  # 12 digits
        assert DataValidator.validate_tracking_number("123456789012345")  # 15 digits

        # Invalid FedEx tracking numbers
        assert not DataValidator.validate_tracking_number("12345678901")  # Too short
        assert not DataValidator.validate_tracking_number("1234567890123456")  # Too long

    def test_validate_tracking_number_usps(self):
        """Test USPS tracking number validation."""
        # Valid USPS tracking numbers
        assert DataValidator.validate_tracking_number("12345678901234567890")  # 20 digits
        assert DataValidator.validate_tracking_number("1234567890123456789012")  # 22 digits

        # Invalid USPS tracking numbers
        assert not DataValidator.validate_tracking_number("1234567890123456789")  # Too short

    def test_validate_tracking_number_invalid(self):
        """Test invalid tracking number validation."""
        assert not DataValidator.validate_tracking_number(None)
        assert not DataValidator.validate_tracking_number("")
        assert not DataValidator.validate_tracking_number("   ")
        assert not DataValidator.validate_tracking_number("INVALID")

    def test_validate_weight(self):
        """Test weight validation."""
        # Valid weights
        assert DataValidator.validate_weight(1.0)
        assert DataValidator.validate_weight(500.0)
        assert DataValidator.validate_weight(5000.0)

        # Invalid weights
        assert not DataValidator.validate_weight(0.5)  # Too light
        assert not DataValidator.validate_weight(6000.0)  # Too heavy
        assert not DataValidator.validate_weight(-10.0)  # Negative
        assert not DataValidator.validate_weight(None)

    def test_validate_zip_code(self):
        """Test ZIP code validation."""
        # Valid ZIP codes
        assert DataValidator.validate_zip_code("12345")
        assert DataValidator.validate_zip_code("12345-6789")

        # Invalid ZIP codes
        assert not DataValidator.validate_zip_code("1234")  # Too short
        assert not DataValidator.validate_zip_code("123456")  # Too long
        assert not DataValidator.validate_zip_code("12345-678")  # Invalid +4
        assert not DataValidator.validate_zip_code("ABCDE")
        assert not DataValidator.validate_zip_code(None)
        assert not DataValidator.validate_zip_code("")

    def test_validate_date(self):
        """Test date validation."""
        # Valid dates
        assert DataValidator.validate_date("01/15/2024")
        assert DataValidator.validate_date("12-25-2023")
        assert DataValidator.validate_date("2024-03-15")
        assert DataValidator.validate_date("January 15, 2024")

        # Invalid dates
        assert not DataValidator.validate_date("13/15/2024")  # Invalid month
        assert not DataValidator.validate_date("01/32/2024")  # Invalid day
        assert not DataValidator.validate_date("2024-13-01")  # Invalid month
        assert not DataValidator.validate_date("01/15/1900")  # Too far in past
        assert not DataValidator.validate_date("01/15/2050")  # Too far in future
        assert not DataValidator.validate_date("INVALID")
        assert not DataValidator.validate_date(None)

    def test_detect_garbage_text(self):
        """Test garbage text detection."""
        # Valid text
        assert not DataValidator.detect_garbage_text("ACME Corporation")
        assert not DataValidator.detect_garbage_text("123 Main Street")
        assert not DataValidator.detect_garbage_text("1Z999AA10123456784")

        # Garbage text
        assert DataValidator.detect_garbage_text("aaaaaaaaaa")  # Repetitive
        assert DataValidator.detect_garbage_text("!@#$%^&*()")  # Too many special chars
        assert DataValidator.detect_garbage_text("bcdfghjklm")  # No vowels
        assert DataValidator.detect_garbage_text("")  # Empty
        assert DataValidator.detect_garbage_text(None)  # None

    def test_validate_po_number(self):
        """Test PO number validation."""
        # Valid PO numbers
        assert DataValidator.validate_po_number("PO-12345")
        assert DataValidator.validate_po_number("ABC123")
        assert DataValidator.validate_po_number("2024/001")

        # Invalid PO numbers
        assert not DataValidator.validate_po_number("AB")  # Too short
        assert not DataValidator.validate_po_number("!@#$%")  # Too many special chars
        assert not DataValidator.validate_po_number(None)
        assert not DataValidator.validate_po_number("")


class TestConfidenceCalculator:
    """Test cases for ConfidenceCalculator class."""

    @pytest.fixture
    def config(self):
        """Test configuration."""
        return {
            'confidence': {
                'auto_accept': 0.85,
                'needs_review': 0.60,
                'auto_reject': 0.40,
                'weights': {
                    'detection_conf': 0.15,
                    'ocr_conf': 0.25,
                    'field_completeness': 0.25,
                    'cross_frame_consistency': 0.20,
                    'data_validation': 0.15
                }
            }
        }

    @pytest.fixture
    def calculator(self, config):
        """Create ConfidenceCalculator instance."""
        return ConfidenceCalculator(config)

    @pytest.fixture
    def sample_pallet_track(self):
        """Create sample pallet track."""
        detections = [
            PalletDetection(
                bbox=BoundingBox(x1=100, y1=200, x2=400, y2=600, confidence=0.9),
                frame_number=i,
                timestamp=i * 0.033,
                track_id=1
            )
            for i in range(10)
        ]

        return PalletTrack(
            track_id=1,
            detections=detections,
            status=TrackStatus.ACTIVE,
            first_seen_frame=0,
            last_seen_frame=9
        )

    @pytest.fixture
    def sample_ocr_results(self):
        """Create sample OCR results."""
        return [
            [
                OCRResult(
                    text="TRACKING: 1Z999AA10123456784",
                    confidence=0.95,
                    bbox=BoundingBox(x1=150, y1=250, x2=350, y2=280, confidence=0.9),
                    frame_number=i
                ),
                OCRResult(
                    text="WEIGHT: 500 LBS",
                    confidence=0.92,
                    bbox=BoundingBox(x1=150, y1=300, x2=350, y2=330, confidence=0.9),
                    frame_number=i
                )
            ]
            for i in range(5)
        ]

    @pytest.fixture
    def sample_extracted_data(self):
        """Create sample extracted data."""
        return ExtractedShippingData(
            track_id=1,
            document_type=DocumentType.SHIPPING_LABEL.value,
            tracking_number="1Z999AA10123456784",
            weight=500.0,
            destination_address="123 Main St, Springfield, IL",
            destination_zip="62701",
            confidence_score=0.85,
            needs_review=False
        )

    def test_detection_confidence(self, calculator, sample_pallet_track):
        """Test detection confidence calculation."""
        confidence = calculator._detection_confidence(sample_pallet_track)
        assert 0.0 <= confidence <= 1.0
        assert confidence == 0.9  # All detections have 0.9 confidence

    def test_detection_confidence_empty(self, calculator):
        """Test detection confidence with empty detections."""
        track = PalletTrack(
            track_id=1,
            detections=[],
            status=TrackStatus.ACTIVE,
            first_seen_frame=0,
            last_seen_frame=0
        )
        confidence = calculator._detection_confidence(track)
        assert confidence == 0.0

    def test_ocr_confidence(self, calculator, sample_ocr_results):
        """Test OCR confidence calculation."""
        confidence = calculator._ocr_confidence(sample_ocr_results)
        assert 0.0 <= confidence <= 1.0
        assert confidence > 0.9  # All OCR results have high confidence

    def test_ocr_confidence_empty(self, calculator):
        """Test OCR confidence with empty results."""
        confidence = calculator._ocr_confidence([])
        assert confidence == 0.0

    def test_field_completeness_complete(self, calculator):
        """Test field completeness with all fields present."""
        data = ExtractedShippingData(
            track_id=1,
            document_type=DocumentType.SHIPPING_LABEL.value,
            tracking_number="1Z999AA10123456784",
            weight=500.0,
            confidence_score=0.85
        )
        completeness = calculator._field_completeness(data)
        assert completeness == 1.0

    def test_field_completeness_partial(self, calculator):
        """Test field completeness with missing fields."""
        data = ExtractedShippingData(
            track_id=1,
            document_type=DocumentType.SHIPPING_LABEL.value,
            tracking_number="1Z999AA10123456784",
            weight=None,  # Missing weight
            confidence_score=0.85
        )
        completeness = calculator._field_completeness(data)
        assert completeness == 0.5  # 1 out of 2 required fields

    def test_field_completeness_packing_list(self, calculator):
        """Test field completeness for packing list."""
        data = ExtractedShippingData(
            track_id=1,
            document_type=DocumentType.PACKING_LIST.value,
            items=[{"description": "Widget A", "quantity": 100}],
            confidence_score=0.85
        )
        completeness = calculator._field_completeness(data)
        assert completeness == 1.0

    def test_data_validation(self, calculator, sample_extracted_data):
        """Test data validation scoring."""
        validation_score = calculator._data_validation(sample_extracted_data)
        assert 0.0 <= validation_score <= 1.0
        # Should be high since all fields are valid
        assert validation_score > 0.7

    def test_data_validation_invalid(self, calculator):
        """Test data validation with invalid data."""
        data = ExtractedShippingData(
            track_id=1,
            document_type=DocumentType.SHIPPING_LABEL.value,
            tracking_number="INVALID",
            weight=10000.0,  # Too heavy
            destination_zip="ABC",  # Invalid ZIP
            confidence_score=0.5
        )
        validation_score = calculator._data_validation(data)
        assert validation_score < 0.5  # Should be low due to invalid fields

    def test_calculate_confidence(
        self,
        calculator,
        sample_pallet_track,
        sample_extracted_data,
        sample_ocr_results
    ):
        """Test overall confidence calculation."""
        overall_conf, breakdown = calculator.calculate_confidence(
            sample_pallet_track,
            sample_extracted_data,
            sample_ocr_results
        )

        # Check overall confidence is in valid range
        assert 0.0 <= overall_conf <= 1.0

        # Check breakdown contains all expected factors
        assert 'detection_confidence' in breakdown
        assert 'ocr_confidence' in breakdown
        assert 'field_completeness' in breakdown
        assert 'cross_frame_consistency' in breakdown
        assert 'data_validation' in breakdown
        assert 'overall_confidence' in breakdown

        # All individual scores should be valid
        for key, value in breakdown.items():
            if key != 'weights' and key != 'overall_confidence':
                assert 0.0 <= value <= 1.0


class TestReviewQueueManager:
    """Test cases for ReviewQueueManager class."""

    @pytest.fixture
    def config(self):
        """Test configuration."""
        return {
            'confidence': {
                'auto_accept': 0.85,
                'needs_review': 0.60,
                'auto_reject': 0.40
            },
            'review_queue': {
                'priority_order': 'confidence',
                'max_queue_size': 100,
                'save_frames': False,  # Disable for tests
                'frame_save_path': 'test_review_frames'
            }
        }

    @pytest.fixture
    def manager(self, config):
        """Create ReviewQueueManager instance."""
        return ReviewQueueManager(config)

    @pytest.fixture
    def sample_extracted_data(self):
        """Create sample extracted data."""
        return ExtractedShippingData(
            track_id=1,
            document_type=DocumentType.SHIPPING_LABEL.value,
            tracking_number="1Z999AA10123456784",
            weight=500.0,
            confidence_score=0.75,
            needs_review=True
        )

    @pytest.fixture
    def sample_confidence_breakdown(self):
        """Create sample confidence breakdown."""
        return {
            'detection_confidence': 0.8,
            'ocr_confidence': 0.9,
            'field_completeness': 0.7,
            'cross_frame_consistency': 0.6,
            'data_validation': 0.8,
            'overall_confidence': 0.75
        }

    def test_route_auto_accept(self, manager, sample_extracted_data, sample_confidence_breakdown):
        """Test routing to auto-accept."""
        high_conf_breakdown = sample_confidence_breakdown.copy()
        high_conf_breakdown['overall_confidence'] = 0.9

        route = manager.route_extraction(sample_extracted_data, high_conf_breakdown)
        assert route == 'AUTO_ACCEPT'
        assert len(manager.auto_accepted) == 1

    def test_route_needs_review(self, manager, sample_extracted_data, sample_confidence_breakdown):
        """Test routing to needs review."""
        route = manager.route_extraction(sample_extracted_data, sample_confidence_breakdown)
        assert route == 'NEEDS_REVIEW'

    def test_route_auto_reject(self, manager, sample_extracted_data, sample_confidence_breakdown):
        """Test routing to auto-reject."""
        low_conf_breakdown = sample_confidence_breakdown.copy()
        low_conf_breakdown['overall_confidence'] = 0.3

        route = manager.route_extraction(sample_extracted_data, low_conf_breakdown)
        assert route == 'AUTO_REJECT'
        assert len(manager.auto_rejected) == 1

    def test_add_to_review_queue(self, manager, sample_extracted_data, sample_confidence_breakdown):
        """Test adding item to review queue."""
        manager.add_to_review_queue(
            sample_extracted_data,
            sample_confidence_breakdown
        )

        assert len(manager.review_queue) == 1
        item = manager.review_queue[0]
        assert item['track_id'] == 1
        assert 'operator_context' in item
        assert 'timestamp' in item

    def test_review_queue_max_size(self, manager):
        """Test review queue respects max size."""
        manager.max_queue_size = 5

        # Add more than max_queue_size items
        for i in range(10):
            data = ExtractedShippingData(
                track_id=i,
                document_type=DocumentType.SHIPPING_LABEL.value,
                confidence_score=0.7,
                needs_review=True
            )
            manager.add_to_review_queue(data, {'overall_confidence': 0.7})

        assert len(manager.review_queue) <= manager.max_queue_size

    def test_get_review_queue_by_confidence(self, manager):
        """Test getting review queue sorted by confidence."""
        # Add items with different confidences
        for i, conf in enumerate([0.8, 0.6, 0.75, 0.65]):
            data = ExtractedShippingData(
                track_id=i,
                document_type=DocumentType.SHIPPING_LABEL.value,
                confidence_score=conf,
                needs_review=True
            )
            manager.add_to_review_queue(data, {'overall_confidence': conf})

        queue = manager.get_review_queue(priority_order='confidence')

        # Should be sorted by confidence (lowest first)
        confidences = [item['confidence_breakdown']['overall_confidence'] for item in queue]
        assert confidences == sorted(confidences)

    def test_generate_operator_context(self, manager, sample_extracted_data, sample_confidence_breakdown):
        """Test operator context generation."""
        context = manager.generate_operator_context(
            sample_extracted_data,
            sample_confidence_breakdown
        )

        assert 'warnings' in context
        assert 'suggestions' in context
        assert 'confidence_details' in context
        assert isinstance(context['warnings'], list)
        assert isinstance(context['suggestions'], list)

    def test_get_queue_stats(self, manager, sample_extracted_data, sample_confidence_breakdown):
        """Test queue statistics."""
        manager.add_to_review_queue(sample_extracted_data, sample_confidence_breakdown)

        stats = manager.get_queue_stats()
        assert 'review_queue_size' in stats
        assert 'auto_accepted_count' in stats
        assert 'auto_rejected_count' in stats
        assert stats['review_queue_size'] == 1

    def test_remove_from_queue(self, manager, sample_extracted_data, sample_confidence_breakdown):
        """Test removing item from queue."""
        manager.add_to_review_queue(sample_extracted_data, sample_confidence_breakdown)
        assert len(manager.review_queue) == 1

        removed = manager.remove_from_queue(track_id=1)
        assert removed is not None
        assert len(manager.review_queue) == 0


class TestQualityMetricsTracker:
    """Test cases for QualityMetricsTracker class."""

    @pytest.fixture
    def tracker(self):
        """Create QualityMetricsTracker instance."""
        return QualityMetricsTracker()

    def test_record_extraction(self, tracker):
        """Test recording extraction."""
        tracker.record_extraction(
            route='AUTO_ACCEPT',
            confidence=0.9,
            processing_time=1.5
        )

        assert tracker.metrics['total_processed'] == 1
        assert tracker.metrics['auto_accepted'] == 1
        assert len(tracker.metrics['avg_confidence']) == 1
        assert tracker.metrics['avg_confidence'][0] == 0.9

    def test_record_multiple_extractions(self, tracker):
        """Test recording multiple extractions."""
        tracker.record_extraction('AUTO_ACCEPT', 0.9, 1.0)
        tracker.record_extraction('NEEDS_REVIEW', 0.7, 1.2)
        tracker.record_extraction('AUTO_REJECT', 0.4, 0.8)

        assert tracker.metrics['total_processed'] == 3
        assert tracker.metrics['auto_accepted'] == 1
        assert tracker.metrics['needs_review'] == 1
        assert tracker.metrics['auto_rejected'] == 1

    def test_record_operator_correction(self, tracker):
        """Test recording operator correction."""
        tracker.record_operator_correction(
            track_id=1,
            field='tracking_number',
            original='1Z999AA1O123456784',  # O instead of 0
            corrected='1Z999AA10123456784',
            confidence=0.7
        )

        assert len(tracker.metrics['operator_corrections']) == 1
        correction = tracker.metrics['operator_corrections'][0]
        assert correction['field'] == 'tracking_number'
        assert correction['original'] == '1Z999AA1O123456784'
        assert correction['corrected'] == '1Z999AA10123456784'

    def test_classify_error_ocr_confusion(self, tracker):
        """Test error classification for OCR confusion."""
        error_type = tracker._classify_error('1Z999AA1O123456784', '1Z999AA10123456784')
        assert error_type == 'ocr_character_confusion'

    def test_classify_error_missing_field(self, tracker):
        """Test error classification for missing field."""
        error_type = tracker._classify_error('', 'SOME_VALUE')
        assert error_type == 'missing_field'

    def test_generate_quality_report_empty(self, tracker):
        """Test quality report with no data."""
        report = tracker.generate_quality_report()
        assert report['summary']['total_processed'] == 0
        assert report['summary']['auto_acceptance_rate'] == 0.0

    def test_generate_quality_report_with_data(self, tracker):
        """Test quality report with data."""
        # Record some extractions
        tracker.record_extraction('AUTO_ACCEPT', 0.9, 1.0)
        tracker.record_extraction('AUTO_ACCEPT', 0.88, 1.1)
        tracker.record_extraction('NEEDS_REVIEW', 0.7, 1.3)
        tracker.record_extraction('AUTO_REJECT', 0.4, 0.9)

        report = tracker.generate_quality_report()

        assert report['summary']['total_processed'] == 4
        assert report['summary']['auto_acceptance_rate'] == 0.5  # 2 out of 4
        assert report['summary']['review_rate'] == 0.25  # 1 out of 4
        assert report['summary']['rejection_rate'] == 0.25  # 1 out of 4
        assert report['performance']['avg_confidence'] > 0

    def test_get_error_patterns(self, tracker):
        """Test error pattern analysis."""
        # Record some corrections with OCR confusions
        tracker.record_operator_correction(1, 'tracking_number', '1Z999AA1O123', '1Z999AA10123')
        tracker.record_operator_correction(2, 'tracking_number', '1Z999AA1O456', '1Z999AA10456')

        patterns = tracker.get_error_patterns()
        assert 'top_character_confusions' in patterns
        assert 'field_specific_errors' in patterns

    def test_reset_metrics(self, tracker):
        """Test resetting metrics."""
        tracker.record_extraction('AUTO_ACCEPT', 0.9, 1.0)
        tracker.reset_metrics()

        assert tracker.metrics['total_processed'] == 0
        assert len(tracker.metrics['avg_confidence']) == 0

    def test_export_metrics(self, tracker):
        """Test exporting metrics."""
        tracker.record_extraction('AUTO_ACCEPT', 0.9, 1.0)
        export = tracker.export_metrics()

        assert 'metrics' in export
        assert 'quality_report' in export
        assert 'extraction_history' in export
