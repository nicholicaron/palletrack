"""Unit tests for PalleTrack CV data models."""

import pytest
from pydantic import ValidationError

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


class TestBoundingBox:
    """Tests for BoundingBox model."""

    def test_valid_bbox(self):
        """Test creating a valid bounding box."""
        bbox = BoundingBox(x1=10, y1=20, x2=100, y2=200, confidence=0.9)
        assert bbox.x1 == 10
        assert bbox.y1 == 20
        assert bbox.x2 == 100
        assert bbox.y2 == 200
        assert bbox.confidence == 0.9

    def test_invalid_x_coordinates(self):
        """Test that x2 must be greater than x1."""
        with pytest.raises(ValidationError):
            BoundingBox(x1=100, y1=20, x2=50, y2=200, confidence=0.9)

    def test_invalid_y_coordinates(self):
        """Test that y2 must be greater than y1."""
        with pytest.raises(ValidationError):
            BoundingBox(x1=10, y1=200, x2=100, y2=20, confidence=0.9)

    def test_invalid_confidence(self):
        """Test that confidence must be between 0 and 1."""
        with pytest.raises(ValidationError):
            BoundingBox(x1=10, y1=20, x2=100, y2=200, confidence=1.5)

        with pytest.raises(ValidationError):
            BoundingBox(x1=10, y1=20, x2=100, y2=200, confidence=-0.1)

    def test_area_calculation(self):
        """Test bounding box area calculation."""
        bbox = BoundingBox(x1=0, y1=0, x2=100, y2=50, confidence=0.9)
        assert bbox.area() == 5000

    def test_center_calculation(self):
        """Test bounding box center calculation."""
        bbox = BoundingBox(x1=0, y1=0, x2=100, y2=50, confidence=0.9)
        center_x, center_y = bbox.center()
        assert center_x == 50
        assert center_y == 25

    def test_contains_point(self):
        """Test point containment check."""
        bbox = BoundingBox(x1=10, y1=20, x2=100, y2=200, confidence=0.9)
        assert bbox.contains_point(50, 100) is True
        assert bbox.contains_point(5, 100) is False
        assert bbox.contains_point(50, 10) is False

    def test_iou_no_overlap(self):
        """Test IoU calculation with no overlap."""
        bbox1 = BoundingBox(x1=0, y1=0, x2=50, y2=50, confidence=0.9)
        bbox2 = BoundingBox(x1=100, y1=100, x2=150, y2=150, confidence=0.9)
        assert bbox1.iou(bbox2) == 0.0

    def test_iou_partial_overlap(self):
        """Test IoU calculation with partial overlap."""
        bbox1 = BoundingBox(x1=0, y1=0, x2=50, y2=50, confidence=0.9)
        bbox2 = BoundingBox(x1=25, y1=25, x2=75, y2=75, confidence=0.9)
        iou = bbox1.iou(bbox2)
        assert 0 < iou < 1

    def test_iou_complete_overlap(self):
        """Test IoU calculation with complete overlap."""
        bbox1 = BoundingBox(x1=0, y1=0, x2=50, y2=50, confidence=0.9)
        bbox2 = BoundingBox(x1=0, y1=0, x2=50, y2=50, confidence=0.9)
        assert bbox1.iou(bbox2) == 1.0

    def test_width_height(self):
        """Test width and height methods."""
        bbox = BoundingBox(x1=10, y1=20, x2=110, y2=70, confidence=0.9)
        assert bbox.width() == 100
        assert bbox.height() == 50


class TestPalletDetection:
    """Tests for PalletDetection model."""

    def test_valid_detection(self):
        """Test creating a valid pallet detection."""
        bbox = BoundingBox(x1=10, y1=20, x2=100, y2=200, confidence=0.9)
        detection = PalletDetection(
            bbox=bbox,
            frame_number=42,
            timestamp=1.4,
            track_id=7
        )
        assert detection.frame_number == 42
        assert detection.timestamp == 1.4
        assert detection.track_id == 7

    def test_detection_without_track_id(self):
        """Test creating detection without track ID."""
        bbox = BoundingBox(x1=10, y1=20, x2=100, y2=200, confidence=0.9)
        detection = PalletDetection(
            bbox=bbox,
            frame_number=42,
            timestamp=1.4
        )
        assert detection.track_id is None

    def test_negative_frame_number(self):
        """Test that frame number must be non-negative."""
        bbox = BoundingBox(x1=10, y1=20, x2=100, y2=200, confidence=0.9)
        with pytest.raises(ValidationError):
            PalletDetection(
                bbox=bbox,
                frame_number=-1,
                timestamp=1.4
            )


class TestDocumentDetection:
    """Tests for DocumentDetection model."""

    def test_valid_document_detection(self):
        """Test creating a valid document detection."""
        bbox = BoundingBox(x1=10, y1=20, x2=100, y2=200, confidence=0.88)
        doc = DocumentDetection(
            bbox=bbox,
            frame_number=45,
            parent_pallet_track_id=7,
            document_type=DocumentType.SHIPPING_LABEL
        )
        assert doc.document_type == DocumentType.SHIPPING_LABEL
        assert doc.parent_pallet_track_id == 7

    def test_unknown_document_type(self):
        """Test default document type is UNKNOWN."""
        bbox = BoundingBox(x1=10, y1=20, x2=100, y2=200, confidence=0.88)
        doc = DocumentDetection(
            bbox=bbox,
            frame_number=45
        )
        assert doc.document_type == DocumentType.UNKNOWN


class TestOCRResult:
    """Tests for OCRResult model."""

    def test_valid_ocr_result(self):
        """Test creating a valid OCR result."""
        bbox = BoundingBox(x1=10, y1=20, x2=100, y2=40, confidence=0.88)
        ocr = OCRResult(
            text="TRACKING: 1Z999AA10123456784",
            confidence=0.94,
            bbox=bbox,
            frame_number=45
        )
        assert ocr.text == "TRACKING: 1Z999AA10123456784"
        assert ocr.confidence == 0.94


class TestPalletTrack:
    """Tests for PalletTrack model."""

    def test_valid_track(self):
        """Test creating a valid pallet track."""
        track = PalletTrack(
            track_id=7,
            status=TrackStatus.ACTIVE,
            first_seen_frame=42,
            last_seen_frame=98
        )
        assert track.track_id == 7
        assert track.status == TrackStatus.ACTIVE
        assert len(track.detections) == 0
        assert len(track.document_regions) == 0
        assert len(track.ocr_results) == 0

    def test_invalid_frame_range(self):
        """Test that last_seen must be >= first_seen."""
        with pytest.raises(ValidationError):
            PalletTrack(
                track_id=7,
                status=TrackStatus.ACTIVE,
                first_seen_frame=100,
                last_seen_frame=50
            )

    def test_duration_frames(self):
        """Test frame duration calculation."""
        track = PalletTrack(
            track_id=7,
            status=TrackStatus.ACTIVE,
            first_seen_frame=42,
            last_seen_frame=98
        )
        assert track.duration_frames() == 57

    def test_get_best_ocr_text(self):
        """Test getting OCR text sorted by confidence."""
        bbox1 = BoundingBox(x1=10, y1=20, x2=100, y2=40, confidence=0.88)
        bbox2 = BoundingBox(x1=10, y1=50, x2=100, y2=70, confidence=0.92)

        track = PalletTrack(
            track_id=7,
            status=TrackStatus.ACTIVE,
            first_seen_frame=42,
            last_seen_frame=98,
            ocr_results=[
                OCRResult(text="Low confidence", confidence=0.6, bbox=bbox1, frame_number=45),
                OCRResult(text="High confidence", confidence=0.95, bbox=bbox2, frame_number=46),
            ]
        )

        ocr_text = track.get_best_ocr_text()
        lines = ocr_text.split("\n")
        assert lines[0] == "High confidence"
        assert lines[1] == "Low confidence"


class TestExtractedShippingData:
    """Tests for ExtractedShippingData model."""

    def test_valid_shipping_data(self):
        """Test creating valid shipping data."""
        data = ExtractedShippingData(
            track_id=7,
            document_type="SHIPPING_LABEL",
            tracking_number="1Z999AA10123456784",
            weight=45.5,
            destination_zip="62701",
            confidence_score=0.89,
            needs_review=False
        )
        assert data.tracking_number == "1Z999AA10123456784"
        assert data.weight == 45.5
        assert data.confidence_score == 0.89
        assert data.needs_review is False

    def test_minimal_shipping_data(self):
        """Test creating shipping data with minimal fields."""
        data = ExtractedShippingData(
            track_id=7,
            document_type="UNKNOWN",
            confidence_score=0.5,
        )
        assert data.tracking_number is None
        assert data.weight is None
        assert len(data.items) == 0

    def test_invalid_weight(self):
        """Test that weight must be positive."""
        with pytest.raises(ValidationError):
            ExtractedShippingData(
                track_id=7,
                document_type="SHIPPING_LABEL",
                weight=-10.0,
                confidence_score=0.89
            )

    def test_json_serialization(self):
        """Test that models can be serialized to JSON."""
        data = ExtractedShippingData(
            track_id=7,
            document_type="SHIPPING_LABEL",
            tracking_number="1Z999AA10123456784",
            confidence_score=0.89,
            items=[
                {"description": "Widget A", "quantity": 100}
            ]
        )
        json_data = data.model_dump_json()
        assert isinstance(json_data, str)
        assert "1Z999AA10123456784" in json_data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
