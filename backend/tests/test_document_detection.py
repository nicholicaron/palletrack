"""Unit tests for document detection module."""

import numpy as np
import pytest

from app.cv_models import BoundingBox, DocumentDetection, PalletDetection, PalletTrack, TrackStatus
from app.cv_processing import DocumentAssociator, DocumentRegionExtractor


@pytest.fixture
def detection_config():
    """Fixture for detection configuration."""
    return {
        'detection': {
            'document_model_path': 'models/document_yolov8n.pt',
            'document_conf_threshold': 0.6,
            'device': 'cpu',
            'max_association_distance': 100,
            'containment_confidence': 0.9,
            'proximity_confidence': 0.7
        }
    }


@pytest.fixture
def sample_pallet_track():
    """Fixture for sample pallet track."""
    bbox = BoundingBox(x1=100, y1=200, x2=500, y2=700, confidence=0.9)
    detection = PalletDetection(bbox=bbox, frame_number=0, timestamp=0.0, track_id=1)

    track = PalletTrack(
        track_id=1,
        detections=[detection],
        first_seen_frame=0,
        last_seen_frame=0,
        status=TrackStatus.ACTIVE
    )
    return track


@pytest.fixture
def sample_document_inside_pallet():
    """Fixture for document detection inside pallet bbox."""
    # Document centered at (300, 450) which is inside pallet (100-500, 200-700)
    bbox = BoundingBox(x1=250, y1=400, x2=350, y2=500, confidence=0.85)
    return DocumentDetection(
        bbox=bbox,
        frame_number=0,
        parent_pallet_track_id=None
    )


@pytest.fixture
def sample_document_outside_pallet():
    """Fixture for document detection outside pallet bbox."""
    # Document centered at (700, 300) which is outside pallet (100-500, 200-700)
    bbox = BoundingBox(x1=650, y1=250, x2=750, y2=350, confidence=0.8)
    return DocumentDetection(
        bbox=bbox,
        frame_number=0,
        parent_pallet_track_id=None
    )


@pytest.fixture
def sample_frame():
    """Fixture for sample video frame."""
    return np.zeros((1080, 1920, 3), dtype=np.uint8)


class TestDocumentAssociator:
    """Tests for DocumentAssociator class."""

    def test_associate_document_inside_pallet(
        self, sample_document_inside_pallet, sample_pallet_track
    ):
        """Test associating document that's inside pallet bbox."""
        documents = [sample_document_inside_pallet]
        pallet_tracks = {1: sample_pallet_track}

        associated = DocumentAssociator.associate_documents_to_pallets(
            documents, pallet_tracks, max_distance=100
        )

        assert len(associated) == 1
        assert associated[0].parent_pallet_track_id == 1

    def test_associate_document_outside_pallet_nearby(
        self, sample_document_outside_pallet, sample_pallet_track
    ):
        """Test associating document outside but near pallet (fallback)."""
        documents = [sample_document_outside_pallet]
        pallet_tracks = {1: sample_pallet_track}

        associated = DocumentAssociator.associate_documents_to_pallets(
            documents, pallet_tracks, max_distance=300  # Increase threshold
        )

        assert len(associated) == 1
        # Should still associate due to proximity
        assert associated[0].parent_pallet_track_id == 1

    def test_associate_document_too_far(
        self, sample_document_outside_pallet, sample_pallet_track
    ):
        """Test document too far from any pallet."""
        documents = [sample_document_outside_pallet]
        pallet_tracks = {1: sample_pallet_track}

        associated = DocumentAssociator.associate_documents_to_pallets(
            documents, pallet_tracks, max_distance=50  # Very small threshold
        )

        assert len(associated) == 1
        # Should not associate (too far)
        assert associated[0].parent_pallet_track_id is None

    def test_associate_multiple_documents(
        self, sample_document_inside_pallet, sample_pallet_track
    ):
        """Test associating multiple documents to same pallet."""
        # Create second document also inside pallet
        bbox2 = BoundingBox(x1=200, y1=300, x2=300, y2=400, confidence=0.8)
        doc2 = DocumentDetection(bbox=bbox2, frame_number=0, parent_pallet_track_id=None)

        documents = [sample_document_inside_pallet, doc2]
        pallet_tracks = {1: sample_pallet_track}

        associated = DocumentAssociator.associate_documents_to_pallets(
            documents, pallet_tracks
        )

        assert len(associated) == 2
        assert all(doc.parent_pallet_track_id == 1 for doc in associated)

    def test_associate_to_closest_pallet(self):
        """Test document associates to closest pallet when multiple available."""
        # Create two pallets
        bbox1 = BoundingBox(x1=100, y1=200, x2=400, y2=600, confidence=0.9)
        det1 = PalletDetection(bbox=bbox1, frame_number=0, timestamp=0.0, track_id=1)
        track1 = PalletTrack(
            track_id=1, detections=[det1], first_seen_frame=0,
            last_seen_frame=0, status=TrackStatus.ACTIVE
        )

        bbox2 = BoundingBox(x1=600, y1=200, x2=900, y2=600, confidence=0.9)
        det2 = PalletDetection(bbox=bbox2, frame_number=0, timestamp=0.0, track_id=2)
        track2 = PalletTrack(
            track_id=2, detections=[det2], first_seen_frame=0,
            last_seen_frame=0, status=TrackStatus.ACTIVE
        )

        # Document closer to track2
        doc_bbox = BoundingBox(x1=700, y1=300, x2=800, y2=400, confidence=0.85)
        document = DocumentDetection(bbox=doc_bbox, frame_number=0)

        pallet_tracks = {1: track1, 2: track2}
        associated = DocumentAssociator.associate_documents_to_pallets(
            [document], pallet_tracks, max_distance=200
        )

        assert len(associated) == 1
        # Should associate to track 2 (closer)
        assert associated[0].parent_pallet_track_id == 2

    def test_validate_association_inside(self, sample_document_inside_pallet, sample_pallet_track):
        """Test validation confidence for document inside pallet."""
        confidence = DocumentAssociator.validate_association(
            sample_document_inside_pallet, sample_pallet_track
        )

        # Should have high confidence (inside pallet)
        assert confidence > 0.7

    def test_validate_association_outside(
        self, sample_document_outside_pallet, sample_pallet_track
    ):
        """Test validation confidence for document outside pallet."""
        confidence = DocumentAssociator.validate_association(
            sample_document_outside_pallet, sample_pallet_track
        )

        # Should have lower confidence (outside pallet)
        assert confidence < 0.9

    def test_validate_association_too_large(self, sample_pallet_track):
        """Test validation penalizes unreasonably large documents."""
        # Document that's 50% of pallet area (too large)
        pallet_bbox = sample_pallet_track.detections[0].bbox
        pallet_width = pallet_bbox.width()
        pallet_height = pallet_bbox.height()

        # Create document that's half the pallet size
        doc_bbox = BoundingBox(
            x1=200, y1=300,
            x2=200 + pallet_width * 0.7,
            y2=300 + pallet_height * 0.7,
            confidence=0.9
        )
        document = DocumentDetection(bbox=doc_bbox, frame_number=0)

        confidence = DocumentAssociator.validate_association(document, sample_pallet_track)

        # Should be penalized for being too large
        assert confidence < 0.9

    def test_empty_inputs(self):
        """Test handling of empty inputs."""
        # Empty documents
        result = DocumentAssociator.associate_documents_to_pallets([], {1: None})
        assert result == []

        # Empty pallets
        doc_bbox = BoundingBox(x1=100, y1=100, x2=200, y2=200, confidence=0.8)
        document = DocumentDetection(bbox=doc_bbox, frame_number=0)
        result = DocumentAssociator.associate_documents_to_pallets([document], {})
        assert len(result) == 1
        assert result[0].parent_pallet_track_id is None


class TestDocumentRegionExtractor:
    """Tests for DocumentRegionExtractor class."""

    def test_extract_document_region(self, sample_frame, sample_document_inside_pallet):
        """Test extracting single document region."""
        region = DocumentRegionExtractor.extract_document_region(
            sample_frame, sample_document_inside_pallet, padding=10
        )

        assert region is not None
        assert region.shape[0] > 0  # Height
        assert region.shape[1] > 0  # Width
        assert region.shape[2] == 3  # BGR channels

    def test_extract_with_padding(self, sample_frame):
        """Test padding is applied correctly."""
        bbox = BoundingBox(x1=100, y1=100, x2=200, y2=200, confidence=0.8)
        document = DocumentDetection(bbox=bbox, frame_number=0)

        # Extract with no padding
        region_no_pad = DocumentRegionExtractor.extract_document_region(
            sample_frame, document, padding=0
        )

        # Extract with padding
        region_with_pad = DocumentRegionExtractor.extract_document_region(
            sample_frame, document, padding=20
        )

        assert region_no_pad is not None
        assert region_with_pad is not None

        # With padding should be larger
        assert region_with_pad.shape[0] > region_no_pad.shape[0]
        assert region_with_pad.shape[1] > region_no_pad.shape[1]

    def test_extract_at_frame_edge(self, sample_frame):
        """Test extraction at frame edge (padding clipped)."""
        # Document at top-left corner
        bbox = BoundingBox(x1=5, y1=5, x2=50, y2=50, confidence=0.8)
        document = DocumentDetection(bbox=bbox, frame_number=0)

        region = DocumentRegionExtractor.extract_document_region(
            sample_frame, document, padding=10
        )

        # Should still extract (padding clipped to frame bounds)
        assert region is not None
        assert region.shape[0] > 0
        assert region.shape[1] > 0

    def test_extract_multiple_regions(
        self, sample_frame, sample_document_inside_pallet, sample_document_outside_pallet
    ):
        """Test extracting multiple document regions."""
        documents = [sample_document_inside_pallet, sample_document_outside_pallet]

        regions = DocumentRegionExtractor.extract_multiple_regions(
            sample_frame, documents, padding=10
        )

        # Should extract both documents
        assert len(regions) == 2
        assert 0 in regions
        assert 1 in regions

        # Each region should be valid
        for region in regions.values():
            assert region.shape[0] > 0
            assert region.shape[1] > 0
            assert region.shape[2] == 3

    def test_extract_with_metadata(self, sample_frame, sample_document_inside_pallet):
        """Test extracting with metadata."""
        result = DocumentRegionExtractor.extract_with_metadata(
            sample_frame, sample_document_inside_pallet, padding=10
        )

        assert result is not None
        assert 'region' in result
        assert 'bbox' in result
        assert 'padded_bbox' in result
        assert 'frame_number' in result
        assert 'confidence' in result

        # Verify region is valid
        assert result['region'].shape[0] > 0
        assert result['region'].shape[1] > 0

        # Verify metadata
        assert result['frame_number'] == 0
        assert result['confidence'] == sample_document_inside_pallet.bbox.confidence

        # Verify padded bbox is larger than original
        original_area = result['bbox'].area()
        padded_area = result['padded_bbox'].area()
        assert padded_area >= original_area

    def test_extract_invalid_bbox(self, sample_frame):
        """Test handling of invalid bounding boxes."""
        # Bbox outside frame bounds
        bbox = BoundingBox(x1=2000, y1=2000, x2=2100, y2=2100, confidence=0.8)
        document = DocumentDetection(bbox=bbox, frame_number=0)

        region = DocumentRegionExtractor.extract_document_region(
            sample_frame, document, padding=10
        )

        # Should return None for invalid bbox
        assert region is None

    def test_extract_zero_size_bbox(self, sample_frame):
        """Test handling of zero-size bbox (after clipping)."""
        # Create document at exact frame edge
        h, w = sample_frame.shape[:2]
        bbox = BoundingBox(x1=float(w-1), y1=float(h-1), x2=float(w), y2=float(h), confidence=0.8)
        document = DocumentDetection(bbox=bbox, frame_number=0)

        region = DocumentRegionExtractor.extract_document_region(
            sample_frame, document, padding=0
        )

        # Might be None or very small region
        # Just ensure it doesn't crash
        if region is not None:
            assert region.size >= 0

    def test_extract_empty_document_list(self, sample_frame):
        """Test extracting from empty document list."""
        regions = DocumentRegionExtractor.extract_multiple_regions(
            sample_frame, [], padding=10
        )

        assert regions == {}


class TestDocumentDetectorIntegration:
    """Integration tests for document detection (without actual YOLO model)."""

    def test_detection_config_validation(self, detection_config):
        """Test configuration structure is valid."""
        assert 'detection' in detection_config
        assert 'document_model_path' in detection_config['detection']
        assert 'document_conf_threshold' in detection_config['detection']
        assert 'device' in detection_config['detection']

    def test_association_workflow(
        self, sample_document_inside_pallet, sample_pallet_track, sample_frame
    ):
        """Test complete association and extraction workflow."""
        # Step 1: Associate document to pallet
        documents = [sample_document_inside_pallet]
        pallet_tracks = {1: sample_pallet_track}

        associated_docs = DocumentAssociator.associate_documents_to_pallets(
            documents, pallet_tracks
        )

        assert len(associated_docs) == 1
        assert associated_docs[0].parent_pallet_track_id == 1

        # Step 2: Extract document region
        region = DocumentRegionExtractor.extract_document_region(
            sample_frame, associated_docs[0], padding=10
        )

        assert region is not None
        assert region.shape[0] > 0

        # Step 3: Verify metadata extraction
        metadata = DocumentRegionExtractor.extract_with_metadata(
            sample_frame, associated_docs[0], padding=10
        )

        assert metadata is not None
        assert metadata['parent_pallet_track_id'] == 1

    def test_multiple_pallets_multiple_documents(self, sample_frame):
        """Test complex scenario with multiple pallets and documents."""
        # Create 2 pallets
        bbox1 = BoundingBox(x1=100, y1=200, x2=400, y2=600, confidence=0.9)
        det1 = PalletDetection(bbox=bbox1, frame_number=0, timestamp=0.0, track_id=1)
        track1 = PalletTrack(
            track_id=1, detections=[det1], first_seen_frame=0,
            last_seen_frame=0, status=TrackStatus.ACTIVE
        )

        bbox2 = BoundingBox(x1=600, y1=200, x2=900, y2=600, confidence=0.9)
        det2 = PalletDetection(bbox=bbox2, frame_number=0, timestamp=0.0, track_id=2)
        track2 = PalletTrack(
            track_id=2, detections=[det2], first_seen_frame=0,
            last_seen_frame=0, status=TrackStatus.ACTIVE
        )

        # Create 3 documents (2 on pallet 1, 1 on pallet 2)
        doc1_bbox = BoundingBox(x1=200, y1=300, x2=300, y2=400, confidence=0.85)
        doc1 = DocumentDetection(bbox=doc1_bbox, frame_number=0)

        doc2_bbox = BoundingBox(x1=250, y1=450, x2=350, y2=550, confidence=0.80)
        doc2 = DocumentDetection(bbox=doc2_bbox, frame_number=0)

        doc3_bbox = BoundingBox(x1=700, y1=300, x2=800, y2=400, confidence=0.88)
        doc3 = DocumentDetection(bbox=doc3_bbox, frame_number=0)

        documents = [doc1, doc2, doc3]
        pallet_tracks = {1: track1, 2: track2}

        # Associate all documents
        associated = DocumentAssociator.associate_documents_to_pallets(
            documents, pallet_tracks
        )

        assert len(associated) == 3

        # doc1 and doc2 should be on pallet 1
        doc1_assoc = [d for d in associated if d.bbox == doc1_bbox][0]
        doc2_assoc = [d for d in associated if d.bbox == doc2_bbox][0]
        assert doc1_assoc.parent_pallet_track_id == 1
        assert doc2_assoc.parent_pallet_track_id == 1

        # doc3 should be on pallet 2
        doc3_assoc = [d for d in associated if d.bbox == doc3_bbox][0]
        assert doc3_assoc.parent_pallet_track_id == 2

        # Extract all regions
        regions = DocumentRegionExtractor.extract_multiple_regions(
            sample_frame, associated
        )

        assert len(regions) == 3
