"""Utilities for document detection processing.

This module provides utilities for associating documents with pallets and
extracting document regions from frames.
"""

import math
from typing import Dict, List, Optional

import numpy as np

from app.cv_models import BoundingBox, DocumentDetection, PalletTrack


class DocumentAssociator:
    """Associate detected documents with tracked pallets.

    Uses geometric analysis to determine which pallet each document belongs to.
    Primary strategy: check if document center is inside pallet bbox.
    Fallback: find closest pallet within proximity threshold.

    Example:
        >>> documents = detector.detect(frame, frame_number, pallet_tracks)
        >>> associated_docs = DocumentAssociator.associate_documents_to_pallets(
        ...     documents, pallet_tracks, max_distance=100
        ... )
    """

    @staticmethod
    def associate_documents_to_pallets(
        documents: List[DocumentDetection],
        pallet_tracks: Dict[int, PalletTrack],
        max_distance: float = 100.0
    ) -> List[DocumentDetection]:
        """Determine which pallet each document belongs to.

        Strategy:
        1. Check if document center is inside pallet bbox (primary)
        2. If not, find closest pallet within proximity threshold (fallback)
        3. Assign parent_pallet_track_id
        4. Adjust confidence if fallback method used

        Args:
            documents: List of unassociated DocumentDetection objects
            pallet_tracks: Dictionary of active pallet tracks (track_id -> PalletTrack)
            max_distance: Maximum distance (pixels) for fallback association

        Returns:
            List of DocumentDetection objects with parent_pallet_track_id assigned

        Example:
            >>> associated = DocumentAssociator.associate_documents_to_pallets(
            ...     documents, pallet_tracks, max_distance=100
            ... )
        """
        if not documents or not pallet_tracks:
            return documents

        associated_documents = []

        for doc in documents:
            doc_center_x, doc_center_y = doc.bbox.center()

            # Try primary strategy: containment check
            best_track_id = None
            best_confidence = 0.0

            for track_id, track in pallet_tracks.items():
                if not track.detections:
                    continue

                # Get latest pallet detection
                latest_detection = track.detections[-1]
                pallet_bbox = latest_detection.bbox

                # Check if document center is inside pallet bbox
                if pallet_bbox.contains_point(doc_center_x, doc_center_y):
                    confidence = DocumentAssociator.validate_association(doc, track)
                    if confidence > best_confidence:
                        best_confidence = confidence
                        best_track_id = track_id

            # If no containment match, try fallback: find closest pallet
            if best_track_id is None:
                best_distance = float('inf')

                for track_id, track in pallet_tracks.items():
                    if not track.detections:
                        continue

                    latest_detection = track.detections[-1]
                    pallet_bbox = latest_detection.bbox
                    pallet_center_x, pallet_center_y = pallet_bbox.center()

                    # Calculate Euclidean distance
                    distance = math.sqrt(
                        (doc_center_x - pallet_center_x) ** 2 +
                        (doc_center_y - pallet_center_y) ** 2
                    )

                    if distance < best_distance and distance <= max_distance:
                        best_distance = distance
                        best_track_id = track_id

                # Lower confidence for proximity-based association
                if best_track_id is not None:
                    best_confidence = DocumentAssociator.validate_association(
                        doc, pallet_tracks[best_track_id]
                    )
                    # Reduce confidence due to fallback method
                    best_confidence *= 0.8

            # Create new DocumentDetection with assigned pallet
            associated_doc = DocumentDetection(
                bbox=doc.bbox,
                frame_number=doc.frame_number,
                parent_pallet_track_id=best_track_id,
                document_type=doc.document_type
            )

            # Optionally adjust document confidence
            if best_track_id is not None and best_confidence < 1.0:
                # Reduce document detection confidence by association confidence
                adjusted_confidence = doc.bbox.confidence * best_confidence
                associated_doc.bbox = BoundingBox(
                    x1=doc.bbox.x1,
                    y1=doc.bbox.y1,
                    x2=doc.bbox.x2,
                    y2=doc.bbox.y2,
                    confidence=adjusted_confidence
                )

            associated_documents.append(associated_doc)

        return associated_documents

    @staticmethod
    def validate_association(
        doc: DocumentDetection,
        pallet: PalletTrack
    ) -> float:
        """Calculate confidence that document belongs to pallet.

        Considers:
        - Geometric containment (is document inside pallet bbox?)
        - Distance (how close are centers?)
        - Relative size (is document reasonably sized relative to pallet?)

        Args:
            doc: DocumentDetection object
            pallet: PalletTrack object

        Returns:
            Confidence score 0-1 (1.0 = high confidence)

        Example:
            >>> confidence = DocumentAssociator.validate_association(doc, pallet_track)
        """
        if not pallet.detections:
            return 0.0

        # Get latest pallet detection
        latest_detection = pallet.detections[-1]
        pallet_bbox = latest_detection.bbox

        confidence_score = 1.0

        # Factor 1: Containment (strongest signal)
        doc_center_x, doc_center_y = doc.bbox.center()
        if not pallet_bbox.contains_point(doc_center_x, doc_center_y):
            confidence_score *= 0.7  # Reduce confidence if not contained

        # Factor 2: Relative distance
        pallet_center_x, pallet_center_y = pallet_bbox.center()
        distance = math.sqrt(
            (doc_center_x - pallet_center_x) ** 2 +
            (doc_center_y - pallet_center_y) ** 2
        )

        # Normalize distance by pallet size
        pallet_size = math.sqrt(pallet_bbox.area())
        if pallet_size > 0:
            normalized_distance = distance / pallet_size
            # Penalize if document is far from pallet center
            if normalized_distance > 0.5:
                confidence_score *= (1.0 - min(normalized_distance - 0.5, 0.5))

        # Factor 3: Relative size (documents should be much smaller than pallets)
        doc_area = doc.bbox.area()
        pallet_area = pallet_bbox.area()
        if pallet_area > 0:
            size_ratio = doc_area / pallet_area
            # Typical documents are 5-20% of pallet area
            if size_ratio > 0.3:  # Too large to be a document
                confidence_score *= 0.6
            elif size_ratio < 0.01:  # Too small to be readable
                confidence_score *= 0.8

        return min(max(confidence_score, 0.0), 1.0)


class DocumentRegionExtractor:
    """Extract document regions from frames for OCR processing.

    Provides methods to crop document regions with optional padding for
    better OCR performance.

    Example:
        >>> extractor = DocumentRegionExtractor()
        >>> region = extractor.extract_document_region(frame, document, padding=10)
    """

    @staticmethod
    def extract_document_region(
        frame: np.ndarray,
        document: DocumentDetection,
        padding: int = 10
    ) -> Optional[np.ndarray]:
        """Extract document region from frame with optional padding.

        Args:
            frame: Input frame (BGR format)
            document: DocumentDetection object
            padding: Pixels of padding to add around bbox (default: 10)

        Returns:
            Cropped image region, or None if extraction fails

        Example:
            >>> region = DocumentRegionExtractor.extract_document_region(
            ...     frame, document, padding=10
            ... )
            >>> if region is not None:
            ...     # Perform OCR on region
        """
        bbox = document.bbox
        h, w = frame.shape[:2]

        # Apply padding
        x1 = max(0, int(bbox.x1 - padding))
        y1 = max(0, int(bbox.y1 - padding))
        x2 = min(w, int(bbox.x2 + padding))
        y2 = min(h, int(bbox.y2 + padding))

        # Validate coordinates
        if x2 <= x1 or y2 <= y1:
            return None

        # Extract region
        try:
            region = frame[y1:y2, x1:x2]
            if region.size == 0:
                return None
            return region
        except Exception:
            return None

    @staticmethod
    def extract_multiple_regions(
        frame: np.ndarray,
        documents: List[DocumentDetection],
        padding: int = 10
    ) -> Dict[int, np.ndarray]:
        """Extract all document regions from frame.

        Args:
            frame: Input frame (BGR format)
            documents: List of DocumentDetection objects
            padding: Pixels of padding to add around bboxes (default: 10)

        Returns:
            Dictionary mapping document index to cropped image region
            (failed extractions are omitted)

        Example:
            >>> regions = DocumentRegionExtractor.extract_multiple_regions(
            ...     frame, documents, padding=10
            ... )
            >>> for idx, region in regions.items():
            ...     print(f"Document {idx}: {region.shape}")
        """
        regions = {}

        for idx, document in enumerate(documents):
            region = DocumentRegionExtractor.extract_document_region(
                frame, document, padding
            )
            if region is not None:
                regions[idx] = region

        return regions

    @staticmethod
    def extract_with_metadata(
        frame: np.ndarray,
        document: DocumentDetection,
        padding: int = 10
    ) -> Optional[Dict]:
        """Extract document region with metadata.

        Returns both the cropped region and useful metadata for OCR processing.

        Args:
            frame: Input frame (BGR format)
            document: DocumentDetection object
            padding: Pixels of padding to add around bbox (default: 10)

        Returns:
            Dictionary with:
                - 'region': Cropped image region
                - 'bbox': Original bounding box
                - 'padded_bbox': Bounding box with padding applied
                - 'frame_number': Frame number
                - 'parent_pallet_track_id': Parent pallet track ID
                - 'confidence': Detection confidence
            Returns None if extraction fails

        Example:
            >>> result = DocumentRegionExtractor.extract_with_metadata(
            ...     frame, document, padding=10
            ... )
            >>> if result:
            ...     print(f"Region shape: {result['region'].shape}")
            ...     print(f"Confidence: {result['confidence']:.2f}")
        """
        region = DocumentRegionExtractor.extract_document_region(
            frame, document, padding
        )

        if region is None:
            return None

        bbox = document.bbox
        h, w = frame.shape[:2]

        # Calculate padded bbox
        padded_x1 = max(0, int(bbox.x1 - padding))
        padded_y1 = max(0, int(bbox.y1 - padding))
        padded_x2 = min(w, int(bbox.x2 + padding))
        padded_y2 = min(h, int(bbox.y2 + padding))

        return {
            'region': region,
            'bbox': bbox,
            'padded_bbox': BoundingBox(
                x1=float(padded_x1),
                y1=float(padded_y1),
                x2=float(padded_x2),
                y2=float(padded_y2),
                confidence=bbox.confidence
            ),
            'frame_number': document.frame_number,
            'parent_pallet_track_id': document.parent_pallet_track_id,
            'confidence': bbox.confidence
        }
