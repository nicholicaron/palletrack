"""Calculate overall confidence scores for extracted shipping data."""

import re
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np

from ...cv_models import DocumentType, ExtractedShippingData, OCRResult, PalletTrack
from .validators import DataValidator


class ConfidenceCalculator:
    """Calculate overall confidence from multiple factors."""

    # Confidence thresholds from config
    THRESHOLDS = {
        'AUTO_ACCEPT': 0.85,
        'NEEDS_REVIEW': 0.60,
        'AUTO_REJECT': 0.40
    }

    def __init__(self, config: Dict):
        """Initialize confidence calculator.

        Args:
            config: Configuration dictionary containing confidence settings
        """
        self.config = config

        # Load weights from config or use defaults
        confidence_config = config.get('confidence', {})
        weights = confidence_config.get('weights', {})

        self.weights = {
            'detection_conf': weights.get('detection_conf', 0.15),
            'ocr_conf': weights.get('ocr_conf', 0.25),
            'field_completeness': weights.get('field_completeness', 0.25),
            'cross_frame_consistency': weights.get('cross_frame_consistency', 0.20),
            'data_validation': weights.get('data_validation', 0.15)
        }

        # Update thresholds from config
        self.THRESHOLDS['AUTO_ACCEPT'] = confidence_config.get('auto_accept', 0.85)
        self.THRESHOLDS['NEEDS_REVIEW'] = confidence_config.get('needs_review', 0.60)
        self.THRESHOLDS['AUTO_REJECT'] = confidence_config.get('auto_reject', 0.40)

    def calculate_confidence(
        self,
        pallet_track: PalletTrack,
        extracted_data: ExtractedShippingData,
        ocr_results: List[List[OCRResult]]
    ) -> Tuple[float, Dict]:
        """Calculate overall confidence score.

        Args:
            pallet_track: Complete pallet tracking information
            extracted_data: Structured data extracted from OCR
            ocr_results: List of OCR results per frame (list of lists)

        Returns:
            Tuple of (overall_confidence, factor_breakdown)
            - overall_confidence: Weighted score between 0-1
            - factor_breakdown: Dict of individual scores for debugging
        """
        # Calculate individual confidence factors
        detection_score = self._detection_confidence(pallet_track)
        ocr_score = self._ocr_confidence(ocr_results)
        completeness_score = self._field_completeness(extracted_data)
        consistency_score = self._cross_frame_consistency(ocr_results)
        validation_score = self._data_validation(extracted_data)

        # Calculate weighted overall confidence
        overall_confidence = (
            self.weights['detection_conf'] * detection_score +
            self.weights['ocr_conf'] * ocr_score +
            self.weights['field_completeness'] * completeness_score +
            self.weights['cross_frame_consistency'] * consistency_score +
            self.weights['data_validation'] * validation_score
        )

        # Factor breakdown for debugging
        factor_breakdown = {
            'detection_confidence': detection_score,
            'ocr_confidence': ocr_score,
            'field_completeness': completeness_score,
            'cross_frame_consistency': consistency_score,
            'data_validation': validation_score,
            'weights': self.weights,
            'overall_confidence': overall_confidence
        }

        return overall_confidence, factor_breakdown

    def _detection_confidence(self, pallet_track: PalletTrack) -> float:
        """Average detection confidence across track.

        How confident were we in pallet/document detections?

        Args:
            pallet_track: Pallet tracking information

        Returns:
            Average confidence score (0-1)
        """
        if not pallet_track.detections:
            return 0.0

        # Get all detection confidences
        detection_confidences = [det.bbox.confidence for det in pallet_track.detections]

        # Calculate average
        avg_confidence = np.mean(detection_confidences)

        return float(avg_confidence)

    def _ocr_confidence(self, ocr_results: List[List[OCRResult]]) -> float:
        """Average OCR confidence across all frames.

        How confident was PaddleOCR in text recognition?

        Args:
            ocr_results: List of OCR results per frame

        Returns:
            Average OCR confidence (0-1)
        """
        if not ocr_results:
            return 0.0

        # Flatten list of lists to get all OCR results
        all_ocr = [ocr for frame_ocr in ocr_results for ocr in frame_ocr]

        if not all_ocr:
            return 0.0

        # Calculate average OCR confidence
        ocr_confidences = [ocr.confidence for ocr in all_ocr]
        avg_confidence = np.mean(ocr_confidences)

        return float(avg_confidence)

    def _field_completeness(self, extracted_data: ExtractedShippingData) -> float:
        """Percentage of required fields successfully extracted.

        Required fields vary by document type:
        - BOL: tracking_number, weight, destination
        - PACKING_LIST: items, po_number
        - SHIPPING_LABEL: tracking_number, weight

        Args:
            extracted_data: Extracted shipping data

        Returns:
            Completeness score (0-1)
        """
        # Define required fields by document type
        required_fields_map = {
            DocumentType.BOL.value: ['tracking_number', 'weight', 'destination_address'],
            DocumentType.PACKING_LIST.value: ['items'],
            DocumentType.SHIPPING_LABEL.value: ['tracking_number', 'weight'],
            DocumentType.UNKNOWN.value: ['tracking_number']  # Minimal requirement
        }

        # Get required fields for this document type
        doc_type = extracted_data.document_type
        required_fields = required_fields_map.get(doc_type, ['tracking_number'])

        if not required_fields:
            return 1.0

        # Count how many required fields are present and non-empty
        fields_present = 0
        for field in required_fields:
            value = getattr(extracted_data, field, None)

            # Check if field is present and non-empty
            if field == 'items':
                # Special case: items is a list
                if value and len(value) > 0:
                    fields_present += 1
            else:
                # Regular field
                if value is not None and value != '':
                    fields_present += 1

        # Calculate completeness ratio
        completeness = fields_present / len(required_fields)

        return completeness

    def _cross_frame_consistency(self, ocr_results: List[List[OCRResult]]) -> float:
        """Agreement across multiple frame readings.

        Process:
        1. Extract key fields from each frame's OCR
        2. Calculate agreement percentage
        3. Higher agreement → higher confidence

        Example:
        Frame 1: "Tracking: 1Z9999999"
        Frame 2: "Tracking: 1Z9999999"
        Frame 3: "Tracking: 1Z9999998"
        Agreement: 66% → confidence 0.66

        Args:
            ocr_results: List of OCR results per frame

        Returns:
            Consistency score (0-1)
        """
        if len(ocr_results) < 2:
            # Not enough frames to compare
            return 0.5  # Neutral score

        # Extract text patterns that look like tracking numbers, weights, etc.
        tracking_patterns = []
        weight_patterns = []
        zip_patterns = []

        for frame_ocr in ocr_results:
            frame_text = ' '.join([ocr.text for ocr in frame_ocr])

            # Find tracking numbers (UPS, FedEx, USPS patterns)
            tracking_matches = re.findall(
                r'\b1Z[A-Z0-9]{16}\b|\b\d{12,15}\b|\b\d{20,22}\b',
                frame_text,
                re.IGNORECASE
            )
            tracking_patterns.extend(tracking_matches)

            # Find weights (number followed by lbs/lb/pounds)
            weight_matches = re.findall(
                r'\b(\d+\.?\d*)\s*(?:lbs?|pounds)\b',
                frame_text,
                re.IGNORECASE
            )
            weight_patterns.extend(weight_matches)

            # Find ZIP codes
            zip_matches = re.findall(r'\b\d{5}(?:-\d{4})?\b', frame_text)
            zip_patterns.extend(zip_matches)

        # Calculate consistency for each field type
        consistencies = []

        if tracking_patterns:
            most_common_tracking = Counter(tracking_patterns).most_common(1)[0]
            tracking_consistency = most_common_tracking[1] / len(tracking_patterns)
            consistencies.append(tracking_consistency)

        if weight_patterns:
            most_common_weight = Counter(weight_patterns).most_common(1)[0]
            weight_consistency = most_common_weight[1] / len(weight_patterns)
            consistencies.append(weight_consistency)

        if zip_patterns:
            most_common_zip = Counter(zip_patterns).most_common(1)[0]
            zip_consistency = most_common_zip[1] / len(zip_patterns)
            consistencies.append(zip_consistency)

        # Return average consistency across all found patterns
        if consistencies:
            return float(np.mean(consistencies))
        else:
            # No recognizable patterns found - return neutral score
            return 0.5

    def _data_validation(self, extracted_data: ExtractedShippingData) -> float:
        """Sanity checks on extracted values.

        Checks:
        - Tracking number format valid
        - Weight reasonable (1-5000 lbs typical)
        - Zip code format valid
        - Date format valid (if present)
        - No placeholder/garbage text

        Args:
            extracted_data: Extracted shipping data

        Returns:
            Percentage of validations passed (0-1)
        """
        validations = []

        # Validate tracking number if present
        if extracted_data.tracking_number:
            is_valid = DataValidator.validate_tracking_number(
                extracted_data.tracking_number
            )
            validations.append(is_valid)

            # Also check if it's garbage text
            is_not_garbage = not DataValidator.detect_garbage_text(
                extracted_data.tracking_number
            )
            validations.append(is_not_garbage)

        # Validate weight if present
        if extracted_data.weight is not None:
            is_valid = DataValidator.validate_weight(extracted_data.weight)
            validations.append(is_valid)

        # Validate ZIP code if present
        if extracted_data.destination_zip:
            is_valid = DataValidator.validate_zip_code(extracted_data.destination_zip)
            validations.append(is_valid)

        # Check destination address for garbage text
        if extracted_data.destination_address:
            is_not_garbage = not DataValidator.detect_garbage_text(
                extracted_data.destination_address
            )
            validations.append(is_not_garbage)

        # Check origin address for garbage text
        if extracted_data.origin_address:
            is_not_garbage = not DataValidator.detect_garbage_text(
                extracted_data.origin_address
            )
            validations.append(is_not_garbage)

        # If no validations were performed, return neutral score
        if not validations:
            return 0.5

        # Calculate percentage of validations passed
        validation_score = sum(validations) / len(validations)

        return validation_score
