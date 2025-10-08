"""Document type classification from OCR text.

This module identifies document types (BOL, Packing List, Shipping Label)
based on keyword matching and signature analysis.
"""

import re
from typing import Dict, List, Tuple


class DocumentTypeClassifier:
    """Identify document type from OCR text using keyword-based classification."""

    # Define signatures for each document type
    DOCUMENT_SIGNATURES = {
        'BOL': {
            'required_keywords': ['bill of lading', 'carrier', 'shipper', 'consignee'],
            'optional_keywords': ['pro number', 'freight charges', 'scac', 'bol', 'b/l'],
            'weight': 1.0
        },
        'PACKING_LIST': {
            'required_keywords': ['packing list', 'quantity', 'item', 'description'],
            'optional_keywords': ['sku', 'part number', 'carton', 'qty', 'pack list'],
            'weight': 1.0
        },
        'SHIPPING_LABEL': {
            'required_keywords': ['tracking', 'service', 'weight'],
            'optional_keywords': ['ups', 'fedex', 'usps', 'barcode', 'delivery', 'ship to'],
            'weight': 1.0
        }
    }

    def __init__(self):
        """Initialize document classifier."""
        pass

    def classify(self, ocr_text: str) -> Tuple[str, float]:
        """Classify document type from OCR text.

        Process:
        1. Normalize text (lowercase, remove extra spaces)
        2. Calculate match score for each document type
        3. Return type with highest score and confidence

        Args:
            ocr_text: OCR text to classify

        Returns:
            Tuple of (document_type, confidence_score)
            document_type: One of 'BOL', 'PACKING_LIST', 'SHIPPING_LABEL', or 'UNKNOWN'
            confidence_score: Match confidence (0-1)
        """
        if not ocr_text or not ocr_text.strip():
            return 'UNKNOWN', 0.0

        # Normalize text
        normalized_text = self._normalize_text(ocr_text)

        # Calculate scores for each document type
        scores = {}
        for doc_type, signature in self.DOCUMENT_SIGNATURES.items():
            score = self.calculate_match_score(normalized_text, signature)
            scores[doc_type] = score

        # Find best match
        if not scores:
            return 'UNKNOWN', 0.0

        best_type = max(scores.items(), key=lambda x: x[1])
        doc_type, confidence = best_type

        # Return UNKNOWN if confidence too low
        if confidence < 0.3:  # Minimum threshold
            return 'UNKNOWN', confidence

        return doc_type, confidence

    @staticmethod
    def calculate_match_score(text: str, signature: Dict) -> float:
        """Calculate how well text matches document signature.

        Scoring:
        - Required keywords: +2 points each if present
        - Optional keywords: +1 point each if present
        - Normalize by total possible points

        Args:
            text: Normalized text to match against
            signature: Document signature with required/optional keywords

        Returns:
            Match score (0-1)
        """
        required_keywords = signature.get('required_keywords', [])
        optional_keywords = signature.get('optional_keywords', [])

        # Count matches
        required_matches = 0
        for keyword in required_keywords:
            if keyword in text:
                required_matches += 1

        optional_matches = 0
        for keyword in optional_keywords:
            if keyword in text:
                optional_matches += 1

        # Calculate score
        # Required: 2 points each, Optional: 1 point each
        score = (required_matches * 2) + optional_matches
        max_score = (len(required_keywords) * 2) + len(optional_keywords)

        if max_score == 0:
            return 0.0

        return score / max_score

    @staticmethod
    def _normalize_text(text: str) -> str:
        """Normalize text for matching.

        Args:
            text: Input text

        Returns:
            Normalized text (lowercase, collapsed whitespace)
        """
        # Convert to lowercase
        text = text.lower()

        # Replace multiple spaces/newlines with single space
        text = re.sub(r'\s+', ' ', text)

        # Strip leading/trailing whitespace
        text = text.strip()

        return text

    def classify_with_details(self, ocr_text: str) -> Dict:
        """Classify document with detailed match information.

        Args:
            ocr_text: OCR text to classify

        Returns:
            Dictionary with classification details:
            {
                'document_type': str,
                'confidence': float,
                'all_scores': Dict[str, float],
                'matched_keywords': Dict[str, List[str]]
            }
        """
        if not ocr_text or not ocr_text.strip():
            return {
                'document_type': 'UNKNOWN',
                'confidence': 0.0,
                'all_scores': {},
                'matched_keywords': {}
            }

        normalized_text = self._normalize_text(ocr_text)

        # Calculate scores and track matched keywords
        all_scores = {}
        matched_keywords = {}

        for doc_type, signature in self.DOCUMENT_SIGNATURES.items():
            score = self.calculate_match_score(normalized_text, signature)
            all_scores[doc_type] = score

            # Track which keywords matched
            matches = []
            for keyword in signature.get('required_keywords', []):
                if keyword in normalized_text:
                    matches.append(keyword)
            for keyword in signature.get('optional_keywords', []):
                if keyword in normalized_text:
                    matches.append(keyword)

            matched_keywords[doc_type] = matches

        # Find best match
        if not all_scores:
            return {
                'document_type': 'UNKNOWN',
                'confidence': 0.0,
                'all_scores': {},
                'matched_keywords': {}
            }

        best_type = max(all_scores.items(), key=lambda x: x[1])
        doc_type, confidence = best_type

        if confidence < 0.3:
            doc_type = 'UNKNOWN'

        return {
            'document_type': doc_type,
            'confidence': confidence,
            'all_scores': all_scores,
            'matched_keywords': matched_keywords
        }
