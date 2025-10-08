"""Main data extraction orchestrator.

This module coordinates all extraction components to parse OCR text
into structured ExtractedShippingData objects.
"""

from typing import Dict, List, Optional

from app.cv_models import ExtractedShippingData

from .address_extractor import AddressExtractor
from .document_classifier import DocumentTypeClassifier
from .field_extractors import CarrierDetector, RegexFieldExtractor, ServiceLevelDetector
from .item_extractor import ItemListExtractor, PackingListSummaryExtractor


class ShippingDataExtractor:
    """High-level orchestrator for data extraction from OCR text."""

    def __init__(self, config: Dict):
        """Initialize extraction pipeline.

        Args:
            config: Configuration dictionary from config.yaml
        """
        self.config = config
        self.extraction_config = config.get('data_extraction', {})

        # Initialize extractors
        self.classifier = DocumentTypeClassifier()
        self.regex_extractor = RegexFieldExtractor()
        self.address_extractor = AddressExtractor()
        self.item_extractor = ItemListExtractor(
            max_items=self.extraction_config.get('field_extraction', {}).get('max_items', 100)
        )

    def extract(self,
                ocr_text: str,
                track_id: int,
                ocr_confidence: float) -> ExtractedShippingData:
        """Complete extraction pipeline.

        Process:
        1. Classify document type
        2. Extract fields based on document type
        3. Validate extracted data
        4. Calculate overall confidence
        5. Return ExtractedShippingData object

        Args:
            ocr_text: OCR text to extract from
            track_id: Pallet track ID
            ocr_confidence: Confidence score from OCR (0-1)

        Returns:
            ExtractedShippingData object with structured fields
        """
        # Step 1: Classify document type
        doc_type, classification_confidence = self.classifier.classify(ocr_text)

        # Step 2: Extract fields based on document type
        extracted_fields = self.extract_by_document_type(ocr_text, doc_type)

        # Step 3: Validate extracted data
        is_valid = self.validate_extracted_data(extracted_fields, doc_type)

        # Step 4: Calculate overall confidence
        overall_confidence = self._calculate_overall_confidence(
            ocr_confidence,
            classification_confidence,
            extracted_fields
        )

        # Step 5: Determine if needs review
        needs_review = self._should_need_review(overall_confidence, is_valid, extracted_fields, doc_type)

        # Build ExtractedShippingData object
        shipping_data = ExtractedShippingData(
            track_id=track_id,
            document_type=doc_type,
            tracking_number=extracted_fields.get('tracking_number'),
            weight=extracted_fields.get('weight'),
            destination_address=extracted_fields.get('destination_address'),
            destination_zip=extracted_fields.get('destination_zip'),
            origin_address=extracted_fields.get('origin_address'),
            items=extracted_fields.get('items', []),
            confidence_score=overall_confidence,
            needs_review=needs_review
        )

        return shipping_data

    def extract_by_document_type(self, ocr_text: str, doc_type: str) -> Dict[str, any]:
        """Extract fields specific to document type.

        BOL → tracking, weight, carrier, addresses
        PACKING_LIST → items, quantities, PO number
        SHIPPING_LABEL → tracking, weight, service level

        Args:
            ocr_text: OCR text to extract from
            doc_type: Document type (BOL, PACKING_LIST, SHIPPING_LABEL, UNKNOWN)

        Returns:
            Dictionary of extracted fields
        """
        fields = {}

        if doc_type == 'BOL':
            fields = self._extract_bol_fields(ocr_text)
        elif doc_type == 'PACKING_LIST':
            fields = self._extract_packing_list_fields(ocr_text)
        elif doc_type == 'SHIPPING_LABEL':
            fields = self._extract_shipping_label_fields(ocr_text)
        else:
            # UNKNOWN - try to extract common fields
            fields = self._extract_common_fields(ocr_text)

        return fields

    def _extract_bol_fields(self, ocr_text: str) -> Dict[str, any]:
        """Extract fields from Bill of Lading.

        Args:
            ocr_text: OCR text

        Returns:
            Dictionary of extracted fields
        """
        fields = {}

        # Extract PRO number (BOL tracking number)
        pro_number = self.regex_extractor.extract_field(ocr_text, 'pro_number')
        if pro_number:
            fields['tracking_number'] = pro_number
        else:
            # Fallback to BOL number
            bol_number = self.regex_extractor.extract_field(ocr_text, 'bol_number')
            if bol_number:
                fields['tracking_number'] = bol_number

        # Extract weight
        weight_str = self.regex_extractor.extract_field(ocr_text, 'weight')
        if weight_str:
            weight_data = self.regex_extractor.parse_weight(weight_str)
            if weight_data:
                fields['weight'] = weight_data['value']

        # Extract carrier
        carrier = CarrierDetector.detect_carrier(ocr_text)
        if carrier:
            fields['carrier'] = carrier

        # Extract addresses
        addresses = self.address_extractor.extract_addresses(ocr_text)
        fields.update(addresses)

        return fields

    def _extract_packing_list_fields(self, ocr_text: str) -> Dict[str, any]:
        """Extract fields from Packing List.

        Args:
            ocr_text: OCR text

        Returns:
            Dictionary of extracted fields
        """
        fields = {}

        # Extract PO number
        po_number = PackingListSummaryExtractor.extract_po_number(ocr_text)
        if po_number:
            fields['po_number'] = po_number

        # Extract invoice number
        invoice_number = PackingListSummaryExtractor.extract_invoice_number(ocr_text)
        if invoice_number:
            fields['invoice_number'] = invoice_number

        # Extract items (if enabled)
        if self.extraction_config.get('field_extraction', {}).get('extract_items', True):
            items = self.item_extractor.extract_items(ocr_text)
            if items:
                fields['items'] = items

                # Calculate totals
                total_qty = self.item_extractor.calculate_total_quantity(items)
                fields['total_quantity'] = total_qty

                total_weight = self.item_extractor.calculate_total_weight(items)
                if total_weight > 0:
                    fields['weight'] = total_weight

        # Extract carton info
        carton_info = self.item_extractor.extract_carton_info(ocr_text)
        if carton_info:
            fields.update(carton_info)

        # Extract addresses (shipper/consignee)
        addresses = self.address_extractor.extract_addresses(ocr_text)
        fields.update(addresses)

        # Extract date
        date_str = PackingListSummaryExtractor.extract_order_date(ocr_text)
        if date_str:
            fields['order_date'] = date_str

        return fields

    def _extract_shipping_label_fields(self, ocr_text: str) -> Dict[str, any]:
        """Extract fields from Shipping Label.

        Args:
            ocr_text: OCR text

        Returns:
            Dictionary of extracted fields
        """
        fields = {}

        # Extract tracking number
        tracking = self.regex_extractor.extract_field(ocr_text, 'tracking_number')
        if tracking:
            fields['tracking_number'] = tracking

        # Extract carrier
        carrier = CarrierDetector.detect_carrier(ocr_text)
        if carrier:
            fields['carrier'] = carrier

            # Validate tracking number against carrier
            if tracking:
                is_valid = self.regex_extractor.validate_tracking_number(tracking, carrier)
                fields['tracking_valid'] = is_valid

        # Extract weight
        weight_str = self.regex_extractor.extract_field(ocr_text, 'weight')
        if weight_str:
            weight_data = self.regex_extractor.parse_weight(weight_str)
            if weight_data:
                fields['weight'] = weight_data['value']

        # Extract service level
        service = ServiceLevelDetector.detect_service_level(ocr_text)
        if service:
            fields['service_level'] = service

        # Extract addresses
        addresses = self.address_extractor.extract_addresses(ocr_text)
        fields.update(addresses)

        return fields

    def _extract_common_fields(self, ocr_text: str) -> Dict[str, any]:
        """Extract common fields (fallback for UNKNOWN documents).

        Args:
            ocr_text: OCR text

        Returns:
            Dictionary of extracted fields
        """
        # Extract all common fields
        fields = self.regex_extractor.extract_all_fields(ocr_text)

        # Parse weight
        if fields.get('weight'):
            weight_data = self.regex_extractor.parse_weight(fields['weight'])
            if weight_data:
                fields['weight'] = weight_data['value']

        # Extract addresses
        addresses = self.address_extractor.extract_addresses(ocr_text)
        fields.update(addresses)

        # Detect carrier
        carrier = CarrierDetector.detect_carrier(ocr_text)
        if carrier:
            fields['carrier'] = carrier

        return fields

    def validate_extracted_data(self, data: Dict, doc_type: str) -> bool:
        """Validate that extracted data makes sense.

        Checks:
        - Required fields present for document type
        - Field formats valid (zip codes, dates, etc.)
        - Values reasonable (weight > 0, etc.)

        Args:
            data: Extracted data dictionary
            doc_type: Document type

        Returns:
            True if data is valid
        """
        validation_config = self.extraction_config.get('validation', {})

        # Check required tracking number
        if validation_config.get('require_tracking_number', True):
            if doc_type in ['BOL', 'SHIPPING_LABEL']:
                if not data.get('tracking_number'):
                    return False

        # Check weight if present
        weight = data.get('weight')
        if weight is not None:
            min_weight = validation_config.get('min_weight', 1.0)
            max_weight = validation_config.get('max_weight', 5000.0)
            if weight < min_weight or weight > max_weight:
                return False

        # Validate addresses if required
        if validation_config.get('validate_addresses', True):
            if doc_type in ['BOL', 'SHIPPING_LABEL']:
                # Should have at least destination city and state
                if not (data.get('destination_city') and data.get('destination_state')):
                    return False

        # Validate items for packing lists
        if doc_type == 'PACKING_LIST':
            items = data.get('items', [])
            if items:
                # Validate items are reasonable
                if not ItemListExtractor.validate_items(items):
                    return False

        return True

    def _calculate_overall_confidence(self,
                                     ocr_confidence: float,
                                     classification_confidence: float,
                                     extracted_fields: Dict) -> float:
        """Calculate overall extraction confidence.

        Combines:
        - OCR confidence
        - Document classification confidence
        - Field extraction success rate

        Args:
            ocr_confidence: OCR confidence score
            classification_confidence: Document classification confidence
            extracted_fields: Extracted fields dictionary

        Returns:
            Overall confidence score (0-1)
        """
        # Weight factors
        ocr_weight = 0.4
        classification_weight = 0.3
        extraction_weight = 0.3

        # Calculate field extraction score (percentage of fields extracted)
        # Define important fields
        important_fields = ['tracking_number', 'weight', 'destination_address']
        extracted_important = sum(1 for f in important_fields if extracted_fields.get(f))
        extraction_score = extracted_important / len(important_fields) if important_fields else 0.0

        # Combine scores
        overall = (
            ocr_confidence * ocr_weight +
            classification_confidence * classification_weight +
            extraction_score * extraction_weight
        )

        return min(1.0, max(0.0, overall))

    def _should_need_review(self,
                           confidence: float,
                           is_valid: bool,
                           extracted_fields: Dict,
                           doc_type: str) -> bool:
        """Determine if extracted data needs manual review.

        Args:
            confidence: Overall confidence score
            is_valid: Whether data passed validation
            extracted_fields: Extracted fields
            doc_type: Document type

        Returns:
            True if data needs review
        """
        confidence_thresholds = self.config.get('confidence', {})

        # Auto-reject if confidence too low
        auto_reject = confidence_thresholds.get('auto_reject', 0.40)
        if confidence < auto_reject:
            return True

        # Auto-accept if confidence high enough
        auto_accept = confidence_thresholds.get('auto_accept', 0.85)
        if confidence >= auto_accept and is_valid:
            return False

        # In between - needs review
        needs_review_threshold = confidence_thresholds.get('needs_review', 0.60)
        if confidence < needs_review_threshold:
            return True

        # Check for missing critical fields
        if doc_type in ['BOL', 'SHIPPING_LABEL']:
            if not extracted_fields.get('tracking_number'):
                return True

        # Failed validation
        if not is_valid:
            return True

        return False

    def extract_with_details(self,
                            ocr_text: str,
                            track_id: int,
                            ocr_confidence: float) -> Dict:
        """Extract data with detailed diagnostic information.

        Useful for debugging and understanding extraction results.

        Args:
            ocr_text: OCR text to extract from
            track_id: Pallet track ID
            ocr_confidence: OCR confidence score

        Returns:
            Dictionary with extraction details:
            {
                'extracted_data': ExtractedShippingData,
                'classification_details': Dict,
                'all_extracted_fields': Dict,
                'validation_result': bool
            }
        """
        # Classify with details
        classification_details = self.classifier.classify_with_details(ocr_text)
        doc_type = classification_details['document_type']

        # Extract fields
        extracted_fields = self.extract_by_document_type(ocr_text, doc_type)

        # Validate
        is_valid = self.validate_extracted_data(extracted_fields, doc_type)

        # Calculate confidence
        overall_confidence = self._calculate_overall_confidence(
            ocr_confidence,
            classification_details['confidence'],
            extracted_fields
        )

        # Determine review status
        needs_review = self._should_need_review(overall_confidence, is_valid, extracted_fields, doc_type)

        # Build ExtractedShippingData
        shipping_data = ExtractedShippingData(
            track_id=track_id,
            document_type=doc_type,
            tracking_number=extracted_fields.get('tracking_number'),
            weight=extracted_fields.get('weight'),
            destination_address=extracted_fields.get('destination_address'),
            destination_zip=extracted_fields.get('destination_zip'),
            origin_address=extracted_fields.get('origin_address'),
            items=extracted_fields.get('items', []),
            confidence_score=overall_confidence,
            needs_review=needs_review
        )

        return {
            'extracted_data': shipping_data,
            'classification_details': classification_details,
            'all_extracted_fields': extracted_fields,
            'validation_result': is_valid
        }
