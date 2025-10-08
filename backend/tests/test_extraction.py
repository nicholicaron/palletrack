"""Unit tests for data extraction components."""

import pytest

from app.cv_processing.extraction import (
    AddressExtractor,
    CarrierDetector,
    CompanyNameExtractor,
    DocumentTypeClassifier,
    ItemListExtractor,
    PackingListSummaryExtractor,
    RegexFieldExtractor,
    ServiceLevelDetector,
    ShippingDataExtractor,
)


# Sample OCR texts for different document types
SAMPLE_BOL_TEXT = """
BILL OF LADING
BOL #: BOL-123456789

SHIPPER:
ACME Manufacturing Inc
456 Factory Road
Chicago, IL 60601

CONSIGNEE:
Widget Distributors LLC
123 Main Street
Boston, MA 02101

PRO NUMBER: PRO-987654321
CARRIER: FEDEX FREIGHT
SCAC: FXFE

WEIGHT: 450 lbs
FREIGHT CHARGES: $250.00
"""

SAMPLE_PACKING_LIST_TEXT = """
PACKING LIST

PO Number: PO-2024-5678
Invoice: INV-123456
Date: 03/15/2024

SHIP TO:
Tech Solutions Corp
789 Business Blvd
Austin, TX 78701

ITEM    DESCRIPTION         QTY    SKU
1       Widget Type A       100    WID-001-A
2       Widget Type B       50     WID-002-B
3       Connector Kit       25     CON-KIT-5

TOTAL QTY: 175
CARTONS: 5
"""

SAMPLE_SHIPPING_LABEL_TEXT = """
FEDEX
Ground Service

TRACKING: 1Z999AA10123456784

SHIP TO:
Sarah Johnson
987 Oak Avenue
Portland, OR 97201

SHIP FROM:
Distribution Center
555 Warehouse Drive
Memphis, TN 38103

WEIGHT: 25.5 lbs
SERVICE: GROUND
DELIVERY BY: 03/20/2024
"""

SAMPLE_UNKNOWN_TEXT = """
Some random text without
clear document structure
or shipping keywords
"""


class TestDocumentTypeClassifier:
    """Test DocumentTypeClassifier."""

    def test_classify_bol(self):
        """Test BOL classification."""
        classifier = DocumentTypeClassifier()
        doc_type, confidence = classifier.classify(SAMPLE_BOL_TEXT)

        assert doc_type == 'BOL'
        assert confidence > 0.5

    def test_classify_packing_list(self):
        """Test packing list classification."""
        classifier = DocumentTypeClassifier()
        doc_type, confidence = classifier.classify(SAMPLE_PACKING_LIST_TEXT)

        assert doc_type == 'PACKING_LIST'
        assert confidence > 0.5

    def test_classify_shipping_label(self):
        """Test shipping label classification."""
        classifier = DocumentTypeClassifier()
        doc_type, confidence = classifier.classify(SAMPLE_SHIPPING_LABEL_TEXT)

        assert doc_type == 'SHIPPING_LABEL'
        assert confidence > 0.5

    def test_classify_unknown(self):
        """Test unknown document classification."""
        classifier = DocumentTypeClassifier()
        doc_type, confidence = classifier.classify(SAMPLE_UNKNOWN_TEXT)

        assert doc_type == 'UNKNOWN'
        assert confidence < 0.3

    def test_classify_empty_text(self):
        """Test classification with empty text."""
        classifier = DocumentTypeClassifier()
        doc_type, confidence = classifier.classify("")

        assert doc_type == 'UNKNOWN'
        assert confidence == 0.0

    def test_classify_with_details(self):
        """Test detailed classification."""
        classifier = DocumentTypeClassifier()
        details = classifier.classify_with_details(SAMPLE_BOL_TEXT)

        assert details['document_type'] == 'BOL'
        assert 'all_scores' in details
        assert 'matched_keywords' in details
        assert len(details['matched_keywords']['BOL']) > 0


class TestRegexFieldExtractor:
    """Test RegexFieldExtractor."""

    def test_extract_tracking_number(self):
        """Test tracking number extraction."""
        extractor = RegexFieldExtractor()
        tracking = extractor.extract_field(SAMPLE_SHIPPING_LABEL_TEXT, 'tracking_number')

        assert tracking is not None
        assert '1Z999AA10123456784' in tracking

    def test_extract_weight(self):
        """Test weight extraction."""
        extractor = RegexFieldExtractor()
        weight = extractor.extract_field(SAMPLE_SHIPPING_LABEL_TEXT, 'weight')

        assert weight is not None
        assert '25.5' in weight

    def test_extract_po_number(self):
        """Test PO number extraction."""
        extractor = RegexFieldExtractor()
        po = extractor.extract_field(SAMPLE_PACKING_LIST_TEXT, 'po_number')

        assert po is not None
        assert 'PO-2024-5678' in po

    def test_extract_zip_code(self):
        """Test ZIP code extraction."""
        extractor = RegexFieldExtractor()
        zip_code = extractor.extract_field(SAMPLE_SHIPPING_LABEL_TEXT, 'zip_code')

        assert zip_code is not None
        assert zip_code in ['97201', '38103', '78701']  # Multiple ZIPs in text

    def test_extract_all_fields(self):
        """Test extracting all fields at once."""
        extractor = RegexFieldExtractor()
        fields = extractor.extract_all_fields(SAMPLE_SHIPPING_LABEL_TEXT)

        assert isinstance(fields, dict)
        assert fields.get('tracking_number') is not None
        assert fields.get('weight') is not None

    def test_validate_tracking_number(self):
        """Test tracking number validation."""
        assert RegexFieldExtractor.validate_tracking_number('1Z999AA10123456784') is True
        assert RegexFieldExtractor.validate_tracking_number('1Z999AA10123456784', 'UPS') is True
        assert RegexFieldExtractor.validate_tracking_number('123456789012', 'FEDEX') is True
        assert RegexFieldExtractor.validate_tracking_number('invalid') is False

    def test_parse_weight(self):
        """Test weight parsing."""
        weight_data = RegexFieldExtractor.parse_weight('25.5 lbs')

        assert weight_data is not None
        assert weight_data['value'] == 25.5
        assert weight_data['unit'] == 'lbs'

    def test_parse_weight_kg(self):
        """Test weight parsing with kg."""
        weight_data = RegexFieldExtractor.parse_weight('10 kg')

        assert weight_data is not None
        assert weight_data['value'] == 10.0
        assert weight_data['unit'] == 'kg'

    def test_parse_dimensions(self):
        """Test dimensions parsing."""
        dims = RegexFieldExtractor.parse_dimensions('12 x 8 x 6 in')

        assert dims is not None
        assert dims['length'] == 12.0
        assert dims['width'] == 8.0
        assert dims['height'] == 6.0
        assert dims['unit'] == 'in'

    def test_extract_all_occurrences(self):
        """Test extracting all occurrences of a field."""
        extractor = RegexFieldExtractor()
        zips = extractor.extract_all_occurrences(SAMPLE_PACKING_LIST_TEXT, 'zip_code')

        assert isinstance(zips, list)
        assert len(zips) > 0


class TestCarrierDetector:
    """Test CarrierDetector."""

    def test_detect_ups(self):
        """Test UPS detection."""
        text = "UPS Ground Tracking: 1Z999AA10123456784"
        carrier = CarrierDetector.detect_carrier(text)
        assert carrier == 'UPS'

    def test_detect_fedex(self):
        """Test FedEx detection."""
        carrier = CarrierDetector.detect_carrier(SAMPLE_BOL_TEXT)
        assert carrier == 'FEDEX'

    def test_detect_no_carrier(self):
        """Test no carrier detected."""
        carrier = CarrierDetector.detect_carrier("No carrier mentioned")
        assert carrier is None


class TestServiceLevelDetector:
    """Test ServiceLevelDetector."""

    def test_detect_ground(self):
        """Test ground service detection."""
        service = ServiceLevelDetector.detect_service_level(SAMPLE_SHIPPING_LABEL_TEXT)
        assert service == 'GROUND'

    def test_detect_overnight(self):
        """Test overnight service detection."""
        text = "FedEx Priority Overnight Delivery"
        service = ServiceLevelDetector.detect_service_level(text)
        assert service == 'OVERNIGHT'

    def test_detect_2day(self):
        """Test 2-day service detection."""
        text = "UPS 2 Day Air"
        service = ServiceLevelDetector.detect_service_level(text)
        assert service == '2DAY'


class TestAddressExtractor:
    """Test AddressExtractor."""

    def test_extract_addresses(self):
        """Test address extraction."""
        extractor = AddressExtractor()
        addresses = extractor.extract_addresses(SAMPLE_BOL_TEXT)

        assert addresses is not None
        assert addresses.get('destination_city') == 'Boston'
        assert addresses.get('destination_state') == 'MA'
        assert addresses.get('destination_zip') == '02101'
        assert addresses.get('origin_city') == 'Chicago'
        assert addresses.get('origin_state') == 'IL'

    def test_parse_address_components(self):
        """Test parsing address into components."""
        address_block = """
        123 Main Street
        Boston, MA 02101
        """
        components = AddressExtractor.parse_address_components(address_block)

        assert components['city'] == 'Boston'
        assert components['state'] == 'MA'
        assert components['zip_code'] == '02101'

    def test_validate_address(self):
        """Test address validation."""
        valid_address = {
            'city': 'Boston',
            'state': 'MA',
            'zip_code': '02101'
        }
        assert AddressExtractor.validate_address(valid_address) is True

        invalid_address = {
            'city': None,
            'state': None
        }
        assert AddressExtractor.validate_address(invalid_address) is False

    def test_extract_all_zip_codes(self):
        """Test extracting all ZIP codes."""
        extractor = AddressExtractor()
        zips = extractor.extract_all_zip_codes(SAMPLE_BOL_TEXT)

        assert len(zips) == 2
        assert '02101' in zips
        assert '60601' in zips

    def test_extract_all_cities_states(self):
        """Test extracting all city/state pairs."""
        extractor = AddressExtractor()
        cities = extractor.extract_all_cities_states(SAMPLE_BOL_TEXT)

        assert len(cities) >= 1
        assert ('Boston', 'MA') in cities or ('Chicago', 'IL') in cities

    def test_format_address(self):
        """Test address formatting."""
        components = {
            'street': '123 Main St',
            'city': 'Boston',
            'state': 'MA',
            'zip_code': '02101'
        }
        formatted = AddressExtractor.format_address(components)

        assert 'Boston' in formatted
        assert 'MA' in formatted
        assert '02101' in formatted


class TestCompanyNameExtractor:
    """Test CompanyNameExtractor."""

    def test_extract_company_name_with_suffix(self):
        """Test extracting company name with business suffix."""
        address_block = """
        ACME Manufacturing Inc
        456 Factory Road
        Chicago, IL 60601
        """
        company = CompanyNameExtractor.extract_company_name(address_block)

        assert company == 'ACME Manufacturing Inc'

    def test_extract_company_name_uppercase(self):
        """Test extracting all-caps company name."""
        address_block = """
        WIDGET DISTRIBUTORS
        123 Main Street
        Boston, MA 02101
        """
        company = CompanyNameExtractor.extract_company_name(address_block)

        assert company == 'WIDGET DISTRIBUTORS'


class TestItemListExtractor:
    """Test ItemListExtractor."""

    def test_extract_items(self):
        """Test item extraction from packing list."""
        extractor = ItemListExtractor()
        items = extractor.extract_items(SAMPLE_PACKING_LIST_TEXT)

        assert len(items) > 0
        assert any('Widget' in item.get('description', '') for item in items)

    def test_detect_table_structure(self):
        """Test table structure detection."""
        extractor = ItemListExtractor()
        table_info = extractor.detect_table_structure(SAMPLE_PACKING_LIST_TEXT)

        assert table_info is not None
        assert table_info.get('has_table') is True

    def test_calculate_total_quantity(self):
        """Test total quantity calculation."""
        extractor = ItemListExtractor()
        items = [
            {'quantity': 10},
            {'quantity': 20},
            {'quantity': 30}
        ]
        total = extractor.calculate_total_quantity(items)

        assert total == 60

    def test_calculate_total_weight(self):
        """Test total weight calculation."""
        extractor = ItemListExtractor()
        items = [
            {'weight': 10.5},
            {'weight': 20.0},
            {'weight': 5.5}
        ]
        total = extractor.calculate_total_weight(items)

        assert total == 36.0

    def test_validate_items(self):
        """Test item validation."""
        valid_items = [
            {'description': 'Widget A', 'quantity': 10},
            {'description': 'Widget B', 'quantity': 20}
        ]
        assert ItemListExtractor.validate_items(valid_items) is True

        invalid_items = []
        assert ItemListExtractor.validate_items(invalid_items) is False

    def test_extract_carton_info(self):
        """Test carton info extraction."""
        extractor = ItemListExtractor()
        carton_info = extractor.extract_carton_info(SAMPLE_PACKING_LIST_TEXT)

        assert carton_info is not None
        assert carton_info.get('carton_count') == 5


class TestPackingListSummaryExtractor:
    """Test PackingListSummaryExtractor."""

    def test_extract_po_number(self):
        """Test PO number extraction."""
        po = PackingListSummaryExtractor.extract_po_number(SAMPLE_PACKING_LIST_TEXT)

        assert po is not None
        assert 'PO-2024-5678' in po

    def test_extract_invoice_number(self):
        """Test invoice number extraction."""
        invoice = PackingListSummaryExtractor.extract_invoice_number(SAMPLE_PACKING_LIST_TEXT)

        assert invoice is not None
        assert 'INV-123456' in invoice

    def test_extract_order_date(self):
        """Test order date extraction."""
        date = PackingListSummaryExtractor.extract_order_date(SAMPLE_PACKING_LIST_TEXT)

        assert date is not None
        assert '03/15/2024' in date


class TestShippingDataExtractor:
    """Test ShippingDataExtractor main orchestrator."""

    @pytest.fixture
    def sample_config(self):
        """Sample configuration."""
        return {
            'data_extraction': {
                'classification': {
                    'min_confidence': 0.6,
                    'require_all_keywords': False
                },
                'field_extraction': {
                    'extract_items': True,
                    'max_items': 100
                },
                'validation': {
                    'require_tracking_number': True,
                    'require_weight': False,
                    'min_weight': 1.0,
                    'max_weight': 5000.0,
                    'validate_addresses': True
                }
            },
            'confidence': {
                'auto_accept': 0.85,
                'needs_review': 0.60,
                'auto_reject': 0.40
            }
        }

    def test_extract_bol_data(self, sample_config):
        """Test extracting data from BOL."""
        extractor = ShippingDataExtractor(sample_config)
        result = extractor.extract(SAMPLE_BOL_TEXT, track_id=1, ocr_confidence=0.9)

        assert result.track_id == 1
        assert result.document_type == 'BOL'
        assert result.tracking_number is not None
        assert result.destination_zip == '02101'
        assert result.confidence_score > 0.0

    def test_extract_packing_list_data(self, sample_config):
        """Test extracting data from packing list."""
        extractor = ShippingDataExtractor(sample_config)
        result = extractor.extract(SAMPLE_PACKING_LIST_TEXT, track_id=2, ocr_confidence=0.85)

        assert result.track_id == 2
        assert result.document_type == 'PACKING_LIST'
        assert len(result.items) > 0

    def test_extract_shipping_label_data(self, sample_config):
        """Test extracting data from shipping label."""
        extractor = ShippingDataExtractor(sample_config)
        result = extractor.extract(SAMPLE_SHIPPING_LABEL_TEXT, track_id=3, ocr_confidence=0.92)

        assert result.track_id == 3
        assert result.document_type == 'SHIPPING_LABEL'
        assert result.tracking_number is not None
        assert result.weight is not None

    def test_extract_unknown_document(self, sample_config):
        """Test extracting data from unknown document."""
        extractor = ShippingDataExtractor(sample_config)
        result = extractor.extract(SAMPLE_UNKNOWN_TEXT, track_id=4, ocr_confidence=0.5)

        assert result.track_id == 4
        assert result.document_type == 'UNKNOWN'

    def test_extract_with_details(self, sample_config):
        """Test detailed extraction."""
        extractor = ShippingDataExtractor(sample_config)
        details = extractor.extract_with_details(SAMPLE_BOL_TEXT, track_id=1, ocr_confidence=0.9)

        assert 'extracted_data' in details
        assert 'classification_details' in details
        assert 'all_extracted_fields' in details
        assert 'validation_result' in details

    def test_needs_review_low_confidence(self, sample_config):
        """Test that low confidence triggers review."""
        extractor = ShippingDataExtractor(sample_config)
        result = extractor.extract(SAMPLE_BOL_TEXT, track_id=1, ocr_confidence=0.3)

        assert result.needs_review is True

    def test_validate_extracted_data(self, sample_config):
        """Test data validation."""
        extractor = ShippingDataExtractor(sample_config)

        valid_data = {
            'tracking_number': 'ABC123',
            'weight': 50.0,
            'destination_city': 'Boston',
            'destination_state': 'MA'
        }
        assert extractor.validate_extracted_data(valid_data, 'BOL') is True

        invalid_data = {
            'tracking_number': None,
            'destination_city': None
        }
        assert extractor.validate_extracted_data(invalid_data, 'BOL') is False


class TestIntegrationExtraction:
    """Integration tests for complete extraction pipeline."""

    @pytest.fixture
    def sample_config(self):
        """Sample configuration."""
        return {
            'data_extraction': {
                'classification': {
                    'min_confidence': 0.6,
                    'require_all_keywords': False
                },
                'field_extraction': {
                    'extract_items': True,
                    'max_items': 100
                },
                'validation': {
                    'require_tracking_number': True,
                    'require_weight': False,
                    'min_weight': 1.0,
                    'max_weight': 5000.0,
                    'validate_addresses': True
                }
            },
            'confidence': {
                'auto_accept': 0.85,
                'needs_review': 0.60,
                'auto_reject': 0.40
            }
        }

    def test_end_to_end_bol_extraction(self, sample_config):
        """Test complete end-to-end BOL extraction."""
        extractor = ShippingDataExtractor(sample_config)
        result = extractor.extract(SAMPLE_BOL_TEXT, track_id=1, ocr_confidence=0.9)

        # Verify all key fields extracted
        assert result.document_type == 'BOL'
        assert result.tracking_number is not None
        assert result.weight is not None
        assert result.destination_address is not None
        assert result.origin_address is not None
        assert result.confidence_score > 0.5

    def test_end_to_end_packing_list_extraction(self, sample_config):
        """Test complete end-to-end packing list extraction."""
        extractor = ShippingDataExtractor(sample_config)
        result = extractor.extract(SAMPLE_PACKING_LIST_TEXT, track_id=2, ocr_confidence=0.85)

        # Verify key fields
        assert result.document_type == 'PACKING_LIST'
        assert len(result.items) > 0
        assert result.destination_zip == '78701'
        assert result.confidence_score > 0.5
