"""Field extraction using regex patterns.

This module provides regex-based extractors for common shipping document fields:
- Tracking numbers (UPS, FedEx, USPS)
- Weights and dimensions
- PO numbers
- Dates
- Zip codes
"""

import re
from datetime import datetime
from typing import Dict, List, Optional


class RegexFieldExtractor:
    """Extract structured fields using regex patterns."""

    # Common patterns for various fields
    PATTERNS = {
        'tracking_number': [
            r'(?:tracking|PRO|tracking\s*#|PRO\s*#)[:\s]*([A-Z0-9]{10,})',
            r'\b(1Z[0-9A-Z]{16})\b',  # UPS format
            r'\b(\d{12})\b',  # FedEx 12-digit
            r'\b(\d{15})\b',  # FedEx 15-digit
            r'\b([0-9]{20,22})\b',  # USPS 20-22 digits
        ],
        'weight': [
            r'(?:weight|wt|gross\s*weight)[:\s]*(\d+(?:\.\d+)?)\s*(?:lbs?|pounds?|kg|kilograms?)',
            r'(\d+(?:\.\d+)?)\s*(?:lbs?|pounds?)',
            r'weight[:\s]*(\d+)',
        ],
        'po_number': [
            r'(?:PO|P\.O\.|purchase\s*order)[:\s#]*([A-Z0-9-]{3,20})',
            r'PO[#:\s]*([A-Z0-9-]+)',
        ],
        'zip_code': [
            r'\b(\d{5}(?:-\d{4})?)\b',  # US ZIP codes
        ],
        'date': [
            r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',  # MM/DD/YYYY or DD/MM/YYYY
            r'(\d{4}-\d{2}-\d{2})',  # YYYY-MM-DD
            r'(?:date|shipped|delivery)[:\s]*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
        ],
        'pro_number': [
            r'(?:PRO|PRO\s*#|PRO\s*NUMBER)[:\s]*([A-Z0-9-]{6,20})',
        ],
        'bol_number': [
            r'(?:BOL|B/L|BILL\s*OF\s*LADING)[:\s#]*([A-Z0-9-]{6,20})',
        ],
        'carrier': [
            r'\b(UPS|FEDEX|FED\s*EX|USPS|DHL)\b',
            r'(?:carrier|scac)[:\s]*([A-Z]{2,4})',
        ],
        'dimensions': [
            r'(\d+(?:\.\d+)?)\s*x\s*(\d+(?:\.\d+)?)\s*x\s*(\d+(?:\.\d+)?)\s*(?:in|inches|cm)',
        ],
    }

    def __init__(self):
        """Initialize field extractor."""
        pass

    def extract_field(self, text: str, field_name: str) -> Optional[str]:
        """Extract field using predefined patterns.

        Try all patterns for field_name, return first match.

        Args:
            text: Input text to extract from
            field_name: Field name (must exist in PATTERNS)

        Returns:
            Extracted value or None if not found
        """
        if field_name not in self.PATTERNS:
            return None

        patterns = self.PATTERNS[field_name]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                # Return first captured group or full match
                return match.group(1) if match.lastindex else match.group(0)

        return None

    def extract_all_occurrences(self, text: str, field_name: str) -> List[str]:
        """Extract all occurrences of a field.

        Args:
            text: Input text to extract from
            field_name: Field name (must exist in PATTERNS)

        Returns:
            List of all extracted values
        """
        if field_name not in self.PATTERNS:
            return []

        results = []
        patterns = self.PATTERNS[field_name]

        for pattern in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                value = match.group(1) if match.lastindex else match.group(0)
                if value and value not in results:
                    results.append(value)

        return results

    def extract_all_fields(self, text: str) -> Dict[str, Optional[str]]:
        """Extract all defined fields from text.

        Returns dict of field_name → value (or None if not found).

        Args:
            text: Input text to extract from

        Returns:
            Dictionary mapping field names to extracted values
        """
        results = {}

        for field_name in self.PATTERNS.keys():
            value = self.extract_field(text, field_name)
            results[field_name] = value

        return results

    @staticmethod
    def validate_tracking_number(tracking: str, carrier: Optional[str] = None) -> bool:
        """Validate tracking number format.

        Basic validation:
        - Correct length
        - Correct character types
        - Optional: carrier-specific validation

        Args:
            tracking: Tracking number to validate
            carrier: Optional carrier name for carrier-specific validation

        Returns:
            True if valid, False otherwise
        """
        if not tracking:
            return False

        # Remove whitespace
        tracking = tracking.strip().replace(' ', '').replace('-', '')

        # UPS tracking numbers
        if tracking.startswith('1Z'):
            # UPS: 1Z followed by 16 alphanumeric characters
            return len(tracking) == 18 and tracking[2:].isalnum()

        # FedEx tracking numbers
        if carrier and carrier.upper() in ['FEDEX', 'FED EX']:
            # FedEx: 12 or 15 digits
            return tracking.isdigit() and len(tracking) in [12, 15]

        # USPS tracking numbers
        if carrier and carrier.upper() == 'USPS':
            # USPS: 20-22 digits
            return tracking.isdigit() and 20 <= len(tracking) <= 22

        # Generic validation: 10-22 alphanumeric characters
        return 10 <= len(tracking) <= 22 and tracking.isalnum()

    @staticmethod
    def parse_weight(weight_str: str) -> Optional[Dict[str, any]]:
        """Parse weight string into value and unit.

        Args:
            weight_str: Weight string (e.g., "150 lbs", "68 kg")

        Returns:
            Dictionary with 'value' and 'unit' keys, or None if invalid
        """
        if not weight_str:
            return None

        # Extract number and unit
        match = re.search(r'(\d+(?:\.\d+)?)\s*(?:(lbs?|pounds?|kg|kilograms?))?',
                         weight_str, re.IGNORECASE)

        if not match:
            return None

        value = float(match.group(1))
        unit = match.group(2) if match.group(2) else 'lbs'  # Default to lbs

        # Normalize unit
        if unit.lower() in ['lb', 'lbs', 'pound', 'pounds']:
            unit = 'lbs'
        elif unit.lower() in ['kg', 'kilogram', 'kilograms']:
            unit = 'kg'

        return {
            'value': value,
            'unit': unit
        }

    @staticmethod
    def parse_date(date_str: str) -> Optional[datetime]:
        """Parse date string into datetime object.

        Tries common date formats:
        - MM/DD/YYYY
        - DD/MM/YYYY
        - YYYY-MM-DD

        Args:
            date_str: Date string to parse

        Returns:
            datetime object or None if parsing fails
        """
        if not date_str:
            return None

        # Try common formats
        formats = [
            '%m/%d/%Y',
            '%m-%d-%Y',
            '%d/%m/%Y',
            '%d-%m-%Y',
            '%Y-%m-%d',
            '%m/%d/%y',
            '%m-%d-%y',
        ]

        for fmt in formats:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue

        return None

    @staticmethod
    def parse_dimensions(dimensions_str: str) -> Optional[Dict[str, any]]:
        """Parse dimensions string into length, width, height.

        Args:
            dimensions_str: Dimensions string (e.g., "12 x 8 x 6 in")

        Returns:
            Dictionary with 'length', 'width', 'height', 'unit' keys, or None
        """
        if not dimensions_str:
            return None

        # Match pattern like "12 x 8 x 6 in" or "12x8x6 cm"
        match = re.search(
            r'(\d+(?:\.\d+)?)\s*x\s*(\d+(?:\.\d+)?)\s*x\s*(\d+(?:\.\d+)?)\s*(?:(in|inches|cm|centimeters?))?',
            dimensions_str,
            re.IGNORECASE
        )

        if not match:
            return None

        length = float(match.group(1))
        width = float(match.group(2))
        height = float(match.group(3))
        unit = match.group(4) if match.group(4) else 'in'  # Default to inches

        # Normalize unit
        if unit.lower() in ['in', 'inch', 'inches']:
            unit = 'in'
        elif unit.lower() in ['cm', 'centimeter', 'centimeters']:
            unit = 'cm'

        return {
            'length': length,
            'width': width,
            'height': height,
            'unit': unit
        }


class CarrierDetector:
    """Detect shipping carrier from document text."""

    CARRIER_PATTERNS = {
        'UPS': [
            r'\bUPS\b',
            r'United\s+Parcel\s+Service',
            r'1Z[0-9A-Z]{16}',  # UPS tracking format
        ],
        'FEDEX': [
            r'\bFedEx\b',
            r'\bFED\s*EX\b',
            r'Federal\s+Express',
        ],
        'USPS': [
            r'\bUSPS\b',
            r'United\s+States\s+Postal\s+Service',
            r'Priority\s+Mail',
        ],
        'DHL': [
            r'\bDHL\b',
        ],
    }

    @classmethod
    def detect_carrier(cls, text: str) -> Optional[str]:
        """Detect carrier from text.

        Args:
            text: Input text

        Returns:
            Carrier name or None if not detected
        """
        for carrier, patterns in cls.CARRIER_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    return carrier

        return None


class ServiceLevelDetector:
    """Detect service level (overnight, 2-day, ground, etc.)."""

    SERVICE_PATTERNS = {
        'OVERNIGHT': [
            r'overnight',
            r'next\s+day',
            r'express',
            r'priority\s+overnight',
        ],
        '2DAY': [
            r'2\s*day',
            r'two\s+day',
            r'2nd\s+day',
        ],
        'GROUND': [
            r'\bground\b',
            r'standard',
        ],
        'PRIORITY': [
            r'priority',
        ],
    }

    @classmethod
    def detect_service_level(cls, text: str) -> Optional[str]:
        """Detect service level from text.

        Args:
            text: Input text

        Returns:
            Service level or None if not detected
        """
        for service, patterns in cls.SERVICE_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    return service

        return None
