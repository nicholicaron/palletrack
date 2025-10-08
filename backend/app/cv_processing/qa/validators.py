"""Data validators for extracted shipping information."""

import re
from datetime import datetime
from typing import Optional


class DataValidator:
    """Validate extracted field values for shipping documents."""

    @staticmethod
    def validate_tracking_number(tracking: Optional[str]) -> bool:
        """Validate tracking number format.

        Common formats:
        - UPS: 1Z followed by 16 alphanumeric
        - FedEx: 12-15 digits
        - USPS: 20-22 digits

        Args:
            tracking: Tracking number string to validate

        Returns:
            True if tracking number matches a valid format
        """
        if not tracking or not isinstance(tracking, str):
            return False

        # Remove whitespace
        tracking = tracking.strip()

        if not tracking:
            return False

        # UPS: 1Z followed by 16 alphanumeric characters
        ups_pattern = r'^1Z[A-Z0-9]{16}$'
        if re.match(ups_pattern, tracking, re.IGNORECASE):
            return True

        # FedEx: 12-15 digits
        fedex_pattern = r'^\d{12,15}$'
        if re.match(fedex_pattern, tracking):
            return True

        # USPS: 20-22 digits
        usps_pattern = r'^\d{20,22}$'
        if re.match(usps_pattern, tracking):
            return True

        return False

    @staticmethod
    def validate_weight(weight: Optional[float]) -> bool:
        """Validate weight is reasonable for pallet.

        Args:
            weight: Weight in pounds

        Returns:
            True if weight is in reasonable range (1-5000 lbs)
        """
        if weight is None or not isinstance(weight, (int, float)):
            return False

        return 1.0 <= weight <= 5000.0

    @staticmethod
    def validate_zip_code(zip_code: Optional[str]) -> bool:
        """Validate US zip code format.

        Args:
            zip_code: ZIP code string to validate

        Returns:
            True if ZIP code matches standard US format (5 digits or 5+4)
        """
        if not zip_code or not isinstance(zip_code, str):
            return False

        # Remove whitespace
        zip_code = zip_code.strip()

        # Standard format: 5 digits or 5+4 digits
        pattern = r'^\d{5}(-\d{4})?$'
        return bool(re.match(pattern, zip_code))

    @staticmethod
    def validate_date(date_str: Optional[str]) -> bool:
        """Validate date format and reasonableness.

        Accepts common date formats:
        - MM/DD/YYYY
        - MM-DD-YYYY
        - YYYY-MM-DD
        - Month DD, YYYY

        Args:
            date_str: Date string to validate

        Returns:
            True if date can be parsed and is reasonable (not too far in past/future)
        """
        if not date_str or not isinstance(date_str, str):
            return False

        date_str = date_str.strip()

        # Common date patterns
        date_patterns = [
            r'^\d{1,2}/\d{1,2}/\d{4}$',  # MM/DD/YYYY
            r'^\d{1,2}-\d{1,2}-\d{4}$',  # MM-DD-YYYY
            r'^\d{4}-\d{1,2}-\d{1,2}$',  # YYYY-MM-DD
        ]

        # Check if matches a pattern
        matches_pattern = any(re.match(pattern, date_str) for pattern in date_patterns)

        if not matches_pattern:
            # Try to check for written month format (e.g., "January 15, 2024")
            month_pattern = r'^(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}$'
            if not re.match(month_pattern, date_str, re.IGNORECASE):
                return False

        # Try to parse the date to ensure it's valid
        try:
            # Try common formats
            for fmt in ['%m/%d/%Y', '%m-%d-%Y', '%Y-%m-%d', '%B %d, %Y']:
                try:
                    parsed_date = datetime.strptime(date_str, fmt)
                    # Check if date is reasonable (within 10 years past/future)
                    now = datetime.now()
                    years_diff = abs((parsed_date - now).days / 365.25)
                    return years_diff <= 10
                except ValueError:
                    continue
            return False
        except Exception:
            return False

    @staticmethod
    def detect_garbage_text(text: Optional[str]) -> bool:
        """Detect if text is garbage/placeholder.

        Indicators:
        - Too many special characters (>40% of text)
        - No vowels in words longer than 3 chars
        - Repetitive patterns (same char repeated >5 times)
        - All same character

        Args:
            text: Text to check

        Returns:
            True if text appears to be garbage, False if it seems valid
        """
        if not text or not isinstance(text, str):
            return True

        text = text.strip()

        if len(text) == 0:
            return True

        # Check if all same character
        if len(set(text.replace(' ', ''))) == 1:
            return True

        # Check for excessive special characters
        special_char_count = sum(1 for c in text if not c.isalnum() and not c.isspace())
        special_char_ratio = special_char_count / len(text) if len(text) > 0 else 0
        if special_char_ratio > 0.4:
            return True

        # Check for repetitive patterns (same character repeated >5 times)
        if re.search(r'(.)\1{5,}', text):
            return True

        # Check for lack of vowels in longer words
        words = text.split()
        long_words = [w for w in words if len(w) > 3]
        if long_words:
            vowels = set('aeiouAEIOU')
            words_without_vowels = sum(
                1 for word in long_words
                if not any(c in vowels for c in word)
            )
            # If more than 70% of long words have no vowels, likely garbage
            if words_without_vowels / len(long_words) > 0.7:
                return True

        return False

    @staticmethod
    def validate_po_number(po_number: Optional[str]) -> bool:
        """Validate Purchase Order number format.

        Args:
            po_number: PO number string to validate

        Returns:
            True if PO number seems valid (not empty, not garbage)
        """
        if not po_number or not isinstance(po_number, str):
            return False

        po_number = po_number.strip()

        # Check minimum length
        if len(po_number) < 3:
            return False

        # Check if it's garbage text
        if DataValidator.detect_garbage_text(po_number):
            return False

        # PO numbers are typically alphanumeric with possible dashes/slashes
        # Allow some flexibility but ensure it's mostly alphanumeric
        alphanumeric_count = sum(1 for c in po_number if c.isalnum())
        if alphanumeric_count / len(po_number) < 0.5:
            return False

        return True
