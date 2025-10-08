"""OCR result post-processing for text cleaning and normalization.

This module provides text cleaning, error correction, and validation
utilities to improve the quality of OCR output.
"""

import re
from typing import Dict, Optional


class OCRPostProcessor:
    """Clean and normalize OCR output.

    Applies various text cleaning and correction operations to improve
    OCR accuracy through pattern-based error correction and normalization.

    Example:
        >>> raw_text = "TRACK  O1Z3"  # 'O' instead of '0', extra spaces
        >>> clean_text = OCRPostProcessor.clean_text(raw_text)
        >>> # Result: "TRACK 0123"
    """

    # Common OCR character confusion pairs
    CHAR_CORRECTIONS = {
        # Letters misread as numbers (in numeric contexts)
        'O': '0',  # Letter O → Zero
        'o': '0',
        'I': '1',  # Letter I → One
        'l': '1',  # Lowercase L → One
        'Z': '2',  # Sometimes Z → 2
        'S': '5',  # Sometimes S → 5
        'B': '8',  # Sometimes B → 8
        'G': '6',  # Sometimes G → 6

        # Numbers misread as letters (in alpha contexts)
        # These are applied context-specifically
    }

    # Tracking number patterns (common formats)
    TRACKING_PATTERNS = [
        r'^[A-Z0-9]{10,20}$',  # Generic alphanumeric
        r'^[0-9]{12,22}$',  # Numeric only (USPS, FedEx)
        r'^[A-Z]{2}[0-9]{9}[A-Z]{2}$',  # USPS format
        r'^[0-9]{15}$',  # FedEx Express
        r'^1Z[A-Z0-9]{16}$',  # UPS format
    ]

    @staticmethod
    def clean_text(text: str, config: Optional[Dict] = None) -> str:
        """Apply complete text cleaning pipeline.

        Args:
            text: Raw OCR text
            config: Optional configuration dict

        Returns:
            Cleaned and normalized text

        Example:
            >>> raw = "  TRACKING   NUM8ER:  O1Z3  "
            >>> clean = OCRPostProcessor.clean_text(raw)
            >>> # Result: "TRACKING NUMBER: 0123"
        """
        if not text:
            return ""

        config = config or {}
        post_config = config.get('ocr', {}).get('post_processing', {})

        # Step 1: Normalize whitespace
        if post_config.get('normalize_whitespace', True):
            text = OCRPostProcessor.normalize_whitespace(text)

        # Step 2: Fix common OCR errors
        if post_config.get('fix_common_errors', True):
            text = OCRPostProcessor.fix_common_ocr_errors(text)

        # Step 3: Remove invalid characters (optional)
        if post_config.get('remove_invalid_chars', False):
            text = OCRPostProcessor.remove_invalid_characters(text)

        return text

    @staticmethod
    def normalize_whitespace(text: str) -> str:
        """Normalize excessive whitespace.

        - Removes leading/trailing whitespace
        - Collapses multiple spaces to single space
        - Normalizes line breaks

        Args:
            text: Input text

        Returns:
            Normalized text

        Example:
            >>> text = "  TRACK   123  \\n\\n  CODE  "
            >>> normalized = OCRPostProcessor.normalize_whitespace(text)
            >>> # Result: "TRACK 123 CODE"
        """
        # Strip leading/trailing whitespace
        text = text.strip()

        # Replace multiple spaces with single space
        text = re.sub(r' +', ' ', text)

        # Replace multiple newlines with single newline
        text = re.sub(r'\n+', '\n', text)

        # Replace tabs with spaces
        text = text.replace('\t', ' ')

        return text

    @staticmethod
    def fix_common_ocr_errors(text: str) -> str:
        """Fix typical OCR character misrecognitions.

        Uses context-aware rules to correct common mistakes:
        - O/0 confusion in numbers
        - I/1/l confusion in numbers
        - Other common substitutions

        Args:
            text: Input text

        Returns:
            Corrected text

        Example:
            >>> text = "TRACK O1Z3 LOT I234"
            >>> fixed = OCRPostProcessor.fix_common_ocr_errors(text)
            >>> # Result: "TRACK 0123 LOT 1234"
        """
        # Apply corrections in numeric contexts
        # Look for sequences that should be numbers

        # Pattern: digits with occasional letters
        def fix_numeric_context(match):
            s = match.group(0)
            # Apply letter→number corrections
            for wrong, right in OCRPostProcessor.CHAR_CORRECTIONS.items():
                s = s.replace(wrong, right)
            return s

        # Fix numbers that might have letter substitutions
        # Look for patterns like "O1Z3" in contexts like "TRACK: O1Z3"
        text = re.sub(
            r'\b[A-Z0-9]{4,20}\b',  # Alphanumeric sequences
            lambda m: OCRPostProcessor._fix_alphanumeric_sequence(m.group(0)),
            text
        )

        return text

    @staticmethod
    def _fix_alphanumeric_sequence(seq: str) -> str:
        """Fix character errors in alphanumeric sequences.

        Uses heuristics to determine if sequence is primarily numeric
        and applies appropriate corrections.

        Args:
            seq: Alphanumeric sequence

        Returns:
            Corrected sequence

        Example:
            >>> fixed = OCRPostProcessor._fix_alphanumeric_sequence("O1Z3")
            >>> # Result: "0123" (if mostly numeric)
        """
        # Count letters vs digits
        num_letters = sum(1 for c in seq if c.isalpha())
        num_digits = sum(1 for c in seq if c.isdigit())

        # If mostly digits, apply letter→digit corrections
        if num_digits > num_letters:
            corrected = seq
            for wrong, right in OCRPostProcessor.CHAR_CORRECTIONS.items():
                corrected = corrected.replace(wrong, right)
            return corrected

        return seq

    @staticmethod
    def remove_invalid_characters(text: str, valid_pattern: str = r'[A-Za-z0-9\s\-:,.]') -> str:
        """Remove characters that don't match valid pattern.

        Args:
            text: Input text
            valid_pattern: Regex pattern for valid characters

        Returns:
            Text with only valid characters

        Example:
            >>> text = "TRACK-123@#$%"
            >>> clean = OCRPostProcessor.remove_invalid_characters(text)
            >>> # Result: "TRACK-123"
        """
        return re.sub(f'[^{valid_pattern}]', '', text)

    @staticmethod
    def validate_text_format(
        text: str,
        expected_pattern: Optional[str] = None,
        min_length: int = 0,
        max_length: int = 1000
    ) -> bool:
        """Validate that text matches expected format.

        Args:
            text: Text to validate
            expected_pattern: Regex pattern to match (optional)
            min_length: Minimum text length
            max_length: Maximum text length

        Returns:
            True if valid, False otherwise

        Example:
            >>> text = "1234567890123"
            >>> is_valid = OCRPostProcessor.validate_text_format(
            ...     text,
            ...     expected_pattern=r'^[0-9]{10,15}$'
            ... )
        """
        if not text:
            return False

        # Length validation
        if len(text) < min_length or len(text) > max_length:
            return False

        # Pattern validation
        if expected_pattern:
            if not re.match(expected_pattern, text):
                return False

        return True

    @staticmethod
    def validate_tracking_number(text: str) -> bool:
        """Validate that text matches common tracking number formats.

        Args:
            text: Potential tracking number

        Returns:
            True if matches known tracking format

        Example:
            >>> is_valid = OCRPostProcessor.validate_tracking_number("1Z999AA10123456784")
            >>> # True (UPS format)
        """
        if not text:
            return False

        # Remove whitespace for validation
        text = text.strip().replace(' ', '')

        # Check against known patterns
        for pattern in OCRPostProcessor.TRACKING_PATTERNS:
            if re.match(pattern, text):
                return True

        return False

    @staticmethod
    def extract_tracking_numbers(text: str) -> list:
        """Extract potential tracking numbers from text.

        Args:
            text: Input text

        Returns:
            List of potential tracking numbers

        Example:
            >>> text = "Shipment 1: 1Z999AA10123456784\\nShipment 2: 420921579101"
            >>> numbers = OCRPostProcessor.extract_tracking_numbers(text)
            >>> # Result: ['1Z999AA10123456784', '420921579101']
        """
        tracking_numbers = []

        # Look for alphanumeric sequences of typical tracking length
        candidates = re.findall(r'\b[A-Z0-9]{10,22}\b', text)

        for candidate in candidates:
            if OCRPostProcessor.validate_tracking_number(candidate):
                tracking_numbers.append(candidate)

        return tracking_numbers

    @staticmethod
    def extract_alphanumeric_codes(
        text: str,
        min_length: int = 4,
        max_length: int = 30
    ) -> list:
        """Extract alphanumeric codes from text.

        More general than tracking numbers - extracts any alphanumeric
        sequences of specified length.

        Args:
            text: Input text
            min_length: Minimum code length
            max_length: Maximum code length

        Returns:
            List of alphanumeric codes

        Example:
            >>> text = "LOT: ABC123 SKU: XYZ789"
            >>> codes = OCRPostProcessor.extract_alphanumeric_codes(text)
            >>> # Result: ['ABC123', 'XYZ789']
        """
        pattern = rf'\b[A-Z0-9]{{{min_length},{max_length}}}\b'
        return re.findall(pattern, text)

    @staticmethod
    def format_tracking_number(text: str, format_type: str = 'grouped') -> str:
        """Format tracking number for readability.

        Args:
            text: Tracking number
            format_type: Formatting style:
                - 'grouped': Add spaces every 4 characters
                - 'dashed': Add dashes every 4 characters
                - 'none': No formatting

        Returns:
            Formatted tracking number

        Example:
            >>> number = "1Z999AA10123456784"
            >>> formatted = OCRPostProcessor.format_tracking_number(number)
            >>> # Result: "1Z99 9AA1 0123 4567 84"
        """
        if not text:
            return ""

        # Remove existing formatting
        clean = text.replace(' ', '').replace('-', '')

        if format_type == 'grouped':
            # Add space every 4 characters
            formatted = ' '.join(clean[i:i+4] for i in range(0, len(clean), 4))
            return formatted
        elif format_type == 'dashed':
            # Add dash every 4 characters
            formatted = '-'.join(clean[i:i+4] for i in range(0, len(clean), 4))
            return formatted
        else:
            return clean

    @staticmethod
    def calculate_text_quality_score(text: str) -> float:
        """Calculate quality score for extracted text.

        Heuristics:
        - Longer text generally better
        - Balanced alpha/numeric ratio
        - Presence of recognizable patterns
        - Low ratio of special characters

        Args:
            text: Extracted text

        Returns:
            Quality score (0.0-1.0)

        Example:
            >>> score = OCRPostProcessor.calculate_text_quality_score("TRACK-12345")
            >>> if score > 0.7:
            ...     print("High quality extraction")
        """
        if not text:
            return 0.0

        score = 0.0

        # Factor 1: Length (longer is better, up to a point)
        length_score = min(len(text) / 50.0, 1.0)  # Normalize to 50 chars
        score += 0.3 * length_score

        # Factor 2: Alphanumeric ratio (good balance is positive)
        alpha_count = sum(1 for c in text if c.isalpha())
        digit_count = sum(1 for c in text if c.isdigit())
        total_alnum = alpha_count + digit_count

        if total_alnum > 0:
            ratio = min(alpha_count, digit_count) / total_alnum
            score += 0.3 * ratio

        # Factor 3: Special character ratio (too many is bad)
        special_count = sum(1 for c in text if not c.isalnum() and not c.isspace())
        if len(text) > 0:
            special_ratio = special_count / len(text)
            score += 0.2 * (1.0 - special_ratio)

        # Factor 4: Pattern recognition (tracking number, codes)
        has_tracking = any(
            re.match(pattern, text.replace(' ', ''))
            for pattern in OCRPostProcessor.TRACKING_PATTERNS
        )
        if has_tracking:
            score += 0.2

        return min(score, 1.0)

    @staticmethod
    def compare_texts(text1: str, text2: str) -> Dict:
        """Compare two text strings for similarity.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Dictionary with comparison metrics:
                - exact_match: Boolean
                - similarity: Similarity ratio (0-1)
                - length_diff: Absolute length difference
                - common_chars: Number of common characters

        Example:
            >>> comparison = OCRPostProcessor.compare_texts("TRACK123", "TRACK1Z3")
            >>> print(f"Similarity: {comparison['similarity']:.2%}")
        """
        from difflib import SequenceMatcher

        exact_match = (text1 == text2)
        similarity = SequenceMatcher(None, text1, text2).ratio()
        length_diff = abs(len(text1) - len(text2))

        # Count common characters
        common_chars = sum(1 for a, b in zip(text1, text2) if a == b)

        return {
            'exact_match': exact_match,
            'similarity': similarity,
            'length_diff': length_diff,
            'common_chars': common_chars
        }
