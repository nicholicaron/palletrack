"""Address extraction and parsing from OCR text.

This module extracts origin and destination addresses from shipping documents,
handling various formats and structures.
"""

import re
from typing import Dict, List, Optional, Tuple


class AddressExtractor:
    """Extract and structure address information from shipping documents."""

    # US state abbreviations for validation
    US_STATES = {
        'AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA',
        'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD',
        'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ',
        'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC',
        'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY',
        'DC'
    }

    # Keywords that indicate address sections
    DESTINATION_KEYWORDS = [
        'ship to', 'deliver to', 'destination', 'consignee',
        'delivery address', 'shipping address', 'recipient'
    ]

    ORIGIN_KEYWORDS = [
        'ship from', 'origin', 'shipper', 'sender',
        'return address', 'from address'
    ]

    def __init__(self):
        """Initialize address extractor."""
        pass

    def extract_addresses(self, text: str) -> Dict[str, Optional[str]]:
        """Extract origin and destination addresses.

        Strategy:
        1. Look for "ship to", "deliver to", "destination" keywords
        2. Look for "ship from", "origin", "shipper" keywords
        3. Extract multi-line address blocks after keywords
        4. Parse into components (street, city, state, zip)

        Args:
            text: OCR text to extract addresses from

        Returns:
            Dictionary with address components:
            {
                'destination_address': "123 Main St, Boston, MA 02101",
                'destination_city': "Boston",
                'destination_state': "MA",
                'destination_zip': "02101",
                'origin_address': "...",
                'origin_city': "...",
                'origin_state': "...",
                'origin_zip': "..."
            }
        """
        result = {
            'destination_address': None,
            'destination_city': None,
            'destination_state': None,
            'destination_zip': None,
            'origin_address': None,
            'origin_city': None,
            'origin_state': None,
            'origin_zip': None,
        }

        # Extract destination address
        dest_block = self._extract_address_block(text, self.DESTINATION_KEYWORDS)
        if dest_block:
            dest_components = self.parse_address_components(dest_block)
            result['destination_address'] = dest_components.get('full_address')
            result['destination_city'] = dest_components.get('city')
            result['destination_state'] = dest_components.get('state')
            result['destination_zip'] = dest_components.get('zip_code')

        # Extract origin address
        origin_block = self._extract_address_block(text, self.ORIGIN_KEYWORDS)
        if origin_block:
            origin_components = self.parse_address_components(origin_block)
            result['origin_address'] = origin_components.get('full_address')
            result['origin_city'] = origin_components.get('city')
            result['origin_state'] = origin_components.get('state')
            result['origin_zip'] = origin_components.get('zip_code')

        return result

    def _extract_address_block(self, text: str, keywords: List[str]) -> Optional[str]:
        """Extract address block following specific keywords.

        Args:
            text: Input text
            keywords: List of keywords to search for

        Returns:
            Address block text or None
        """
        # Try each keyword
        for keyword in keywords:
            # Case-insensitive search for keyword
            pattern = rf'(?i){re.escape(keyword)}[:\s]*\n?(.*?)(?:\n\n|\n(?=[A-Z][a-z]+:)|$)'
            match = re.search(pattern, text, re.DOTALL)

            if match:
                address_block = match.group(1).strip()
                # Extract up to 5 lines (typical address length)
                lines = address_block.split('\n')[:5]
                return '\n'.join(lines).strip()

        # If no keyword match, try to find address-like patterns
        return self._extract_address_by_pattern(text)

    def _extract_address_by_pattern(self, text: str) -> Optional[str]:
        """Extract address by pattern matching (city, state, zip).

        Args:
            text: Input text

        Returns:
            Address block or None
        """
        # Look for pattern: City, ST ZIP
        pattern = r'([A-Z][a-z\s]+),\s*([A-Z]{2})\s+(\d{5}(?:-\d{4})?)'
        matches = re.finditer(pattern, text)

        for match in matches:
            # Get context around the match (2 lines before)
            start = max(0, match.start() - 100)
            end = match.end()
            context = text[start:end]

            # Split into lines and take last 3-4 lines
            lines = context.split('\n')
            address_lines = lines[-4:] if len(lines) >= 4 else lines

            return '\n'.join(line.strip() for line in address_lines if line.strip())

        return None

    @staticmethod
    def parse_address_components(address_block: str) -> Dict[str, Optional[str]]:
        """Parse address string into components.

        Handle variations:
        - Multi-line vs single-line
        - With/without country
        - Various formatting

        Args:
            address_block: Address text (possibly multi-line)

        Returns:
            Dictionary with parsed components:
            {
                'full_address': str,
                'street': str,
                'city': str,
                'state': str,
                'zip_code': str,
                'country': str
            }
        """
        if not address_block:
            return {
                'full_address': None,
                'street': None,
                'city': None,
                'state': None,
                'zip_code': None,
                'country': None
            }

        result = {
            'full_address': address_block.replace('\n', ', '),
            'street': None,
            'city': None,
            'state': None,
            'zip_code': None,
            'country': None
        }

        # Extract ZIP code
        zip_match = re.search(r'\b(\d{5}(?:-\d{4})?)\b', address_block)
        if zip_match:
            result['zip_code'] = zip_match.group(1)

        # Extract state (2-letter abbreviation)
        state_match = re.search(r'\b([A-Z]{2})\b', address_block)
        if state_match:
            state = state_match.group(1)
            if state in AddressExtractor.US_STATES:
                result['state'] = state

        # Extract city (word(s) before state)
        if result['state']:
            city_pattern = rf'([A-Z][a-z\s]+?),?\s+{result["state"]}'
            city_match = re.search(city_pattern, address_block)
            if city_match:
                result['city'] = city_match.group(1).strip()

        # Extract street address (lines before city/state/zip)
        lines = address_block.split('\n')
        if len(lines) > 1:
            # Assume first line(s) are street address
            street_lines = []
            for line in lines:
                # Stop when we hit city/state/zip line
                if result['city'] and result['city'] in line:
                    break
                if result['state'] and result['state'] in line:
                    break
                if result['zip_code'] and result['zip_code'] in line:
                    break
                street_lines.append(line.strip())

            if street_lines:
                result['street'] = ' '.join(street_lines)

        return result

    @staticmethod
    def validate_address(address_components: Dict[str, Optional[str]]) -> bool:
        """Validate that address has required components.

        Args:
            address_components: Parsed address components

        Returns:
            True if address has city and state (minimum required)
        """
        return bool(
            address_components.get('city') and
            address_components.get('state')
        )

    def extract_all_zip_codes(self, text: str) -> List[str]:
        """Extract all ZIP codes from text.

        Useful for finding multiple addresses or validating extraction.

        Args:
            text: Input text

        Returns:
            List of ZIP codes found
        """
        pattern = r'\b(\d{5}(?:-\d{4})?)\b'
        matches = re.findall(pattern, text)
        return matches

    def extract_all_cities_states(self, text: str) -> List[Tuple[str, str]]:
        """Extract all city, state pairs from text.

        Args:
            text: Input text

        Returns:
            List of (city, state) tuples
        """
        # Pattern: City, ST
        pattern = r'([A-Z][a-z\s]+),\s*([A-Z]{2})\b'
        matches = re.findall(pattern, text)

        # Filter to only valid US states
        valid_matches = [
            (city.strip(), state)
            for city, state in matches
            if state in self.US_STATES
        ]

        return valid_matches

    @staticmethod
    def format_address(components: Dict[str, Optional[str]]) -> str:
        """Format address components into single-line string.

        Args:
            components: Address components dictionary

        Returns:
            Formatted address string
        """
        parts = []

        if components.get('street'):
            parts.append(components['street'])

        if components.get('city'):
            parts.append(components['city'])

        # State and ZIP on same part
        state_zip = []
        if components.get('state'):
            state_zip.append(components['state'])
        if components.get('zip_code'):
            state_zip.append(components['zip_code'])

        if state_zip:
            parts.append(' '.join(state_zip))

        if components.get('country'):
            parts.append(components['country'])

        return ', '.join(parts)


class CompanyNameExtractor:
    """Extract company names from addresses."""

    # Common business suffixes
    BUSINESS_SUFFIXES = [
        'inc', 'llc', 'corp', 'corporation', 'company', 'co',
        'ltd', 'limited', 'enterprises', 'industries', 'services'
    ]

    @classmethod
    def extract_company_name(cls, address_block: str) -> Optional[str]:
        """Extract company name from address block.

        Typically the first line of an address.

        Args:
            address_block: Address text

        Returns:
            Company name or None
        """
        if not address_block:
            return None

        lines = address_block.split('\n')
        if not lines:
            return None

        first_line = lines[0].strip()

        # Check if first line looks like a company name
        # (has business suffix or is all caps)
        lower_line = first_line.lower()

        for suffix in cls.BUSINESS_SUFFIXES:
            if suffix in lower_line:
                return first_line

        # Check if mostly uppercase (common for company names)
        if first_line.isupper() and len(first_line) > 3:
            return first_line

        # If it doesn't look like a street address, assume it's a name
        if not re.search(r'\d+', first_line):  # No numbers
            return first_line

        return None
