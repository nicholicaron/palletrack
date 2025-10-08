"""Item list extraction from packing lists.

This module extracts line items from packing list documents,
detecting tabular data and parsing into structured items.
"""

import re
from typing import Dict, List, Optional


class ItemListExtractor:
    """Extract line items from packing lists."""

    # Common column headers
    ITEM_HEADERS = ['item', 'description', 'product', 'part', 'part number', 'part#']
    QUANTITY_HEADERS = ['qty', 'quantity', 'qnty', 'count']
    SKU_HEADERS = ['sku', 'item#', 'item number', 'product code']
    WEIGHT_HEADERS = ['weight', 'wt', 'unit weight']

    def __init__(self, max_items: int = 100):
        """Initialize item extractor.

        Args:
            max_items: Maximum number of items to extract (safety limit)
        """
        self.max_items = max_items

    def extract_items(self, text: str) -> List[Dict[str, any]]:
        """Extract item list from packing list.

        Look for tabular data with columns like:
        - Item/Description/Part Number
        - Quantity
        - Weight
        - SKU

        Args:
            text: OCR text from packing list

        Returns:
            List of dicts:
            [
                {'description': 'Widget A', 'quantity': 10, 'sku': 'WID-001'},
                ...
            ]
        """
        # First, try to detect table structure
        table_info = self.detect_table_structure(text)

        if table_info and table_info.get('has_table'):
            # Extract items from detected table
            return self._extract_from_table(text, table_info)
        else:
            # Fallback: try to extract items by pattern matching
            return self._extract_by_patterns(text)

    def detect_table_structure(self, text: str) -> Optional[Dict]:
        """Detect if text contains tabular data.

        Identify column headers and data rows.

        Args:
            text: Input text

        Returns:
            Dictionary with table structure info:
            {
                'has_table': bool,
                'headers': List[str],
                'header_line': int,
                'data_start_line': int
            }
        """
        lines = text.split('\n')

        # Look for header line containing item-related keywords
        for i, line in enumerate(lines):
            line_lower = line.lower()

            # Check for multiple column headers on same line
            header_count = 0
            found_headers = []

            # Check for item headers
            for header in self.ITEM_HEADERS:
                if header in line_lower:
                    header_count += 1
                    found_headers.append(header)

            # Check for quantity headers
            for header in self.QUANTITY_HEADERS:
                if header in line_lower:
                    header_count += 1
                    found_headers.append(header)

            # If we found 2+ headers, likely a table header row
            if header_count >= 2:
                return {
                    'has_table': True,
                    'headers': found_headers,
                    'header_line': i,
                    'data_start_line': i + 1
                }

        return {
            'has_table': False,
            'headers': [],
            'header_line': -1,
            'data_start_line': -1
        }

    def _extract_from_table(self, text: str, table_info: Dict) -> List[Dict[str, any]]:
        """Extract items from detected table structure.

        Args:
            text: Input text
            table_info: Table structure information

        Returns:
            List of item dictionaries
        """
        lines = text.split('\n')
        start_line = table_info['data_start_line']

        if start_line < 0 or start_line >= len(lines):
            return []

        items = []

        # Process lines after header
        for line in lines[start_line:]:
            if not line.strip():
                continue

            # Try to extract item from line
            item = self._parse_item_line(line)
            if item:
                items.append(item)

            # Safety limit
            if len(items) >= self.max_items:
                break

        return items

    def _parse_item_line(self, line: str) -> Optional[Dict[str, any]]:
        """Parse a single line as an item entry.

        Args:
            line: Text line to parse

        Returns:
            Item dictionary or None if not a valid item line
        """
        # Skip lines that are clearly not items
        line_lower = line.lower()
        if any(keyword in line_lower for keyword in ['total', 'subtotal', 'page', 'continued']):
            return None

        # Try to extract components
        item = {}

        # Extract quantity (numbers at start or after QTY keyword)
        qty_match = re.search(r'(?:^|\bqty\s*:?\s*)(\d+)', line, re.IGNORECASE)
        if qty_match:
            item['quantity'] = int(qty_match.group(1))

        # Extract SKU/part number (alphanumeric with dashes)
        sku_match = re.search(r'\b([A-Z0-9]+-[A-Z0-9-]+)\b', line)
        if sku_match:
            item['sku'] = sku_match.group(1)

        # Extract weight
        weight_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:lbs?|kg)', line, re.IGNORECASE)
        if weight_match:
            item['weight'] = float(weight_match.group(1))

        # Extract description (remaining text)
        # Remove quantity, SKU, weight from line
        description = line
        if qty_match:
            description = description.replace(qty_match.group(0), '')
        if sku_match:
            description = description.replace(sku_match.group(0), '')
        if weight_match:
            description = description.replace(weight_match.group(0), '')

        # Clean up description
        description = re.sub(r'\s+', ' ', description).strip()
        if description:
            item['description'] = description

        # Only return if we found at least a description or SKU
        if item.get('description') or item.get('sku'):
            return item

        return None

    def _extract_by_patterns(self, text: str) -> List[Dict[str, any]]:
        """Extract items by pattern matching (fallback method).

        Args:
            text: Input text

        Returns:
            List of item dictionaries
        """
        items = []
        lines = text.split('\n')

        for line in lines:
            # Look for lines with quantity patterns like "10x Widget" or "Qty: 5 - Product"
            patterns = [
                r'(\d+)\s*x\s+(.+)',  # "10x Widget"
                r'qty[:\s]*(\d+)[:\s-]+(.+)',  # "Qty: 10 - Widget"
                r'(\d+)\s+(.+?)\s+\d+(?:\.\d+)?\s*(?:lbs?|kg)',  # "10 Widget 5.5 lbs"
            ]

            for pattern in patterns:
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    quantity = int(match.group(1))
                    description = match.group(2).strip()

                    item = {
                        'quantity': quantity,
                        'description': description
                    }

                    # Try to extract SKU from description
                    sku_match = re.search(r'\b([A-Z0-9]+-[A-Z0-9-]+)\b', description)
                    if sku_match:
                        item['sku'] = sku_match.group(1)

                    items.append(item)
                    break  # Only match one pattern per line

            # Safety limit
            if len(items) >= self.max_items:
                break

        return items

    def calculate_total_quantity(self, items: List[Dict[str, any]]) -> int:
        """Calculate total quantity across all items.

        Args:
            items: List of item dictionaries

        Returns:
            Total quantity
        """
        total = 0
        for item in items:
            qty = item.get('quantity', 0)
            if isinstance(qty, (int, float)):
                total += qty
        return total

    def calculate_total_weight(self, items: List[Dict[str, any]]) -> float:
        """Calculate total weight across all items.

        Args:
            items: List of item dictionaries

        Returns:
            Total weight
        """
        total = 0.0
        for item in items:
            weight = item.get('weight', 0.0)
            if isinstance(weight, (int, float)):
                total += weight
        return total

    @staticmethod
    def validate_items(items: List[Dict[str, any]]) -> bool:
        """Validate that extracted items are reasonable.

        Args:
            items: List of item dictionaries

        Returns:
            True if items appear valid
        """
        if not items:
            return False

        # Check that at least some items have descriptions
        items_with_desc = sum(1 for item in items if item.get('description'))
        if items_with_desc < len(items) * 0.5:  # At least 50% should have descriptions
            return False

        # Check that quantities are reasonable
        for item in items:
            qty = item.get('quantity', 0)
            if qty < 0 or qty > 10000:  # Unreasonable quantity
                return False

        return True

    def extract_carton_info(self, text: str) -> Optional[Dict[str, any]]:
        """Extract carton/package information.

        Args:
            text: Input text

        Returns:
            Dictionary with carton info:
            {
                'carton_count': int,
                'total_pieces': int
            }
        """
        result = {}

        # Extract carton count
        carton_patterns = [
            r'(?:total\s+)?cartons?[:\s]*(\d+)',
            r'(\d+)\s+cartons?',
            r'(?:number\s+of\s+)?packages?[:\s]*(\d+)',
        ]

        for pattern in carton_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                result['carton_count'] = int(match.group(1))
                break

        # Extract total pieces
        pieces_patterns = [
            r'total\s+pieces?[:\s]*(\d+)',
            r'total\s+qty[:\s]*(\d+)',
        ]

        for pattern in pieces_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                result['total_pieces'] = int(match.group(1))
                break

        return result if result else None


class PackingListSummaryExtractor:
    """Extract summary information from packing lists."""

    @staticmethod
    def extract_po_number(text: str) -> Optional[str]:
        """Extract PO number from packing list.

        Args:
            text: Input text

        Returns:
            PO number or None
        """
        patterns = [
            r'(?:PO|P\.O\.|purchase\s*order)[:\s#]*([A-Z0-9-]{3,20})',
            r'PO[#:\s]*([A-Z0-9-]+)',
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)

        return None

    @staticmethod
    def extract_invoice_number(text: str) -> Optional[str]:
        """Extract invoice number.

        Args:
            text: Input text

        Returns:
            Invoice number or None
        """
        patterns = [
            r'invoice[:\s#]*([A-Z0-9-]{3,20})',
            r'inv[#:\s]*([A-Z0-9-]+)',
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)

        return None

    @staticmethod
    def extract_order_date(text: str) -> Optional[str]:
        """Extract order date.

        Args:
            text: Input text

        Returns:
            Date string or None
        """
        patterns = [
            r'(?:order\s+date|date)[:\s]*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
            r'(?:shipped\s+date)[:\s]*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)

        return None
