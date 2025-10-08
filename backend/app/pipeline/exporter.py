"""Export results in various formats for integration."""

import csv
import json
from pathlib import Path
from typing import Dict, List

from ..cv_models import ExtractedShippingData


class ResultsExporter:
    """Export extraction results in various formats."""

    @staticmethod
    def export_to_json(
        extractions: List[ExtractedShippingData], output_path: str
    ) -> None:
        """Export to JSON format.

        Args:
            extractions: List of extracted shipping data
            output_path: Output file path
        """
        # Ensure output directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # Convert to dictionaries
        data = [extraction.dict() for extraction in extractions]

        # Write to JSON file
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)

        print(f"Exported {len(extractions)} extractions to JSON: {output_path}")

    @staticmethod
    def export_to_csv(
        extractions: List[ExtractedShippingData], output_path: str
    ) -> None:
        """Export to CSV format.

        Args:
            extractions: List of extracted shipping data
            output_path: Output file path
        """
        if not extractions:
            print("No extractions to export")
            return

        # Ensure output directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # Define CSV columns (flattened structure)
        fieldnames = [
            'track_id',
            'document_type',
            'tracking_number',
            'po_number',
            'weight',
            'destination_address',
            'destination_city',
            'destination_state',
            'destination_zip',
            'origin_address',
            'origin_city',
            'origin_state',
            'origin_zip',
            'carrier',
            'service_type',
            'declared_value',
            'confidence_score',
            'needs_review',
        ]

        # Write CSV file
        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for extraction in extractions:
                # Flatten extraction data
                row = {
                    'track_id': extraction.track_id,
                    'document_type': extraction.document_type,
                    'tracking_number': extraction.tracking_number or '',
                    'po_number': extraction.po_number or '',
                    'weight': extraction.weight if extraction.weight is not None else '',
                    'destination_address': extraction.destination_address or '',
                    'destination_city': extraction.destination_city or '',
                    'destination_state': extraction.destination_state or '',
                    'destination_zip': extraction.destination_zip or '',
                    'origin_address': extraction.origin_address or '',
                    'origin_city': extraction.origin_city or '',
                    'origin_state': extraction.origin_state or '',
                    'origin_zip': extraction.origin_zip or '',
                    'carrier': extraction.carrier or '',
                    'service_type': extraction.service_type or '',
                    'declared_value': (
                        extraction.declared_value
                        if extraction.declared_value is not None
                        else ''
                    ),
                    'confidence_score': (
                        extraction.confidence_score
                        if extraction.confidence_score is not None
                        else ''
                    ),
                    'needs_review': extraction.needs_review,
                }
                writer.writerow(row)

        print(f"Exported {len(extractions)} extractions to CSV: {output_path}")

    @staticmethod
    def export_review_queue(review_items: List[Dict], output_path: str) -> None:
        """Export review queue to JSON.

        Args:
            review_items: List of review queue items
            output_path: Output file path
        """
        # Ensure output directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # Convert ExtractedShippingData to dict if needed
        processed_items = []
        for item in review_items:
            processed_item = item.copy()

            # Convert ExtractedShippingData to dict
            if isinstance(item.get('extracted_data'), ExtractedShippingData):
                processed_item['extracted_data'] = item['extracted_data'].dict()

            # Convert datetime to string
            if 'timestamp' in processed_item:
                processed_item['timestamp'] = str(processed_item['timestamp'])

            # Remove frame_path (just keep the path string)
            if 'frame_path' in processed_item and processed_item['frame_path']:
                processed_item['frame_path'] = str(processed_item['frame_path'])

            processed_items.append(processed_item)

        # Write to JSON file
        with open(output_path, 'w') as f:
            json.dump(processed_items, f, indent=2, default=str)

        print(f"Exported {len(review_items)} review items to: {output_path}")

    @staticmethod
    def export_to_database(
        extractions: List[ExtractedShippingData], db_connection_string: str
    ) -> None:
        """Export to database (for WMS integration).

        NOTE: Stub implementation - will be completed with specific database schema.

        Args:
            extractions: List of extracted shipping data
            db_connection_string: Database connection string
        """
        # TODO: Implement database export once schema is defined
        # Example implementation outline:
        #
        # from sqlalchemy import create_engine
        # engine = create_engine(db_connection_string)
        #
        # with engine.connect() as conn:
        #     for extraction in extractions:
        #         # Insert into database
        #         conn.execute(
        #             "INSERT INTO pallet_extractions (...) VALUES (...)",
        #             extraction.dict()
        #         )

        print(
            f"Database export not yet implemented. "
            f"Would export {len(extractions)} extractions to {db_connection_string}"
        )

    @staticmethod
    def create_wms_payload(extraction: ExtractedShippingData) -> Dict:
        """Format extraction for WMS API integration.

        NOTE: Stub implementation - will be completed with specific WMS API specs.

        Returns standardized payload for warehouse management system.

        Args:
            extraction: Extracted shipping data

        Returns:
            WMS-formatted payload dictionary
        """
        # TODO: Implement WMS-specific formatting once API spec is available
        # Example payload structure:
        payload = {
            'pallet_id': extraction.track_id,
            'document_type': extraction.document_type,
            'shipment_info': {
                'tracking_number': extraction.tracking_number,
                'po_number': extraction.po_number,
                'carrier': extraction.carrier,
                'service_type': extraction.service_type,
            },
            'weight': {
                'value': extraction.weight,
                'unit': 'lbs',
            },
            'destination': {
                'address': extraction.destination_address,
                'city': extraction.destination_city,
                'state': extraction.destination_state,
                'zip': extraction.destination_zip,
            },
            'origin': {
                'address': extraction.origin_address,
                'city': extraction.origin_city,
                'state': extraction.origin_state,
                'zip': extraction.origin_zip,
            },
            'metadata': {
                'confidence_score': extraction.confidence_score,
                'needs_review': extraction.needs_review,
                'scan_timestamp': str(extraction.timestamp) if extraction.timestamp else None,
            },
        }

        return payload

    @staticmethod
    def batch_export_wms_payloads(
        extractions: List[ExtractedShippingData], output_path: str
    ) -> None:
        """Export batch of WMS payloads to JSON.

        Args:
            extractions: List of extracted shipping data
            output_path: Output file path
        """
        # Ensure output directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # Create payloads
        payloads = [
            ResultsExporter.create_wms_payload(extraction)
            for extraction in extractions
        ]

        # Write to JSON file
        with open(output_path, 'w') as f:
            json.dump(payloads, f, indent=2, default=str)

        print(f"Exported {len(payloads)} WMS payloads to: {output_path}")
