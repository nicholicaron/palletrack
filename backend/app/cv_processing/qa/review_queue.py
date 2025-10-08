"""Review queue management for items needing human verification."""

import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

from ...cv_models import ExtractedShippingData


class ReviewQueueManager:
    """Manage items needing human review."""

    def __init__(self, config: Dict):
        """Initialize review queue manager.

        Args:
            config: Configuration dictionary
        """
        confidence_config = config.get('confidence', {})
        review_config = config.get('review_queue', {})

        # Load thresholds
        self.thresholds = {
            'auto_accept': confidence_config.get('auto_accept', 0.85),
            'needs_review': confidence_config.get('needs_review', 0.60),
            'auto_reject': confidence_config.get('auto_reject', 0.40)
        }

        # Review queue settings
        self.priority_order = review_config.get('priority_order', 'confidence')
        self.max_queue_size = review_config.get('max_queue_size', 1000)
        self.save_frames = review_config.get('save_frames', True)
        self.frame_save_path = review_config.get('frame_save_path', 'review_frames')

        # Create frame save directory if needed
        if self.save_frames:
            Path(self.frame_save_path).mkdir(parents=True, exist_ok=True)

        # Storage for different routes
        self.review_queue: List[Dict] = []
        self.auto_accepted: List[Dict] = []
        self.auto_rejected: List[Dict] = []

    def route_extraction(
        self,
        extracted_data: ExtractedShippingData,
        confidence_breakdown: Dict
    ) -> str:
        """Route extraction based on confidence.

        Routes:
        - AUTO_ACCEPT (≥0.85): Ready for WMS integration
        - NEEDS_REVIEW (0.60-0.85): Human verification needed
        - AUTO_REJECT (<0.60): Flag for re-scan or manual entry

        Args:
            extracted_data: Extracted shipping data
            confidence_breakdown: Detailed confidence scores

        Returns:
            Route name: 'AUTO_ACCEPT', 'NEEDS_REVIEW', or 'AUTO_REJECT'
        """
        overall_confidence = confidence_breakdown.get('overall_confidence', 0.0)

        if overall_confidence >= self.thresholds['auto_accept']:
            route = 'AUTO_ACCEPT'
            self.auto_accepted.append({
                'extracted_data': extracted_data,
                'confidence_breakdown': confidence_breakdown,
                'timestamp': datetime.now()
            })
        elif overall_confidence >= self.thresholds['needs_review']:
            route = 'NEEDS_REVIEW'
        else:
            route = 'AUTO_REJECT'
            self.auto_rejected.append({
                'extracted_data': extracted_data,
                'confidence_breakdown': confidence_breakdown,
                'timestamp': datetime.now()
            })

        return route

    def add_to_review_queue(
        self,
        extracted_data: ExtractedShippingData,
        confidence_breakdown: Dict,
        best_frame: Optional[np.ndarray] = None
    ) -> None:
        """Add item to review queue with context for operator.

        Store:
        - Extracted data
        - Confidence scores
        - Best frame image for visual verification
        - Specific issues/warnings

        Args:
            extracted_data: Extracted shipping data
            confidence_breakdown: Detailed confidence scores
            best_frame: Best quality frame image for verification (BGR format)
        """
        # Check queue size limit
        if len(self.review_queue) >= self.max_queue_size:
            # Remove oldest item if at capacity
            self.review_queue.pop(0)

        # Generate operator context (warnings, suggestions)
        operator_context = self.generate_operator_context(
            extracted_data,
            confidence_breakdown
        )

        # Save best frame if provided
        frame_path = None
        if best_frame is not None and self.save_frames:
            frame_filename = f"track_{extracted_data.track_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            frame_path = os.path.join(self.frame_save_path, frame_filename)
            cv2.imwrite(frame_path, best_frame)

        # Add to queue
        review_item = {
            'extracted_data': extracted_data,
            'confidence_breakdown': confidence_breakdown,
            'operator_context': operator_context,
            'frame_path': frame_path,
            'timestamp': datetime.now(),
            'track_id': extracted_data.track_id
        }

        self.review_queue.append(review_item)

    def get_review_queue(self, priority_order: Optional[str] = None) -> List[Dict]:
        """Get items needing review.

        Priority orders:
        - 'confidence': Lowest confidence first (hardest cases)
        - 'timestamp': Oldest first (FIFO)
        - 'importance': Based on document type or other factors

        Args:
            priority_order: Sorting order (uses config default if None)

        Returns:
            List of review items sorted by priority
        """
        if priority_order is None:
            priority_order = self.priority_order

        # Sort based on priority order
        if priority_order == 'confidence':
            # Lowest confidence first (hardest cases need attention first)
            sorted_queue = sorted(
                self.review_queue,
                key=lambda x: x['confidence_breakdown']['overall_confidence']
            )
        elif priority_order == 'timestamp':
            # Oldest first (FIFO)
            sorted_queue = sorted(
                self.review_queue,
                key=lambda x: x['timestamp']
            )
        elif priority_order == 'importance':
            # Sort by document type importance (BOL > PACKING_LIST > SHIPPING_LABEL)
            importance_order = {
                'BOL': 3,
                'PACKING_LIST': 2,
                'SHIPPING_LABEL': 1,
                'UNKNOWN': 0
            }
            sorted_queue = sorted(
                self.review_queue,
                key=lambda x: importance_order.get(
                    x['extracted_data'].document_type,
                    0
                ),
                reverse=True
            )
        else:
            # Default: return as-is
            sorted_queue = self.review_queue.copy()

        return sorted_queue

    def generate_operator_context(
        self,
        extracted_data: ExtractedShippingData,
        confidence_breakdown: Dict
    ) -> Dict:
        """Generate helpful context for human operator.

        Args:
            extracted_data: Extracted shipping data
            confidence_breakdown: Detailed confidence scores

        Returns:
            Dictionary with warnings, suggestions, and confidence details
        """
        warnings = []
        suggestions = []

        # Check individual confidence factors
        if confidence_breakdown.get('ocr_confidence', 1.0) < 0.7:
            warnings.append('Low OCR confidence on text recognition')
            suggestions.append('Verify text carefully against source image')

        if confidence_breakdown.get('detection_confidence', 1.0) < 0.7:
            warnings.append('Low detection confidence for pallet/document')
            suggestions.append('Check if document is visible and well-positioned')

        if confidence_breakdown.get('cross_frame_consistency', 1.0) < 0.6:
            warnings.append('Inconsistent readings across frames')
            suggestions.append('Compare values from multiple frames if available')

        if confidence_breakdown.get('data_validation', 1.0) < 0.6:
            warnings.append('Data validation issues detected')
            suggestions.append('Check field formats and values for correctness')

        # Check specific field issues
        if extracted_data.tracking_number:
            from .validators import DataValidator
            if not DataValidator.validate_tracking_number(extracted_data.tracking_number):
                warnings.append(f'Invalid tracking number format: {extracted_data.tracking_number}')
                suggestions.append('Verify tracking number characters (O vs 0, I vs 1)')

        if extracted_data.weight is not None:
            from .validators import DataValidator
            if not DataValidator.validate_weight(extracted_data.weight):
                warnings.append(f'Weight outside normal range: {extracted_data.weight} lbs')
                suggestions.append('Verify weight unit (lbs vs kg) and value')

        if extracted_data.destination_zip:
            from .validators import DataValidator
            if not DataValidator.validate_zip_code(extracted_data.destination_zip):
                warnings.append(f'Invalid ZIP code format: {extracted_data.destination_zip}')
                suggestions.append('Check ZIP code digits')

        # Check field completeness
        completeness = confidence_breakdown.get('field_completeness', 0.0)
        if completeness < 0.7:
            warnings.append('Missing required fields')
            suggestions.append('Check document for missing information')

        return {
            'warnings': warnings,
            'suggestions': suggestions,
            'confidence_details': confidence_breakdown
        }

    def get_queue_stats(self) -> Dict:
        """Get statistics about the review queue.

        Returns:
            Dictionary with queue statistics
        """
        return {
            'review_queue_size': len(self.review_queue),
            'auto_accepted_count': len(self.auto_accepted),
            'auto_rejected_count': len(self.auto_rejected),
            'total_processed': len(self.auto_accepted) + len(self.auto_rejected) + len(self.review_queue),
            'avg_confidence_in_queue': (
                np.mean([
                    item['confidence_breakdown']['overall_confidence']
                    for item in self.review_queue
                ]) if self.review_queue else 0.0
            )
        }

    def clear_queue(self) -> None:
        """Clear all items from the review queue."""
        self.review_queue.clear()

    def remove_from_queue(self, track_id: int) -> Optional[Dict]:
        """Remove a specific item from the review queue.

        Args:
            track_id: Track ID of the item to remove

        Returns:
            The removed item, or None if not found
        """
        for i, item in enumerate(self.review_queue):
            if item['track_id'] == track_id:
                return self.review_queue.pop(i)
        return None
