"""Track quality metrics and system performance over time."""

from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np


class QualityMetricsTracker:
    """Track system performance over time."""

    def __init__(self):
        """Initialize quality metrics tracker."""
        self.metrics = {
            'total_processed': 0,
            'auto_accepted': 0,
            'needs_review': 0,
            'auto_rejected': 0,
            'avg_confidence': [],
            'processing_time': [],
            'operator_corrections': []
        }

        # Track corrections by field for pattern analysis
        self.field_corrections = defaultdict(list)

        # Track errors by type
        self.error_types = defaultdict(int)

        # Store detailed extraction records
        self.extraction_history: List[Dict] = []

    def record_extraction(
        self,
        route: str,
        confidence: float,
        processing_time: float,
        extracted_data: Optional[Dict] = None
    ) -> None:
        """Record metrics for completed extraction.

        Args:
            route: Routing decision ('AUTO_ACCEPT', 'NEEDS_REVIEW', 'AUTO_REJECT')
            confidence: Overall confidence score (0-1)
            processing_time: Time taken to process in seconds
            extracted_data: Optional extracted data for detailed tracking
        """
        self.metrics['total_processed'] += 1
        self.metrics['avg_confidence'].append(confidence)
        self.metrics['processing_time'].append(processing_time)

        # Update route counters
        if route == 'AUTO_ACCEPT':
            self.metrics['auto_accepted'] += 1
        elif route == 'NEEDS_REVIEW':
            self.metrics['needs_review'] += 1
        elif route == 'AUTO_REJECT':
            self.metrics['auto_rejected'] += 1

        # Store detailed record
        record = {
            'timestamp': datetime.now(),
            'route': route,
            'confidence': confidence,
            'processing_time': processing_time,
            'extracted_data': extracted_data
        }
        self.extraction_history.append(record)

    def record_operator_correction(
        self,
        track_id: int,
        field: str,
        original: str,
        corrected: str,
        confidence: Optional[float] = None
    ) -> None:
        """Record human corrections for model improvement.

        This data can be used to:
        - Retrain OCR models
        - Improve extraction patterns
        - Identify systematic errors

        Args:
            track_id: Track ID of the corrected extraction
            field: Field name that was corrected
            original: Original (incorrect) value
            corrected: Corrected value
            confidence: Confidence score of the original extraction
        """
        correction = {
            'timestamp': datetime.now(),
            'track_id': track_id,
            'field': field,
            'original': original,
            'corrected': corrected,
            'confidence': confidence
        }

        self.metrics['operator_corrections'].append(correction)
        self.field_corrections[field].append(correction)

        # Analyze error type
        error_type = self._classify_error(original, corrected)
        self.error_types[error_type] += 1

    def _classify_error(self, original: str, corrected: str) -> str:
        """Classify the type of error based on original and corrected values.

        Args:
            original: Original incorrect value
            corrected: Corrected value

        Returns:
            Error type classification
        """
        if not original:
            return 'missing_field'

        # Check for common OCR character confusion
        ocr_confusions = {
            ('0', 'O'), ('O', '0'),
            ('1', 'I'), ('I', '1'), ('1', 'l'),
            ('5', 'S'), ('S', '5'),
            ('8', 'B'), ('B', '8')
        }

        # Compare character by character
        if len(original) == len(corrected):
            diff_count = sum(1 for a, b in zip(original, corrected) if a != b)

            if diff_count == 1:
                # Single character difference - likely OCR confusion
                for i, (a, b) in enumerate(zip(original, corrected)):
                    if a != b and (a, b) in ocr_confusions:
                        return 'ocr_character_confusion'
                return 'single_character_error'

            elif diff_count <= 3:
                return 'multiple_character_errors'

        # Check for missing/extra characters
        if len(original) < len(corrected):
            return 'missing_characters'
        elif len(original) > len(corrected):
            return 'extra_characters'

        # Complete mismatch
        return 'complete_mismatch'

    def generate_quality_report(self) -> Dict:
        """Generate quality metrics report.

        Metrics:
        - Auto-acceptance rate
        - Average confidence scores
        - Common failure modes
        - Processing throughput

        Returns:
            Dictionary with quality metrics and statistics
        """
        total = self.metrics['total_processed']

        if total == 0:
            return {
                'total_processed': 0,
                'auto_acceptance_rate': 0.0,
                'review_rate': 0.0,
                'rejection_rate': 0.0,
                'avg_confidence': 0.0,
                'avg_processing_time': 0.0,
                'correction_rate': 0.0,
                'common_errors': {},
                'field_accuracy': {}
            }

        # Calculate rates
        auto_acceptance_rate = self.metrics['auto_accepted'] / total
        review_rate = self.metrics['needs_review'] / total
        rejection_rate = self.metrics['auto_rejected'] / total

        # Calculate averages
        avg_confidence = (
            np.mean(self.metrics['avg_confidence'])
            if self.metrics['avg_confidence'] else 0.0
        )
        avg_processing_time = (
            np.mean(self.metrics['processing_time'])
            if self.metrics['processing_time'] else 0.0
        )

        # Calculate correction rate
        correction_count = len(self.metrics['operator_corrections'])
        correction_rate = correction_count / total if total > 0 else 0.0

        # Get most common error types
        common_errors = dict(
            sorted(
                self.error_types.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]  # Top 5 error types
        )

        # Calculate field-level accuracy
        field_accuracy = {}
        for field, corrections in self.field_corrections.items():
            # Estimate accuracy based on corrections
            # If we processed N items and had C corrections for a field,
            # accuracy ≈ (N - C) / N
            field_error_rate = len(corrections) / total
            field_accuracy[field] = {
                'accuracy': 1.0 - field_error_rate,
                'error_count': len(corrections),
                'error_rate': field_error_rate
            }

        # Build comprehensive report
        report = {
            'summary': {
                'total_processed': total,
                'auto_acceptance_rate': auto_acceptance_rate,
                'review_rate': review_rate,
                'rejection_rate': rejection_rate,
                'correction_rate': correction_rate
            },
            'performance': {
                'avg_confidence': float(avg_confidence),
                'avg_processing_time': float(avg_processing_time),
                'min_confidence': float(np.min(self.metrics['avg_confidence'])) if self.metrics['avg_confidence'] else 0.0,
                'max_confidence': float(np.max(self.metrics['avg_confidence'])) if self.metrics['avg_confidence'] else 0.0
            },
            'errors': {
                'common_error_types': common_errors,
                'field_accuracy': field_accuracy,
                'total_corrections': correction_count
            },
            'breakdown': {
                'auto_accepted': self.metrics['auto_accepted'],
                'needs_review': self.metrics['needs_review'],
                'auto_rejected': self.metrics['auto_rejected']
            }
        }

        return report

    def get_recent_extractions(self, limit: int = 100) -> List[Dict]:
        """Get most recent extraction records.

        Args:
            limit: Maximum number of records to return

        Returns:
            List of recent extraction records
        """
        return self.extraction_history[-limit:]

    def get_field_corrections(self, field: str) -> List[Dict]:
        """Get all corrections for a specific field.

        Args:
            field: Field name

        Returns:
            List of corrections for the specified field
        """
        return self.field_corrections.get(field, [])

    def get_error_patterns(self) -> Dict:
        """Analyze and return common error patterns.

        Returns:
            Dictionary with error pattern analysis
        """
        patterns = {
            'character_confusions': defaultdict(int),
            'field_specific_errors': defaultdict(list),
            'confidence_correlation': []
        }

        # Analyze character-level confusions
        for correction in self.metrics['operator_corrections']:
            original = correction['original']
            corrected = correction['corrected']

            if len(original) == len(corrected):
                for o_char, c_char in zip(original, corrected):
                    if o_char != c_char:
                        confusion_pair = f"{o_char}→{c_char}"
                        patterns['character_confusions'][confusion_pair] += 1

            # Track field-specific error patterns
            field = correction['field']
            error_type = self._classify_error(original, corrected)
            patterns['field_specific_errors'][field].append({
                'error_type': error_type,
                'example_original': original,
                'example_corrected': corrected
            })

            # Correlate errors with confidence scores
            if correction.get('confidence') is not None:
                patterns['confidence_correlation'].append({
                    'confidence': correction['confidence'],
                    'had_error': True
                })

        # Convert character confusions to sorted list
        patterns['top_character_confusions'] = dict(
            sorted(
                patterns['character_confusions'].items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]  # Top 10 confusions
        )

        return patterns

    def reset_metrics(self) -> None:
        """Reset all metrics to initial state."""
        self.metrics = {
            'total_processed': 0,
            'auto_accepted': 0,
            'needs_review': 0,
            'auto_rejected': 0,
            'avg_confidence': [],
            'processing_time': [],
            'operator_corrections': []
        }
        self.field_corrections.clear()
        self.error_types.clear()
        self.extraction_history.clear()

    def export_metrics(self) -> Dict:
        """Export all metrics for external analysis or storage.

        Returns:
            Complete metrics data structure
        """
        return {
            'metrics': self.metrics,
            'field_corrections': dict(self.field_corrections),
            'error_types': dict(self.error_types),
            'extraction_history': self.extraction_history,
            'quality_report': self.generate_quality_report()
        }
