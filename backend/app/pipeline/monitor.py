"""Monitoring and logging for pipeline health and performance."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from ..cv_models import ExtractedShippingData


class PipelineMonitor:
    """Monitor pipeline health and performance."""

    def __init__(self, log_dir: str = "logs", log_level: str = "INFO"):
        """Initialize pipeline monitor.

        Args:
            log_dir: Directory for log files
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.setup_logging(log_level)

        self.metrics = {
            'start_time': datetime.now(),
            'frames_processed': 0,
            'pallets_tracked': 0,
            'documents_detected': 0,
            'extractions': [],
            'errors': [],
            'warnings': [],
        }

        self.logger = logging.getLogger(__name__)

    def setup_logging(self, log_level: str = "INFO"):
        """Configure structured logging.

        Args:
            log_level: Logging level
        """
        # Create log filename with timestamp
        log_filename = (
            f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
        log_path = self.log_dir / log_filename

        # Configure logging
        logging.basicConfig(
            level=getattr(logging, log_level.upper(), logging.INFO),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_path),
                logging.StreamHandler(),
            ],
        )

        logging.info(f"Pipeline monitor initialized. Logging to: {log_path}")

    def log_frame_processing(self, frame_number: int, result: Dict):
        """Log frame processing details.

        Args:
            frame_number: Frame number
            result: Processing result dictionary
        """
        self.metrics['frames_processed'] += 1

        # Log significant events
        if result.get('ocr_processed', False):
            self.logger.debug(f"Frame {frame_number}: OCR processed")

        completed = result.get('completed_extractions', [])
        if completed:
            self.logger.info(
                f"Frame {frame_number}: {len(completed)} extractions completed"
            )

    def log_extraction(
        self, extraction: ExtractedShippingData, confidence: float, route: str
    ):
        """Log completed extraction.

        Args:
            extraction: Extracted shipping data
            confidence: Overall confidence score
            route: Routing decision (AUTO_ACCEPT, NEEDS_REVIEW, AUTO_REJECT)
        """
        self.metrics['extractions'].append(
            {
                'timestamp': datetime.now(),
                'track_id': extraction.track_id,
                'document_type': extraction.document_type,
                'confidence': confidence,
                'route': route,
            }
        )

        self.logger.info(
            f"Extraction completed - Track {extraction.track_id}: "
            f"{extraction.document_type} (confidence: {confidence:.2f}, route: {route})"
        )

    def log_error(self, error: Exception, context: Dict):
        """Log errors with context.

        Args:
            error: Exception that occurred
            context: Context information (frame number, component, etc.)
        """
        error_record = {
            'timestamp': datetime.now(),
            'error': str(error),
            'type': type(error).__name__,
            'context': context,
        }

        self.metrics['errors'].append(error_record)

        self.logger.error(
            f"Error in {context.get('component', 'unknown')}: {error}",
            exc_info=True,
        )

    def log_warning(self, message: str, context: Optional[Dict] = None):
        """Log warning message.

        Args:
            message: Warning message
            context: Optional context information
        """
        warning_record = {
            'timestamp': datetime.now(),
            'message': message,
            'context': context or {},
        }

        self.metrics['warnings'].append(warning_record)
        self.logger.warning(message)

    def generate_performance_report(self) -> Dict:
        """Generate performance report.

        Returns:
            Dictionary with performance metrics
        """
        elapsed_time = (datetime.now() - self.metrics['start_time']).total_seconds()

        # Calculate rates
        frames_per_second = (
            self.metrics['frames_processed'] / elapsed_time if elapsed_time > 0 else 0
        )

        # Count extractions by route
        routes = {'AUTO_ACCEPT': 0, 'NEEDS_REVIEW': 0, 'AUTO_REJECT': 0}
        for extraction in self.metrics['extractions']:
            route = extraction.get('route', 'UNKNOWN')
            if route in routes:
                routes[route] += 1

        # Calculate average confidence
        confidences = [e['confidence'] for e in self.metrics['extractions']]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0

        report = {
            'processing_time': elapsed_time,
            'frames_processed': self.metrics['frames_processed'],
            'frames_per_second': frames_per_second,
            'total_extractions': len(self.metrics['extractions']),
            'extraction_routes': routes,
            'avg_confidence': avg_confidence,
            'errors_count': len(self.metrics['errors']),
            'warnings_count': len(self.metrics['warnings']),
        }

        return report

    def save_detailed_report(self, output_path: Optional[str] = None):
        """Save detailed report to JSON file.

        Args:
            output_path: Output file path (default: logs/report_<timestamp>.json)
        """
        if output_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = str(self.log_dir / f"report_{timestamp}.json")

        report = {
            'performance': self.generate_performance_report(),
            'metrics': {
                'start_time': self.metrics['start_time'].isoformat(),
                'end_time': datetime.now().isoformat(),
                'frames_processed': self.metrics['frames_processed'],
                'extractions': [
                    {
                        **e,
                        'timestamp': e['timestamp'].isoformat(),
                    }
                    for e in self.metrics['extractions']
                ],
                'errors': [
                    {
                        **e,
                        'timestamp': e['timestamp'].isoformat(),
                    }
                    for e in self.metrics['errors']
                ],
                'warnings': [
                    {
                        **w,
                        'timestamp': w['timestamp'].isoformat(),
                    }
                    for w in self.metrics['warnings']
                ],
            },
        }

        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)

        self.logger.info(f"Detailed report saved to: {output_path}")

    def print_summary(self):
        """Print summary of processing to console."""
        report = self.generate_performance_report()

        print("\n" + "=" * 70)
        print("PALLETTRACK PIPELINE SUMMARY")
        print("=" * 70)
        print(f"Processing time: {report['processing_time']:.2f}s")
        print(f"Frames processed: {report['frames_processed']}")
        print(f"Average FPS: {report['frames_per_second']:.2f}")
        print(f"\nTotal extractions: {report['total_extractions']}")
        print(f"  Auto-accepted: {report['extraction_routes']['AUTO_ACCEPT']}")
        print(f"  Needs review: {report['extraction_routes']['NEEDS_REVIEW']}")
        print(f"  Auto-rejected: {report['extraction_routes']['AUTO_REJECT']}")
        print(f"\nAverage confidence: {report['avg_confidence']:.2f}")
        print(f"Errors: {report['errors_count']}")
        print(f"Warnings: {report['warnings_count']}")
        print("=" * 70 + "\n")
