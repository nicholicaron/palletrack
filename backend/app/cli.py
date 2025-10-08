"""Command-line interface for PalletTrack pipeline."""

import argparse
import logging
import sys
from pathlib import Path

from .pipeline import (
    PalletScannerPipeline,
    PipelineMonitor,
    ResultsExporter,
    VideoStreamProcessor,
)

logger = logging.getLogger(__name__)


class PalletScannerCLI:
    """Command-line interface for the PalletTrack scanner."""

    @staticmethod
    def main():
        """Main CLI entry point."""
        parser = argparse.ArgumentParser(
            description="PalletTrack - Automated Pallet Label Scanner",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  # Process video file with visualization
  python -m app.cli video.mp4 --output-video annotated.mp4

  # Process video and export results to JSON
  python -m app.cli video.mp4 --output-json results.json

  # Process live stream from webcam
  python -m app.cli --live-stream 0

  # Process RTSP stream
  python -m app.cli --live-stream rtsp://camera-ip/stream

  # Process without visualization (faster)
  python -m app.cli video.mp4 --no-viz --output-json results.json
            """,
        )

        # Input source
        parser.add_argument(
            'video_path',
            type=str,
            nargs='?',
            help='Path to input video file (required unless --live-stream is used)',
        )

        # Configuration
        parser.add_argument(
            '--config',
            type=str,
            default='app/cv_config/config.yaml',
            help='Path to configuration file (default: app/cv_config/config.yaml)',
        )

        # Output options
        parser.add_argument(
            '--output-video',
            type=str,
            help='Path to save annotated output video',
        )

        parser.add_argument(
            '--output-json',
            type=str,
            help='Path to save extracted data as JSON',
        )

        parser.add_argument(
            '--output-csv',
            type=str,
            help='Path to save extracted data as CSV',
        )

        parser.add_argument(
            '--export-review-queue',
            type=str,
            help='Path to export review queue to JSON',
        )

        parser.add_argument(
            '--export-wms',
            type=str,
            help='Path to export WMS-formatted payloads to JSON',
        )

        # Processing options
        parser.add_argument(
            '--no-viz',
            action='store_true',
            help='Disable visualization (faster processing)',
        )

        parser.add_argument(
            '--display',
            action='store_true',
            help='Display video during processing (for debugging)',
        )

        parser.add_argument(
            '--skip-frames',
            type=int,
            default=0,
            help='Process every Nth frame (0 = process all frames)',
        )

        # Live stream options
        parser.add_argument(
            '--live-stream',
            type=str,
            help='Live stream URL (RTSP, webcam index, etc.)',
        )

        parser.add_argument(
            '--duration',
            type=int,
            help='Duration to process live stream (seconds)',
        )

        # Logging options
        parser.add_argument(
            '--log-dir',
            type=str,
            default='logs',
            help='Directory for log files (default: logs)',
        )

        parser.add_argument(
            '--log-level',
            type=str,
            default='INFO',
            choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
            help='Logging level (default: INFO)',
        )

        # Statistics and reporting
        parser.add_argument(
            '--stats',
            action='store_true',
            help='Print detailed statistics after processing',
        )

        parser.add_argument(
            '--save-report',
            type=str,
            help='Save detailed processing report to JSON file',
        )

        args = parser.parse_args()

        # Validate input source
        if not args.video_path and not args.live_stream:
            parser.error("Either video_path or --live-stream must be provided")

        if args.video_path and args.live_stream:
            parser.error("Cannot use both video_path and --live-stream")

        # Run pipeline
        try:
            PalletScannerCLI.run_pipeline(args)
        except KeyboardInterrupt:
            logger.info("\nProcessing interrupted by user")
            sys.exit(0)
        except Exception as e:
            logger.error(f"Error during processing: {e}", exc_info=True)
            sys.exit(1)

    @staticmethod
    def run_pipeline(args):
        """Run the processing pipeline.

        Args:
            args: Parsed command-line arguments
        """
        # Initialize monitor
        monitor = PipelineMonitor(log_dir=args.log_dir, log_level=args.log_level)
        monitor.logger.info("=" * 70)
        monitor.logger.info("PALLETTRACK - Automated Pallet Label Scanner")
        monitor.logger.info("=" * 70)

        # Initialize pipeline
        monitor.logger.info(f"Loading configuration from: {args.config}")
        pipeline = PalletScannerPipeline(args.config)

        # Initialize video processor
        processor = VideoStreamProcessor(pipeline, monitor)

        # Process video or stream
        if args.live_stream:
            monitor.logger.info(f"Processing live stream: {args.live_stream}")
            summary = processor.process_live_stream(
                stream_url=args.live_stream,
                duration_seconds=args.duration,
                output_video_path=args.output_video,
                visualize=not args.no_viz,
            )
        else:
            monitor.logger.info(f"Processing video: {args.video_path}")
            summary = processor.process_video_file(
                video_path=args.video_path,
                output_video_path=args.output_video,
                visualize=not args.no_viz,
                display=args.display,
                skip_frames=args.skip_frames,
            )

        # Export results
        extractions = pipeline.completed_extractions

        if args.output_json:
            ResultsExporter.export_to_json(extractions, args.output_json)

        if args.output_csv:
            ResultsExporter.export_to_csv(extractions, args.output_csv)

        if args.export_review_queue:
            review_queue = pipeline.get_review_queue()
            ResultsExporter.export_review_queue(
                review_queue, args.export_review_queue
            )

        if args.export_wms:
            ResultsExporter.batch_export_wms_payloads(extractions, args.export_wms)

        # Save detailed report
        if args.save_report:
            monitor.save_detailed_report(args.save_report)

        # Print statistics
        if args.stats:
            PalletScannerCLI._print_detailed_statistics(summary, pipeline, monitor)
        else:
            # Print basic summary
            monitor.print_summary()

        monitor.logger.info("Processing complete!")

    @staticmethod
    def _print_detailed_statistics(summary: dict, pipeline, monitor):
        """Print detailed processing statistics.

        Args:
            summary: Processing summary
            pipeline: PalletScannerPipeline instance
            monitor: PipelineMonitor instance
        """
        print("\n" + "=" * 70)
        print("DETAILED PROCESSING STATISTICS")
        print("=" * 70)

        # Processing metrics
        print("\n[Processing Metrics]")
        print(f"Total frames: {summary['total_frames']}")
        print(f"Frames processed: {summary['frames_processed']}")
        print(f"Processing time: {summary['elapsed_time']:.2f}s")
        print(f"Average FPS: {summary['avg_fps']:.2f}")

        # Pipeline statistics
        pipeline_stats = summary['pipeline_stats']
        print("\n[Detection & Tracking]")
        print(f"Pallets tracked: {pipeline_stats['pallets_tracked']}")
        print(f"Documents detected: {pipeline_stats['documents_detected']}")
        print(f"OCR runs: {pipeline_stats['ocr_runs']}")

        # Extraction statistics
        print("\n[Data Extraction]")
        print(f"Total extractions: {pipeline_stats['extractions_completed']}")

        queue_stats = pipeline_stats['queue_stats']
        print(f"  Auto-accepted: {queue_stats['auto_accepted_count']}")
        print(f"  Needs review: {queue_stats['review_queue_size']}")
        print(f"  Auto-rejected: {queue_stats['auto_rejected_count']}")

        if queue_stats['review_queue_size'] > 0:
            print(
                f"  Avg confidence in queue: {queue_stats['avg_confidence_in_queue']:.2f}"
            )

        # Quality metrics
        quality_report = pipeline_stats['quality_report']
        if quality_report['summary']['total_processed'] > 0:
            print("\n[Quality Metrics]")
            summary_data = quality_report['summary']
            print(
                f"Auto-acceptance rate: {summary_data['auto_acceptance_rate']:.1%}"
            )
            print(f"Review rate: {summary_data['review_rate']:.1%}")
            print(f"Rejection rate: {summary_data['rejection_rate']:.1%}")

            performance = quality_report['performance']
            print(f"Average confidence: {performance['avg_confidence']:.2f}")
            print(
                f"Confidence range: {performance['min_confidence']:.2f} - {performance['max_confidence']:.2f}"
            )

        # Monitor statistics
        monitor_report = monitor.generate_performance_report()
        if monitor_report['errors_count'] > 0 or monitor_report['warnings_count'] > 0:
            print("\n[Errors & Warnings]")
            print(f"Errors: {monitor_report['errors_count']}")
            print(f"Warnings: {monitor_report['warnings_count']}")

        print("=" * 70 + "\n")


if __name__ == "__main__":
    PalletScannerCLI.main()
