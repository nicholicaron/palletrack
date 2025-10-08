"""Video stream processor for pipeline."""

import logging
import time
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np

from .main_pipeline import PalletScannerPipeline
from .monitor import PipelineMonitor
from .visualizer import FrameAnnotator, FPSTracker

logger = logging.getLogger(__name__)


class VideoStreamProcessor:
    """Process video files or live streams through the pipeline."""

    def __init__(self, pipeline: PalletScannerPipeline, monitor: Optional[PipelineMonitor] = None):
        """Initialize video stream processor.

        Args:
            pipeline: PalletScannerPipeline instance
            monitor: Optional PipelineMonitor for logging
        """
        self.pipeline = pipeline
        self.monitor = monitor
        self.fps_tracker = FPSTracker()

    def process_video_file(
        self,
        video_path: str,
        output_video_path: Optional[str] = None,
        visualize: bool = True,
        display: bool = False,
        skip_frames: int = 0,
    ) -> Dict:
        """Process video file through pipeline.

        Args:
            video_path: Path to input video file
            output_video_path: Optional path to save annotated video
            visualize: Whether to draw detections/tracks on frames
            display: Whether to display video during processing (for debugging)
            skip_frames: Process every Nth frame (0 = process all frames)

        Returns:
            Processing summary with metrics
        """
        # Open video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        logger.info(f"Processing video: {video_path}")
        logger.info(f"FPS: {fps}, Total frames: {total_frames}, Resolution: {width}x{height}")

        # Create video writer if output path specified
        writer = None
        if output_video_path:
            writer = self._create_video_writer(
                output_video_path, width, height, fps
            )
            logger.info(f"Saving output video to: {output_video_path}")

        # Processing loop
        frame_number = 0
        frames_processed = 0
        start_time = time.time()

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Skip frames if requested
                if skip_frames > 0 and frame_number % (skip_frames + 1) != 0:
                    frame_number += 1
                    continue

                # Calculate timestamp
                timestamp = frame_number / fps if fps > 0 else frame_number

                # Process frame through pipeline
                result = self.pipeline.process_frame(frame, frame_number, timestamp)

                # Update FPS tracker
                self.fps_tracker.update()

                # Visualize if requested
                if visualize or writer or display:
                    annotated_frame = self._annotate_frame(frame, result)
                else:
                    annotated_frame = frame

                # Write to output video
                if writer:
                    writer.write(annotated_frame)

                # Display if requested
                if display:
                    cv2.imshow('PalletTrack Pipeline', annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        logger.info("Processing interrupted by user")
                        break

                # Log progress
                if frames_processed % 100 == 0:
                    self._log_progress(frames_processed, total_frames)

                frames_processed += 1
                frame_number += 1

        finally:
            # Cleanup
            cap.release()
            if writer:
                writer.release()
            if display:
                cv2.destroyAllWindows()

        # Finalize pipeline (process remaining tracks)
        logger.info("Finalizing pipeline...")
        self.pipeline.finalize()

        # Generate summary
        elapsed_time = time.time() - start_time
        summary = self._generate_summary(frames_processed, total_frames, elapsed_time)

        logger.info("Processing complete!")
        logger.info(f"Processed {frames_processed} frames in {elapsed_time:.2f}s")
        logger.info(f"Average FPS: {summary['avg_fps']:.2f}")

        return summary

    def process_live_stream(
        self,
        stream_url: str,
        duration_seconds: Optional[int] = None,
        output_video_path: Optional[str] = None,
        visualize: bool = True,
    ) -> Dict:
        """Process live video stream (RTSP, webcam, etc.).

        Args:
            stream_url: Stream URL (e.g., 'rtsp://...', '0' for webcam)
            duration_seconds: Optional duration limit in seconds
            output_video_path: Optional path to save annotated video
            visualize: Whether to draw detections/tracks on frames

        Returns:
            Processing summary with metrics
        """
        # Open stream
        # Try to convert to int for webcam index
        try:
            source = int(stream_url)
        except ValueError:
            source = stream_url

        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise ValueError(f"Could not open stream: {stream_url}")

        # Get stream properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # If FPS is 0 or invalid, use default
        if fps <= 0:
            fps = 30.0
            logger.warning(f"Invalid FPS from stream, using default: {fps}")

        logger.info(f"Processing live stream: {stream_url}")
        logger.info(f"FPS: {fps}, Resolution: {width}x{height}")

        # Create video writer if output path specified
        writer = None
        if output_video_path:
            writer = self._create_video_writer(
                output_video_path, width, height, fps
            )
            logger.info(f"Saving output video to: {output_video_path}")

        # Processing loop
        frame_number = 0
        start_time = time.time()

        try:
            while True:
                # Check duration limit
                if duration_seconds is not None:
                    elapsed = time.time() - start_time
                    if elapsed >= duration_seconds:
                        logger.info(f"Duration limit reached: {duration_seconds}s")
                        break

                ret, frame = cap.read()
                if not ret:
                    logger.warning("Failed to read frame from stream")
                    break

                # Calculate timestamp
                timestamp = time.time() - start_time

                # Process frame through pipeline
                result = self.pipeline.process_frame(frame, frame_number, timestamp)

                # Update FPS tracker
                self.fps_tracker.update()

                # Visualize if requested
                if visualize or writer:
                    annotated_frame = self._annotate_frame(frame, result)
                else:
                    annotated_frame = frame

                # Write to output video
                if writer:
                    writer.write(annotated_frame)

                # Display (always show for live streams)
                cv2.imshow('PalletTrack Live Stream', annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    logger.info("Processing interrupted by user")
                    break

                # Log progress every 100 frames
                if frame_number % 100 == 0:
                    logger.info(
                        f"Processed {frame_number} frames, "
                        f"FPS: {self.fps_tracker.get_fps():.2f}"
                    )

                frame_number += 1

        finally:
            # Cleanup
            cap.release()
            if writer:
                writer.release()
            cv2.destroyAllWindows()

        # Finalize pipeline
        logger.info("Finalizing pipeline...")
        self.pipeline.finalize()

        # Generate summary
        elapsed_time = time.time() - start_time
        summary = self._generate_summary(frame_number, frame_number, elapsed_time)

        logger.info("Processing complete!")

        return summary

    def _create_video_writer(
        self, output_path: str, width: int, height: int, fps: float
    ) -> cv2.VideoWriter:
        """Create video writer for output.

        Args:
            output_path: Output video path
            width: Frame width
            height: Frame height
            fps: Frames per second

        Returns:
            VideoWriter instance
        """
        # Ensure output directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # Use mp4v codec for MP4 files
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        writer = cv2.VideoWriter(
            output_path,
            fourcc,
            fps,
            (width, height)
        )

        if not writer.isOpened():
            raise ValueError(f"Could not create video writer: {output_path}")

        return writer

    def _annotate_frame(self, frame: np.ndarray, result: Dict) -> np.ndarray:
        """Draw detections, tracks, and info on frame.

        Args:
            frame: Input frame
            result: Processing result from pipeline

        Returns:
            Annotated frame
        """
        annotated = FrameAnnotator.annotate_complete_frame(
            frame=frame,
            pallet_detections=result.get('pallet_detections', []),
            active_tracks=result.get('active_tracks', {}),
            document_detections=result.get('document_detections', []),
            ocr_processed=result.get('ocr_processed', False),
        )

        # Add info panel
        annotated = FrameAnnotator.add_info_panel(
            frame=annotated,
            fps=self.fps_tracker.get_fps(),
            active_tracks=len(result.get('active_tracks', {})),
            processed_extractions=len(self.pipeline.completed_extractions),
            frame_number=result.get('frame_number', 0),
        )

        return annotated

    def _log_progress(self, current_frame: int, total_frames: int):
        """Log processing progress.

        Args:
            current_frame: Current frame number
            total_frames: Total frames in video
        """
        if total_frames > 0:
            progress_pct = (current_frame / total_frames) * 100
            logger.info(
                f"Progress: {current_frame}/{total_frames} ({progress_pct:.1f}%) - "
                f"FPS: {self.fps_tracker.get_fps():.2f}"
            )
        else:
            logger.info(
                f"Processed {current_frame} frames - "
                f"FPS: {self.fps_tracker.get_fps():.2f}"
            )

    def _generate_summary(
        self, frames_processed: int, total_frames: int, elapsed_time: float
    ) -> Dict:
        """Generate processing summary.

        Args:
            frames_processed: Number of frames processed
            total_frames: Total frames in video
            elapsed_time: Total processing time in seconds

        Returns:
            Summary dictionary
        """
        # Get pipeline statistics
        pipeline_stats = self.pipeline.get_statistics()

        # Calculate processing metrics
        avg_fps = frames_processed / elapsed_time if elapsed_time > 0 else 0

        summary = {
            'frames_processed': frames_processed,
            'total_frames': total_frames,
            'elapsed_time': elapsed_time,
            'avg_fps': avg_fps,
            'pipeline_stats': pipeline_stats,
        }

        return summary
