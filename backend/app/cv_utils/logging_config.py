"""Logging infrastructure for PalleTrack CV pipeline."""

import logging
import os
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

from app.cv_config import get_config


class PerformanceMetricsFormatter(logging.Formatter):
    """Custom formatter that includes performance metrics in log messages."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record with optional performance metrics.

        Args:
            record: Log record to format

        Returns:
            Formatted log message
        """
        # Add performance metrics if available
        if hasattr(record, 'fps'):
            record.msg = f"[FPS: {record.fps:.2f}] {record.msg}"
        if hasattr(record, 'latency_ms'):
            record.msg = f"[Latency: {record.latency_ms:.2f}ms] {record.msg}"

        return super().format(record)


class TrackLogger:
    """Per-track logger for debugging specific pallet tracks.

    Attributes:
        track_id: ID of the track being logged
        logger: Logger instance for this track
        log_dir: Directory for track-specific logs
    """

    def __init__(self, track_id: int, log_dir: str):
        """Initialize track logger.

        Args:
            track_id: ID of the track
            log_dir: Directory for log files
        """
        self.track_id = track_id
        self.log_dir = Path(log_dir) / "tracks"
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Create track-specific logger
        logger_name = f"palletrack.track.{track_id}"
        self.logger = logging.getLogger(logger_name)

        # Only add handler if not already present
        if not self.logger.handlers:
            self.logger.setLevel(logging.DEBUG)

            # Create rotating file handler for this track
            log_file = self.log_dir / f"track_{track_id}.log"
            handler = RotatingFileHandler(
                log_file,
                maxBytes=10 * 1024 * 1024,  # 10 MB
                backupCount=3
            )

            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def debug(self, msg: str, **kwargs) -> None:
        """Log debug message for this track."""
        self.logger.debug(msg, **kwargs)

    def info(self, msg: str, **kwargs) -> None:
        """Log info message for this track."""
        self.logger.info(msg, **kwargs)

    def warning(self, msg: str, **kwargs) -> None:
        """Log warning message for this track."""
        self.logger.warning(msg, **kwargs)

    def error(self, msg: str, **kwargs) -> None:
        """Log error message for this track."""
        self.logger.error(msg, **kwargs)

    def log_detection(self, frame_number: int, bbox: tuple, confidence: float) -> None:
        """Log a detection for this track.

        Args:
            frame_number: Frame number
            bbox: Bounding box coordinates (x1, y1, x2, y2)
            confidence: Detection confidence
        """
        self.info(
            f"Frame {frame_number}: Detection at {bbox} "
            f"(confidence: {confidence:.3f})"
        )

    def log_ocr(self, frame_number: int, text: str, confidence: float) -> None:
        """Log OCR result for this track.

        Args:
            frame_number: Frame number
            text: Extracted text
            confidence: OCR confidence
        """
        self.info(
            f"Frame {frame_number}: OCR result '{text}' "
            f"(confidence: {confidence:.3f})"
        )


def setup_logging(
    log_dir: Optional[str] = None,
    log_level: Optional[str] = None,
    enable_frame_logging: Optional[bool] = None
) -> None:
    """Setup logging infrastructure for the application.

    Args:
        log_dir: Directory for log files (overrides config)
        log_level: Logging level (overrides config)
        enable_frame_logging: Enable frame logging (overrides config)
    """
    # Get config values
    config = get_config()

    if log_dir is None:
        log_dir = config.logging.log_dir
    if log_level is None:
        log_level = config.logging.level
    if enable_frame_logging is None:
        enable_frame_logging = config.logging.enable_frame_logging

    # Create log directory
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    # Create subdirectories
    (log_path / "tracks").mkdir(exist_ok=True)
    if enable_frame_logging:
        (log_path / "frames").mkdir(exist_ok=True)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level))

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Console handler with performance metrics formatter
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level))
    console_formatter = PerformanceMetricsFormatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # Main application log file with rotation
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    main_log_file = log_path / f"palletrack_{timestamp}.log"
    file_handler = RotatingFileHandler(
        main_log_file,
        maxBytes=50 * 1024 * 1024,  # 50 MB
        backupCount=10
    )
    file_handler.setLevel(logging.DEBUG)  # Always log DEBUG to file
    file_formatter = PerformanceMetricsFormatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)

    # Performance metrics log file
    perf_log_file = log_path / f"performance_{timestamp}.log"
    perf_handler = RotatingFileHandler(
        perf_log_file,
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5
    )
    perf_handler.setLevel(logging.INFO)
    perf_formatter = logging.Formatter(
        '%(asctime)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    perf_handler.setFormatter(perf_formatter)

    # Create performance logger
    perf_logger = logging.getLogger('palletrack.performance')
    perf_logger.addHandler(perf_handler)
    perf_logger.setLevel(logging.INFO)

    # Log startup message
    root_logger.info("="*60)
    root_logger.info(f"PalleTrack CV Pipeline Started")
    root_logger.info(f"Log Level: {log_level}")
    root_logger.info(f"Log Directory: {log_path}")
    root_logger.info(f"Frame Logging: {'Enabled' if enable_frame_logging else 'Disabled'}")
    root_logger.info("="*60)


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with the given name.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Logger instance
    """
    return logging.getLogger(name)


def log_performance_metrics(
    logger: logging.Logger,
    fps: float,
    latency_ms: float,
    frame_number: int,
    active_tracks: int
) -> None:
    """Log performance metrics.

    Args:
        logger: Logger instance
        fps: Frames per second
        latency_ms: Processing latency in milliseconds
        frame_number: Current frame number
        active_tracks: Number of active tracks
    """
    perf_logger = logging.getLogger('palletrack.performance')
    perf_logger.info(
        f"Frame {frame_number}: FPS={fps:.2f}, "
        f"Latency={latency_ms:.2f}ms, "
        f"Active Tracks={active_tracks}"
    )


def save_debug_frame(
    frame,
    frame_number: int,
    track_id: Optional[int] = None,
    suffix: str = ""
) -> Optional[str]:
    """Save a frame for debugging purposes.

    Args:
        frame: Frame image (numpy array)
        frame_number: Frame number
        track_id: Optional track ID
        suffix: Optional suffix for filename

    Returns:
        Path to saved frame or None if frame logging is disabled
    """
    config = get_config()
    if not config.logging.enable_frame_logging:
        return None

    try:
        import cv2

        frames_dir = Path(config.logging.log_dir) / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename
        if track_id is not None:
            filename = f"frame_{frame_number:06d}_track_{track_id}"
        else:
            filename = f"frame_{frame_number:06d}"

        if suffix:
            filename += f"_{suffix}"

        filename += ".jpg"

        frame_path = frames_dir / filename
        cv2.imwrite(str(frame_path), frame)

        return str(frame_path)
    except Exception as e:
        logger = get_logger(__name__)
        logger.error(f"Failed to save debug frame: {e}")
        return None
