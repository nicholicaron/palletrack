"""Visualization utilities for pipeline output."""

import time
from typing import Dict, List, Tuple

import cv2
import numpy as np

from ..cv_models import BoundingBox, DocumentDetection, PalletDetection, PalletTrack


class FPSTracker:
    """Track processing FPS over a sliding window."""

    def __init__(self, window_size: int = 30):
        """Initialize FPS tracker.

        Args:
            window_size: Number of frames to average over
        """
        self.window_size = window_size
        self.frame_times = []
        self.last_time = time.time()

    def update(self):
        """Update with new frame."""
        current_time = time.time()
        frame_time = current_time - self.last_time
        self.frame_times.append(frame_time)

        # Keep only last N frame times
        if len(self.frame_times) > self.window_size:
            self.frame_times.pop(0)

        self.last_time = current_time

    def get_fps(self) -> float:
        """Calculate current FPS.

        Returns:
            Current FPS (averaged over window)
        """
        if not self.frame_times:
            return 0.0

        avg_time = sum(self.frame_times) / len(self.frame_times)
        return 1.0 / avg_time if avg_time > 0 else 0.0

    def reset(self):
        """Reset FPS tracker."""
        self.frame_times = []
        self.last_time = time.time()


class FrameAnnotator:
    """Draw visual annotations on frames."""

    # Colors (BGR format)
    COLOR_PALLET = (0, 255, 0)  # Green
    COLOR_DOCUMENT = (255, 0, 0)  # Blue
    COLOR_TRACK_TRAIL = (0, 255, 255)  # Yellow
    COLOR_TEXT = (255, 255, 255)  # White
    COLOR_PANEL_BG = (0, 0, 0)  # Black

    @staticmethod
    def annotate_complete_frame(
        frame: np.ndarray,
        pallet_detections: List[PalletDetection],
        active_tracks: Dict[int, PalletTrack],
        document_detections: List[DocumentDetection],
        ocr_processed: bool = False,
    ) -> np.ndarray:
        """Draw all annotations on frame.

        Annotations:
        - Pallet bboxes (green)
        - Track IDs and trails
        - Document region bboxes (blue)
        - OCR indicator

        Args:
            frame: Input frame
            pallet_detections: List of pallet detections
            active_tracks: Dictionary of active tracks
            document_detections: List of document detections
            ocr_processed: Whether OCR was processed this frame

        Returns:
            Annotated frame
        """
        annotated = frame.copy()

        # Draw pallet detections
        for detection in pallet_detections:
            label = f"Pallet {detection.track_id}" if detection.track_id else "Pallet"
            conf_str = f" ({detection.bbox.confidence:.2f})"
            annotated = FrameAnnotator._draw_bbox(
                annotated,
                detection.bbox,
                color=FrameAnnotator.COLOR_PALLET,
                label=label + conf_str,
                thickness=2,
            )

        # Draw track trails
        for track_id, track in active_tracks.items():
            annotated = FrameAnnotator._draw_track_trail(annotated, track)

        # Draw document regions
        for doc in document_detections:
            label = f"Doc (P{doc.parent_pallet_track_id})"
            conf_str = f" ({doc.bbox.confidence:.2f})"
            annotated = FrameAnnotator._draw_bbox(
                annotated,
                doc.bbox,
                color=FrameAnnotator.COLOR_DOCUMENT,
                label=label + conf_str,
                thickness=2,
            )

        # Draw OCR indicator if processed
        if ocr_processed:
            cv2.putText(
                annotated,
                "OCR",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),  # Red
                2,
            )

        return annotated

    @staticmethod
    def _draw_bbox(
        frame: np.ndarray,
        bbox: BoundingBox,
        color: Tuple[int, int, int],
        label: str = "",
        thickness: int = 2,
    ) -> np.ndarray:
        """Draw bounding box with label.

        Args:
            frame: Input frame
            bbox: BoundingBox to draw
            color: Color (BGR)
            label: Optional label text
            thickness: Line thickness

        Returns:
            Frame with bbox drawn
        """
        # Draw rectangle
        x1, y1 = int(bbox.x1), int(bbox.y1)
        x2, y2 = int(bbox.x2), int(bbox.y2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        # Draw label if provided
        if label:
            # Get text size for background
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )

            # Draw background rectangle for text
            cv2.rectangle(
                frame,
                (x1, y1 - text_height - baseline - 5),
                (x1 + text_width, y1),
                color,
                -1,  # Filled
            )

            # Draw text
            cv2.putText(
                frame,
                label,
                (x1, y1 - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),  # Black text
                1,
            )

        return frame

    @staticmethod
    def _draw_track_trail(frame: np.ndarray, track: PalletTrack) -> np.ndarray:
        """Draw track movement trail.

        Args:
            frame: Input frame
            track: PalletTrack to draw trail for

        Returns:
            Frame with trail drawn
        """
        if len(track.detections) < 2:
            return frame

        # Get center points of last N detections
        max_trail_length = 20
        recent_detections = track.detections[-max_trail_length:]

        points = []
        for detection in recent_detections:
            cx = int((detection.bbox.x1 + detection.bbox.x2) / 2)
            cy = int((detection.bbox.y1 + detection.bbox.y2) / 2)
            points.append((cx, cy))

        # Draw lines between points
        for i in range(len(points) - 1):
            # Fade color based on age (older = more transparent)
            alpha = i / len(points)
            color = tuple(int(c * alpha) for c in FrameAnnotator.COLOR_TRACK_TRAIL)

            cv2.line(frame, points[i], points[i + 1], color, 2)

        # Draw track ID at current position
        if points:
            current_pos = points[-1]
            cv2.putText(
                frame,
                f"ID:{track.track_id}",
                (current_pos[0] + 10, current_pos[1]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                FrameAnnotator.COLOR_TEXT,
                2,
            )

        return frame

    @staticmethod
    def add_info_panel(
        frame: np.ndarray,
        fps: float,
        active_tracks: int,
        processed_extractions: int,
        frame_number: int,
    ) -> np.ndarray:
        """Add info panel with system stats.

        Args:
            frame: Input frame
            fps: Current FPS
            active_tracks: Number of active tracks
            processed_extractions: Number of completed extractions
            frame_number: Current frame number

        Returns:
            Frame with info panel
        """
        h, w = frame.shape[:2]

        # Panel dimensions
        panel_height = 120
        panel_width = 300
        panel_x = w - panel_width - 10
        panel_y = 10

        # Draw semi-transparent panel background
        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (panel_x, panel_y),
            (panel_x + panel_width, panel_y + panel_height),
            FrameAnnotator.COLOR_PANEL_BG,
            -1,
        )
        # Blend with original frame
        alpha = 0.7
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        # Add text information
        text_x = panel_x + 10
        text_y = panel_y + 25
        line_height = 25

        info_lines = [
            f"Frame: {frame_number}",
            f"FPS: {fps:.1f}",
            f"Active Tracks: {active_tracks}",
            f"Extractions: {processed_extractions}",
        ]

        for i, line in enumerate(info_lines):
            cv2.putText(
                frame,
                line,
                (text_x, text_y + i * line_height),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                FrameAnnotator.COLOR_TEXT,
                1,
            )

        return frame

    @staticmethod
    def create_side_by_side(
        frame1: np.ndarray, frame2: np.ndarray, label1: str = "", label2: str = ""
    ) -> np.ndarray:
        """Create side-by-side comparison of two frames.

        Args:
            frame1: First frame
            frame2: Second frame
            label1: Label for first frame
            label2: Label for second frame

        Returns:
            Combined frame
        """
        # Ensure frames have same height
        h1, w1 = frame1.shape[:2]
        h2, w2 = frame2.shape[:2]

        if h1 != h2:
            # Resize second frame to match first
            frame2 = cv2.resize(frame2, (int(w2 * h1 / h2), h1))
            h2, w2 = frame2.shape[:2]

        # Concatenate horizontally
        combined = np.hstack([frame1, frame2])

        # Add labels if provided
        if label1:
            cv2.putText(
                combined,
                label1,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                2,
            )

        if label2:
            cv2.putText(
                combined,
                label2,
                (w1 + 10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                2,
            )

        return combined
