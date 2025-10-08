"""Utility functions and visualization for pallet tracking.

This module provides helper functions for track visualization and analysis.
"""

from collections import deque
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from app.cv_models import PalletTrack


class TrackVisualizer:
    """Visualize pallet tracks with bounding boxes and trails.

    Provides visualization utilities for displaying active tracks,
    track history, and status information on video frames.

    Attributes:
        trail_history: Dictionary of track trail points (deque of center points)
        max_trail_length: Maximum number of points to keep in trail

    Example:
        >>> visualizer = TrackVisualizer(max_trail_length=30)
        >>> frame = visualizer.draw_tracks(frame, active_tracks, show_trail=True)
    """

    # Color palette for tracks (BGR format for OpenCV)
    TRACK_COLORS = [
        (255, 0, 0),      # Blue
        (0, 255, 0),      # Green
        (0, 0, 255),      # Red
        (255, 255, 0),    # Cyan
        (255, 0, 255),    # Magenta
        (0, 255, 255),    # Yellow
        (128, 0, 128),    # Purple
        (0, 128, 128),    # Teal
        (128, 128, 0),    # Olive
        (255, 128, 0),    # Orange
    ]

    def __init__(self, max_trail_length: int = 30):
        """Initialize track visualizer.

        Args:
            max_trail_length: Maximum number of trail points to display
        """
        self.trail_history: Dict[int, deque] = {}
        self.max_trail_length = max_trail_length

    @staticmethod
    def draw_tracks(frame: np.ndarray,
                   active_tracks: Dict[int, PalletTrack],
                   show_trail: bool = True,
                   show_info: bool = True) -> np.ndarray:
        """Visualize active tracks with bounding boxes and optional trails.

        Args:
            frame: Input frame (BGR format)
            active_tracks: Dictionary of active PalletTrack objects
            show_trail: Whether to show track movement trails
            show_info: Whether to show track information text

        Returns:
            Frame with visualizations drawn
        """
        # Create a copy to avoid modifying original
        vis_frame = frame.copy()

        # Create visualizer instance for trail tracking
        visualizer = TrackVisualizer()

        for track_id, track in active_tracks.items():
            if not track.detections:
                continue

            # Get latest detection
            latest_detection = track.detections[-1]
            bbox = latest_detection.bbox

            # Choose color based on track ID
            color = visualizer._get_track_color(track_id)

            # Draw bounding box
            x1, y1, x2, y2 = int(bbox.x1), int(bbox.y1), int(bbox.x2), int(bbox.y2)
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)

            # Draw track ID and info
            if show_info:
                # Prepare track info text
                track_length = len(track.detections)
                conf = bbox.confidence
                label = f"ID:{track_id} L:{track_length} C:{conf:.2f}"

                # Calculate text size for background
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.6
                thickness = 2
                (text_width, text_height), baseline = cv2.getTextSize(
                    label, font, font_scale, thickness
                )

                # Draw text background
                text_x = x1
                text_y = y1 - 10 if y1 - 10 > text_height else y1 + text_height + 10
                cv2.rectangle(
                    vis_frame,
                    (text_x, text_y - text_height - baseline),
                    (text_x + text_width, text_y + baseline),
                    color,
                    -1  # Filled
                )

                # Draw text
                cv2.putText(
                    vis_frame,
                    label,
                    (text_x, text_y),
                    font,
                    font_scale,
                    (255, 255, 255),  # White text
                    thickness
                )

            # Draw trail
            if show_trail and len(track.detections) > 1:
                visualizer._draw_track_trail(vis_frame, track, color)

        return vis_frame

    def _draw_track_trail(self, frame: np.ndarray, track: PalletTrack, color: Tuple[int, int, int]):
        """Draw movement trail for a track.

        Args:
            frame: Frame to draw on
            track: PalletTrack object
            color: Color for trail (BGR)
        """
        # Get center points from detection history
        centers = []
        for detection in track.detections[-self.max_trail_length:]:
            center = detection.bbox.center()
            centers.append((int(center[0]), int(center[1])))

        # Draw trail with decreasing opacity
        if len(centers) < 2:
            return

        for i in range(len(centers) - 1):
            # Calculate alpha based on position in trail (older = more transparent)
            alpha = (i + 1) / len(centers)
            thickness = max(1, int(3 * alpha))

            # Draw line segment
            cv2.line(
                frame,
                centers[i],
                centers[i + 1],
                color,
                thickness
            )

    def _get_track_color(self, track_id: int) -> Tuple[int, int, int]:
        """Get consistent color for a track ID.

        Args:
            track_id: Track ID

        Returns:
            BGR color tuple
        """
        return self.TRACK_COLORS[track_id % len(self.TRACK_COLORS)]

    @staticmethod
    def draw_track_stats(frame: np.ndarray,
                        active_count: int,
                        completed_count: int,
                        frame_number: int) -> np.ndarray:
        """Draw tracking statistics on frame.

        Args:
            frame: Input frame
            active_count: Number of active tracks
            completed_count: Number of completed tracks
            frame_number: Current frame number

        Returns:
            Frame with statistics drawn
        """
        vis_frame = frame.copy()

        # Prepare stats text
        stats = [
            f"Frame: {frame_number}",
            f"Active Tracks: {active_count}",
            f"Completed: {completed_count}"
        ]

        # Draw background panel
        panel_height = 100
        panel_width = 300
        overlay = vis_frame.copy()
        cv2.rectangle(
            overlay,
            (10, 10),
            (10 + panel_width, 10 + panel_height),
            (0, 0, 0),
            -1
        )
        # Blend with original
        cv2.addWeighted(overlay, 0.6, vis_frame, 0.4, 0, vis_frame)

        # Draw text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        y_offset = 35

        for i, stat in enumerate(stats):
            cv2.putText(
                vis_frame,
                stat,
                (20, y_offset + i * 25),
                font,
                font_scale,
                (255, 255, 255),
                thickness
            )

        return vis_frame


def calculate_track_velocity(track: PalletTrack,
                             num_frames: int = 5) -> Tuple[float, float]:
    """Calculate track velocity over recent frames.

    Args:
        track: PalletTrack object
        num_frames: Number of recent frames to use for calculation

    Returns:
        Tuple of (velocity_x, velocity_y) in pixels per frame
    """
    if len(track.detections) < 2:
        return (0.0, 0.0)

    # Get recent detections
    recent = track.detections[-min(num_frames, len(track.detections)):]

    # Calculate displacement
    first_center = recent[0].bbox.center()
    last_center = recent[-1].bbox.center()

    # Calculate frames elapsed
    frames_elapsed = recent[-1].frame_number - recent[0].frame_number
    if frames_elapsed == 0:
        return (0.0, 0.0)

    # Calculate velocity
    velocity_x = (last_center[0] - first_center[0]) / frames_elapsed
    velocity_y = (last_center[1] - first_center[1]) / frames_elapsed

    return (velocity_x, velocity_y)


def calculate_track_size_change(track: PalletTrack,
                                num_frames: int = 5) -> float:
    """Calculate relative size change of track over recent frames.

    Args:
        track: PalletTrack object
        num_frames: Number of recent frames to use

    Returns:
        Relative size change ratio (e.g., 0.2 = 20% size change)
    """
    if len(track.detections) < 2:
        return 0.0

    # Get recent detections
    recent = track.detections[-min(num_frames, len(track.detections)):]

    # Calculate size change
    first_area = recent[0].bbox.area()
    last_area = recent[-1].bbox.area()

    if first_area == 0:
        return 0.0

    size_change_ratio = abs(last_area - first_area) / first_area
    return size_change_ratio


def get_track_summary(track: PalletTrack) -> Dict:
    """Generate summary statistics for a track.

    Args:
        track: PalletTrack object

    Returns:
        Dictionary with track summary statistics
    """
    if not track.detections:
        return {
            'track_id': track.track_id,
            'num_detections': 0,
            'duration_frames': 0,
            'status': track.status.value
        }

    # Calculate statistics
    confidences = [det.bbox.confidence for det in track.detections]
    areas = [det.bbox.area() for det in track.detections]

    velocity_x, velocity_y = calculate_track_velocity(track)
    size_change = calculate_track_size_change(track)

    return {
        'track_id': track.track_id,
        'num_detections': len(track.detections),
        'duration_frames': track.duration_frames(),
        'first_frame': track.first_seen_frame,
        'last_frame': track.last_seen_frame,
        'avg_confidence': np.mean(confidences),
        'min_confidence': np.min(confidences),
        'max_confidence': np.max(confidences),
        'avg_area': np.mean(areas),
        'velocity_x': velocity_x,
        'velocity_y': velocity_y,
        'size_change_ratio': size_change,
        'num_documents': len(track.document_regions),
        'num_ocr_results': len(track.ocr_results),
        'status': track.status.value
    }
