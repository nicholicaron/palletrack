"""YOLOv8 pallet detection module.

This module provides a clean wrapper around YOLOv8 for detecting pallets
in warehouse video frames with post-processing and visualization helpers.
"""

from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

from app.cv_models import BoundingBox, PalletDetection


class PalletDetector:
    """YOLOv8-based pallet detector with configurable settings.

    Wraps Ultralytics YOLOv8 for pallet detection in warehouse environments.
    Handles model loading, inference, and conversion to PalletDetection objects.

    Attributes:
        model: YOLOv8 model instance
        conf_threshold: Minimum confidence threshold for detections
        iou_threshold: IoU threshold for Non-Maximum Suppression
        device: Device to run inference on ('cuda' or 'cpu')

    Example:
        >>> config = {
        ...     'detection': {
        ...         'pallet_model_path': 'models/pallet_yolov8n.pt',
        ...         'pallet_conf_threshold': 0.5,
        ...         'pallet_iou_threshold': 0.45,
        ...         'device': 'cuda'
        ...     }
        ... }
        >>> detector = PalletDetector(config)
        >>> detections = detector.detect(frame, frame_number=0, timestamp=0.0)
    """

    def __init__(self, config: Dict):
        """Initialize YOLOv8 model for pallet detection.

        Args:
            config: Configuration dict with detection settings
                - detection.pallet_model_path: Path to YOLOv8 model weights
                - detection.pallet_conf_threshold: Confidence threshold (default: 0.5)
                - detection.pallet_iou_threshold: IoU threshold for NMS (default: 0.45)
                - detection.device: Device to use - 'cuda' or 'cpu' (default: 'cuda')

        Raises:
            FileNotFoundError: If model file doesn't exist
            RuntimeError: If model fails to load
        """
        detection_config = config.get("detection", {})

        # Model path
        model_path = detection_config.get("pallet_model_path", "models/pallet_yolov8n.pt")
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Pallet detection model not found: {model_path}")

        # Thresholds
        self.conf_threshold = detection_config.get("pallet_conf_threshold", 0.5)
        self.iou_threshold = detection_config.get("pallet_iou_threshold", 0.45)

        # Device
        self.device = detection_config.get("device", "cuda")

        # Load YOLO model
        try:
            self.model = YOLO(model_path)
            self.model.to(self.device)
        except Exception as e:
            raise RuntimeError(f"Failed to load YOLO model from {model_path}: {e}")

    def detect(
        self, frame: np.ndarray, frame_number: int, timestamp: float
    ) -> List[PalletDetection]:
        """Detect pallets in a single frame.

        Args:
            frame: Input frame (BGR format, numpy array)
            frame_number: Frame index in video sequence
            timestamp: Timestamp of frame in seconds

        Returns:
            List of PalletDetection objects, filtered by confidence threshold

        Example:
            >>> detections = detector.detect(frame, frame_number=42, timestamp=1.4)
            >>> for det in detections:
            ...     print(f"Pallet at {det.bbox} with conf {det.confidence:.2f}")
        """
        # Run inference
        results = self.model(
            frame, conf=self.conf_threshold, iou=self.iou_threshold, verbose=False
        )

        # Convert to PalletDetection objects
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Extract box coordinates and confidence
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = float(box.conf[0].cpu().numpy())

                # Create BoundingBox
                bbox = BoundingBox(
                    x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2), confidence=confidence
                )

                # Create PalletDetection
                detection = PalletDetection(
                    bbox=bbox, frame_number=frame_number, timestamp=timestamp
                )
                detections.append(detection)

        return detections

    def detect_batch(
        self, frames: List[np.ndarray], frame_numbers: List[int], timestamps: List[float]
    ) -> List[List[PalletDetection]]:
        """Batch inference for efficiency.

        Processes multiple frames in a single batch for better GPU utilization.

        Args:
            frames: List of input frames (BGR format)
            frame_numbers: List of frame indices
            timestamps: List of timestamps (seconds)

        Returns:
            List of detection lists (one per frame)

        Raises:
            ValueError: If input lists have different lengths

        Example:
            >>> batch_detections = detector.detect_batch(
            ...     frames=[frame1, frame2, frame3],
            ...     frame_numbers=[0, 1, 2],
            ...     timestamps=[0.0, 0.033, 0.066]
            ... )
            >>> for frame_dets in batch_detections:
            ...     print(f"Found {len(frame_dets)} pallets")
        """
        if not (len(frames) == len(frame_numbers) == len(timestamps)):
            raise ValueError("frames, frame_numbers, and timestamps must have same length")

        if not frames:
            return []

        # Run batch inference
        results = self.model(
            frames, conf=self.conf_threshold, iou=self.iou_threshold, verbose=False
        )

        # Convert results to PalletDetection objects
        all_detections = []
        for idx, result in enumerate(results):
            frame_detections = []
            boxes = result.boxes

            for box in boxes:
                # Extract box coordinates and confidence
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = float(box.conf[0].cpu().numpy())

                # Create BoundingBox
                bbox = BoundingBox(
                    x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2), confidence=confidence
                )

                # Create PalletDetection
                detection = PalletDetection(
                    bbox=bbox, frame_number=frame_numbers[idx], timestamp=timestamps[idx]
                )
                frame_detections.append(detection)

            all_detections.append(frame_detections)

        return all_detections


class DetectionPostProcessor:
    """Post-processing utilities for pallet detections.

    Provides filtering and refinement operations on detection results.
    """

    @staticmethod
    def filter_overlapping_detections(
        detections: List[PalletDetection], iou_threshold: float = 0.5
    ) -> List[PalletDetection]:
        """Apply Non-Maximum Suppression to remove overlapping detections.

        YOLO usually applies NMS internally, but this provides additional filtering
        when needed (e.g., combining detections from multiple models).

        Args:
            detections: List of PalletDetection objects
            iou_threshold: IoU threshold for considering boxes as overlapping

        Returns:
            Filtered list with overlapping detections removed

        Example:
            >>> filtered = DetectionPostProcessor.filter_overlapping_detections(
            ...     detections, iou_threshold=0.5
            ... )
        """
        if not detections:
            return []

        # Sort by confidence (descending)
        sorted_dets = sorted(detections, key=lambda d: d.bbox.confidence, reverse=True)

        # NMS algorithm
        keep = []
        while sorted_dets:
            # Keep highest confidence detection
            current = sorted_dets.pop(0)
            keep.append(current)

            # Remove detections with high IoU
            sorted_dets = [
                det
                for det in sorted_dets
                if current.bbox.iou(det.bbox) < iou_threshold
            ]

        return keep

    @staticmethod
    def filter_by_size(
        detections: List[PalletDetection], min_area: int = 10000, max_area: int = 500000
    ) -> List[PalletDetection]:
        """Filter detections by bounding box area.

        Removes unreasonably small or large detections that are likely false positives.
        Pallets in warehouse video should fall within expected size range.

        Args:
            detections: List of PalletDetection objects
            min_area: Minimum bbox area in pixels (default: 10000)
            max_area: Maximum bbox area in pixels (default: 500000)

        Returns:
            Filtered list with only detections in valid size range

        Example:
            >>> filtered = DetectionPostProcessor.filter_by_size(
            ...     detections, min_area=10000, max_area=500000
            ... )
        """
        return [
            det for det in detections if min_area <= det.bbox.area() <= max_area
        ]

    @staticmethod
    def filter_by_confidence(
        detections: List[PalletDetection], min_confidence: float = 0.5
    ) -> List[PalletDetection]:
        """Filter detections by confidence threshold.

        Args:
            detections: List of PalletDetection objects
            min_confidence: Minimum confidence threshold (0.0-1.0)

        Returns:
            Filtered list with only high-confidence detections

        Example:
            >>> high_conf = DetectionPostProcessor.filter_by_confidence(
            ...     detections, min_confidence=0.7
            ... )
        """
        return [det for det in detections if det.bbox.confidence >= min_confidence]


class DetectionVisualizer:
    """Visualization utilities for pallet detections.

    Provides methods to draw detection results on frames for debugging and analysis.
    """

    @staticmethod
    def draw_detections(
        frame: np.ndarray,
        detections: List[PalletDetection],
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2,
        show_confidence: bool = True,
        show_id: bool = False,
    ) -> np.ndarray:
        """Draw bounding boxes and labels on frame.

        Args:
            frame: Input frame (BGR format)
            detections: List of PalletDetection objects
            color: Box color in BGR format (default: green)
            thickness: Line thickness in pixels (default: 2)
            show_confidence: Whether to display confidence score (default: True)
            show_id: Whether to display detection ID if available (default: False)

        Returns:
            Annotated frame copy (original frame is not modified)

        Example:
            >>> annotated = DetectionVisualizer.draw_detections(
            ...     frame, detections, color=(0, 255, 0), thickness=2
            ... )
            >>> cv2.imshow('Detections', annotated)
        """
        # Create copy to avoid modifying original
        annotated = frame.copy()

        for detection in detections:
            bbox = detection.bbox

            # Draw bounding box
            pt1 = (int(bbox.x1), int(bbox.y1))
            pt2 = (int(bbox.x2), int(bbox.y2))
            cv2.rectangle(annotated, pt1, pt2, color, thickness)

            # Prepare label text
            label_parts = []
            if show_id and hasattr(detection, "track_id") and detection.track_id is not None:
                label_parts.append(f"ID:{detection.track_id}")
            if show_confidence:
                label_parts.append(f"{detection.bbox.confidence:.2f}")

            if label_parts:
                label = " ".join(label_parts)

                # Calculate text size for background
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                font_thickness = 1
                (text_width, text_height), baseline = cv2.getTextSize(
                    label, font, font_scale, font_thickness
                )

                # Draw background rectangle for text
                bg_pt1 = (int(bbox.x1), int(bbox.y1) - text_height - baseline - 5)
                bg_pt2 = (int(bbox.x1) + text_width + 5, int(bbox.y1))
                cv2.rectangle(annotated, bg_pt1, bg_pt2, color, -1)

                # Draw text
                text_pt = (int(bbox.x1) + 2, int(bbox.y1) - 5)
                cv2.putText(
                    annotated,
                    label,
                    text_pt,
                    font,
                    font_scale,
                    (255, 255, 255),
                    font_thickness,
                )

        return annotated

    @staticmethod
    def draw_detection_grid(
        frames: List[np.ndarray],
        detections_per_frame: List[List[PalletDetection]],
        grid_cols: int = 4,
        color: Tuple[int, int, int] = (0, 255, 0),
    ) -> np.ndarray:
        """Draw multiple frames with detections in a grid layout.

        Useful for visualizing batch detection results.

        Args:
            frames: List of input frames
            detections_per_frame: List of detection lists (one per frame)
            grid_cols: Number of columns in grid (default: 4)
            color: Box color in BGR format (default: green)

        Returns:
            Grid image with annotated frames

        Example:
            >>> grid = DetectionVisualizer.draw_detection_grid(
            ...     frames=[frame1, frame2, frame3, frame4],
            ...     detections_per_frame=[dets1, dets2, dets3, dets4],
            ...     grid_cols=2
            ... )
            >>> cv2.imshow('Detection Grid', grid)
        """
        if not frames or len(frames) != len(detections_per_frame):
            raise ValueError("frames and detections_per_frame must have same length")

        # Annotate each frame
        annotated_frames = [
            DetectionVisualizer.draw_detections(frame, dets, color=color)
            for frame, dets in zip(frames, detections_per_frame)
        ]

        # Calculate grid dimensions
        n_frames = len(annotated_frames)
        grid_rows = (n_frames + grid_cols - 1) // grid_cols

        # Get frame dimensions (assume all frames same size)
        h, w = annotated_frames[0].shape[:2]

        # Create empty grid
        grid = np.zeros((grid_rows * h, grid_cols * w, 3), dtype=np.uint8)

        # Place frames in grid
        for idx, frame in enumerate(annotated_frames):
            row = idx // grid_cols
            col = idx % grid_cols
            grid[row * h : (row + 1) * h, col * w : (col + 1) * w] = frame

        return grid
