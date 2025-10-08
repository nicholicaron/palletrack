"""YOLOv8 document detection module.

This module provides a wrapper around YOLOv8 for detecting shipping document
pouches on pallets. Optimized to search only within pallet bounding boxes for
efficiency.
"""

from pathlib import Path
from typing import Dict, List

import numpy as np
from ultralytics import YOLO

from app.cv_models import BoundingBox, DocumentDetection, PalletTrack


class DocumentDetector:
    """YOLOv8-based document pouch detector.

    Detects shipping documents attached to pallets. Optimized to search only
    within known pallet regions for improved performance.

    Attributes:
        model: YOLOv8 model instance
        conf_threshold: Minimum confidence threshold for detections
        device: Device to run inference on ('cuda' or 'cpu')

    Example:
        >>> config = {
        ...     'detection': {
        ...         'document_model_path': 'models/document_yolov8n.pt',
        ...         'document_conf_threshold': 0.6,
        ...         'device': 'cuda'
        ...     }
        ... }
        >>> detector = DocumentDetector(config)
        >>> docs = detector.detect(frame, frame_number=0, pallet_tracks=active_tracks)
    """

    def __init__(self, config: Dict):
        """Initialize YOLOv8 model for document detection.

        Args:
            config: Configuration dict with detection settings
                - detection.document_model_path: Path to YOLOv8 model weights
                - detection.document_conf_threshold: Confidence threshold (default: 0.6)
                - detection.device: Device to use - 'cuda' or 'cpu' (default: 'cuda')

        Raises:
            FileNotFoundError: If model file doesn't exist
            RuntimeError: If model fails to load
        """
        detection_config = config.get("detection", {})

        # Model path
        model_path = detection_config.get("document_model_path", "models/document_yolov8n.pt")
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Document detection model not found: {model_path}")

        # Threshold
        self.conf_threshold = detection_config.get("document_conf_threshold", 0.6)

        # Device
        self.device = detection_config.get("device", "cuda")

        # Load YOLO model
        try:
            self.model = YOLO(model_path)
            self.model.to(self.device)
        except Exception as e:
            raise RuntimeError(f"Failed to load YOLO model from {model_path}: {e}")

    def detect(
        self,
        frame: np.ndarray,
        frame_number: int,
        pallet_tracks: Dict[int, PalletTrack]
    ) -> List[DocumentDetection]:
        """Detect document regions in frame.

        Searches only within pallet bounding boxes for efficiency. Each detected
        document is initially unassociated (parent_pallet_track_id=None).
        Use DocumentAssociator to assign documents to pallets.

        Args:
            frame: Input frame (BGR format, numpy array)
            frame_number: Frame index in video sequence
            pallet_tracks: Dictionary of active pallet tracks (track_id -> PalletTrack)

        Returns:
            List of DocumentDetection objects (unassociated with pallets)

        Example:
            >>> detections = detector.detect(frame, 42, active_tracks)
            >>> for det in detections:
            ...     print(f"Document at {det.bbox} with conf {det.bbox.confidence:.2f}")
        """
        if not pallet_tracks:
            # No pallets to search within
            return []

        all_documents = []

        # Search within each pallet's bounding box
        for track_id, track in pallet_tracks.items():
            if not track.detections:
                continue

            # Get most recent pallet detection
            latest_detection = track.detections[-1]
            pallet_bbox = latest_detection.bbox

            # Extract pallet region from frame
            x1, y1 = int(pallet_bbox.x1), int(pallet_bbox.y1)
            x2, y2 = int(pallet_bbox.x2), int(pallet_bbox.y2)

            # Validate coordinates are within frame bounds
            h, w = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            if x2 <= x1 or y2 <= y1:
                continue

            # Crop pallet region
            pallet_region = frame[y1:y2, x1:x2]

            # Run document detection on pallet region
            results = self.model(
                pallet_region,
                conf=self.conf_threshold,
                verbose=False
            )

            # Convert detections to global frame coordinates
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    # Extract box coordinates (relative to pallet_region)
                    box_x1, box_y1, box_x2, box_y2 = box.xyxy[0].cpu().numpy()
                    confidence = float(box.conf[0].cpu().numpy())

                    # Convert to global frame coordinates
                    global_x1 = x1 + box_x1
                    global_y1 = y1 + box_y1
                    global_x2 = x1 + box_x2
                    global_y2 = y1 + box_y2

                    # Create BoundingBox in global coordinates
                    bbox = BoundingBox(
                        x1=float(global_x1),
                        y1=float(global_y1),
                        x2=float(global_x2),
                        y2=float(global_y2),
                        confidence=confidence
                    )

                    # Create DocumentDetection (unassociated)
                    document = DocumentDetection(
                        bbox=bbox,
                        frame_number=frame_number,
                        parent_pallet_track_id=None  # Will be assigned by DocumentAssociator
                    )
                    all_documents.append(document)

        return all_documents

    def detect_full_frame(
        self,
        frame: np.ndarray,
        frame_number: int
    ) -> List[DocumentDetection]:
        """Detect documents in full frame (fallback method).

        Use this when pallet tracking is not available. Less efficient than
        searching within pallet regions.

        Args:
            frame: Input frame (BGR format, numpy array)
            frame_number: Frame index in video sequence

        Returns:
            List of DocumentDetection objects (unassociated)

        Example:
            >>> detections = detector.detect_full_frame(frame, 42)
        """
        # Run inference on full frame
        results = self.model(
            frame,
            conf=self.conf_threshold,
            verbose=False
        )

        # Convert to DocumentDetection objects
        documents = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Extract box coordinates and confidence
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = float(box.conf[0].cpu().numpy())

                # Create BoundingBox
                bbox = BoundingBox(
                    x1=float(x1),
                    y1=float(y1),
                    x2=float(x2),
                    y2=float(y2),
                    confidence=confidence
                )

                # Create DocumentDetection
                document = DocumentDetection(
                    bbox=bbox,
                    frame_number=frame_number,
                    parent_pallet_track_id=None
                )
                documents.append(document)

        return documents
