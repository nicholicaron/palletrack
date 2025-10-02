import tempfile
from pathlib import Path

import cv2
import numpy as np
import supervision as sv
from paddleocr import PaddleOCR
from ultralytics import YOLO

from app.schemas import OCRTextData, PalletData, VideoProcessResponse


class VideoProcessor:
    def __init__(self):
        # Initialize YOLO model for pallet detection
        # Using YOLOv8 nano for speed, can upgrade to yolov8s/m/l for accuracy
        self.model = YOLO("yolov8n.pt")

        # Initialize ByteTrack tracker via supervision
        self.tracker = sv.ByteTrack()

        # Initialize PaddleOCR
        self.ocr = PaddleOCR(use_angle_cls=True, lang='en')

        # Store pallet tracking data
        self.pallet_data: dict[int, dict] = {}

    def process_video(self, video_path: str | Path) -> VideoProcessResponse:
        """
        Process video to detect and track pallets, extract text via OCR.

        Args:
            video_path: Path to input video file

        Returns:
            VideoProcessResponse with detected pallet data
        """
        video_path = Path(video_path)
        if not video_path.exists():
            return VideoProcessResponse(
                success=False,
                message=f"Video file not found: {video_path}",
                pallets=[],
                total_frames=0,
                fps=0.0
            )

        # Open video
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        frame_idx = 0
        self.pallet_data = {}

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Run YOLOv8 detection
                # We'll use class 0 (person) as a proxy for now
                # In production, you'd fine-tune YOLO on pallet images
                results = self.model(frame, verbose=False)[0]

                # Convert to supervision Detections format
                detections = sv.Detections.from_ultralytics(results)

                # Update tracker
                detections = self.tracker.update_with_detections(detections)

                # Process each tracked detection
                if len(detections) > 0:
                    self._process_detections(frame, detections, frame_idx)

                frame_idx += 1

        finally:
            cap.release()

        # Convert tracking data to response format
        pallets = self._build_pallet_list()

        return VideoProcessResponse(
            success=True,
            message=f"Processed {frame_idx} frames, found {len(pallets)} tracked pallets",
            pallets=pallets,
            total_frames=frame_idx,
            fps=fps
        )

    def _process_detections(
        self,
        frame: np.ndarray,
        detections: sv.Detections,
        frame_idx: int
    ) -> None:
        """Process detections for a single frame."""
        for i in range(len(detections)):
            # Get tracking ID
            track_id = detections.tracker_id[i] if detections.tracker_id is not None else -1
            if track_id == -1:
                continue

            # Get bounding box
            bbox = detections.xyxy[i]  # [x1, y1, x2, y2]
            x1, y1, x2, y2 = map(int, bbox)

            # Initialize tracking data if new pallet
            if track_id not in self.pallet_data:
                self.pallet_data[track_id] = {
                    "track_id": track_id,
                    "first_seen_frame": frame_idx,
                    "last_seen_frame": frame_idx,
                    "bbox": bbox.tolist(),
                    "ocr_texts": []
                }
            else:
                # Update last seen frame and bbox
                self.pallet_data[track_id]["last_seen_frame"] = frame_idx
                self.pallet_data[track_id]["bbox"] = bbox.tolist()

            # Run OCR on pallet region (sample every 10 frames to save compute)
            if frame_idx % 10 == 0:
                self._run_ocr_on_region(frame, x1, y1, x2, y2, track_id)

    def _run_ocr_on_region(
        self,
        frame: np.ndarray,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        track_id: int
    ) -> None:
        """Run OCR on a specific region of the frame."""
        # Extract region
        region = frame[y1:y2, x1:x2]

        # Skip if region is too small
        if region.shape[0] < 10 or region.shape[1] < 10:
            return

        # Run PaddleOCR
        try:
            result = self.ocr.ocr(region, cls=True)

            if result and result[0]:
                for line in result[0]:
                    # line format: [bbox_coords, (text, confidence)]
                    bbox_coords = line[0]
                    text, confidence = line[1]

                    # Convert relative coords to absolute frame coords
                    # bbox_coords is [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                    x_coords = [pt[0] + x1 for pt in bbox_coords]
                    y_coords = [pt[1] + y1 for pt in bbox_coords]
                    abs_bbox = [
                        min(x_coords),
                        min(y_coords),
                        max(x_coords),
                        max(y_coords)
                    ]

                    ocr_data = OCRTextData(
                        text=text,
                        confidence=confidence,
                        bbox=abs_bbox
                    )

                    # Add to pallet data if not duplicate
                    if not self._is_duplicate_text(track_id, text):
                        self.pallet_data[track_id]["ocr_texts"].append(
                            ocr_data.model_dump()
                        )
        except Exception:
            # Silently skip OCR errors
            pass

    def _is_duplicate_text(self, track_id: int, text: str) -> bool:
        """Check if text already exists for this pallet."""
        existing_texts = [
            ocr["text"] for ocr in self.pallet_data[track_id]["ocr_texts"]
        ]
        return text in existing_texts

    def _build_pallet_list(self) -> list[PalletData]:
        """Convert internal tracking data to PalletData list."""
        pallets = []
        for track_id, data in self.pallet_data.items():
            pallet = PalletData(
                track_id=data["track_id"],
                first_seen_frame=data["first_seen_frame"],
                last_seen_frame=data["last_seen_frame"],
                bbox=data["bbox"],
                ocr_texts=[OCRTextData(**ocr) for ocr in data["ocr_texts"]]
            )
            pallets.append(pallet)

        return pallets


# Singleton instance
_video_processor: VideoProcessor | None = None


def get_video_processor() -> VideoProcessor:
    """Get or create video processor singleton."""
    global _video_processor
    if _video_processor is None:
        _video_processor = VideoProcessor()
    return _video_processor
