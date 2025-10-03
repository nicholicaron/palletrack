"""Core data models for pallet detection, tracking, OCR, and data extraction."""

from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator


class BoundingBox(BaseModel):
    """Bounding box for detected objects with utility methods.

    Attributes:
        x1: Left x coordinate
        y1: Top y coordinate
        x2: Right x coordinate
        y2: Bottom y coordinate
        confidence: Detection confidence score (0-1)
    """

    x1: float = Field(..., ge=0)
    y1: float = Field(..., ge=0)
    x2: float = Field(..., ge=0)
    y2: float = Field(..., ge=0)
    confidence: float = Field(..., ge=0.0, le=1.0)

    @field_validator('x2')
    @classmethod
    def x2_greater_than_x1(cls, v: float, info) -> float:
        """Validate x2 > x1."""
        if 'x1' in info.data and v <= info.data['x1']:
            raise ValueError('x2 must be greater than x1')
        return v

    @field_validator('y2')
    @classmethod
    def y2_greater_than_y1(cls, v: float, info) -> float:
        """Validate y2 > y1."""
        if 'y1' in info.data and v <= info.data['y1']:
            raise ValueError('y2 must be greater than y1')
        return v

    def area(self) -> float:
        """Calculate bounding box area.

        Returns:
            Area in square pixels
        """
        return (self.x2 - self.x1) * (self.y2 - self.y1)

    def center(self) -> Tuple[float, float]:
        """Calculate center point of bounding box.

        Returns:
            Tuple of (center_x, center_y)
        """
        return (
            (self.x1 + self.x2) / 2,
            (self.y1 + self.y2) / 2
        )

    def contains_point(self, x: float, y: float) -> bool:
        """Check if a point is inside the bounding box.

        Args:
            x: X coordinate
            y: Y coordinate

        Returns:
            True if point is inside the box
        """
        return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2

    def iou(self, other: "BoundingBox") -> float:
        """Calculate Intersection over Union with another bounding box.

        Args:
            other: Another BoundingBox instance

        Returns:
            IoU score between 0 and 1
        """
        # Calculate intersection coordinates
        x1_inter = max(self.x1, other.x1)
        y1_inter = max(self.y1, other.y1)
        x2_inter = min(self.x2, other.x2)
        y2_inter = min(self.y2, other.y2)

        # Check if there's an intersection
        if x2_inter < x1_inter or y2_inter < y1_inter:
            return 0.0

        # Calculate intersection area
        intersection = (x2_inter - x1_inter) * (y2_inter - y1_inter)

        # Calculate union area
        area1 = self.area()
        area2 = other.area()
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    def width(self) -> float:
        """Get bounding box width."""
        return self.x2 - self.x1

    def height(self) -> float:
        """Get bounding box height."""
        return self.y2 - self.y1


class PalletDetection(BaseModel):
    """Single pallet detection in a frame.

    Attributes:
        bbox: Bounding box of detected pallet
        frame_number: Frame number in video
        timestamp: Timestamp in seconds
        track_id: Optional tracking ID assigned by tracker
    """

    bbox: BoundingBox
    frame_number: int = Field(..., ge=0)
    timestamp: float = Field(..., ge=0.0)
    track_id: Optional[int] = Field(default=None, ge=0)

    class Config:
        json_schema_extra = {
            "example": {
                "bbox": {
                    "x1": 100,
                    "y1": 200,
                    "x2": 400,
                    "y2": 600,
                    "confidence": 0.92
                },
                "frame_number": 42,
                "timestamp": 1.4,
                "track_id": 7
            }
        }


class DocumentType(str, Enum):
    """Types of shipping documents that can be detected."""

    BOL = "BOL"  # Bill of Lading
    PACKING_LIST = "PACKING_LIST"
    SHIPPING_LABEL = "SHIPPING_LABEL"
    UNKNOWN = "UNKNOWN"


class DocumentDetection(BaseModel):
    """Single document detection on a pallet.

    Attributes:
        bbox: Bounding box of detected document
        frame_number: Frame number in video
        parent_pallet_track_id: Track ID of the pallet this document belongs to
        document_type: Type of shipping document detected
    """

    bbox: BoundingBox
    frame_number: int = Field(..., ge=0)
    parent_pallet_track_id: Optional[int] = Field(default=None, ge=0)
    document_type: Optional[DocumentType] = DocumentType.UNKNOWN

    class Config:
        json_schema_extra = {
            "example": {
                "bbox": {
                    "x1": 150,
                    "y1": 250,
                    "x2": 300,
                    "y2": 350,
                    "confidence": 0.88
                },
                "frame_number": 45,
                "parent_pallet_track_id": 7,
                "document_type": "SHIPPING_LABEL"
            }
        }


class OCRResult(BaseModel):
    """OCR extraction result from a document region.

    Attributes:
        text: Extracted text
        confidence: OCR confidence score (0-1)
        bbox: Bounding box of the text region
        frame_number: Frame number where text was extracted
    """

    text: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    bbox: BoundingBox
    frame_number: int = Field(..., ge=0)

    class Config:
        json_schema_extra = {
            "example": {
                "text": "TRACKING: 1Z999AA10123456784",
                "confidence": 0.94,
                "bbox": {
                    "x1": 160,
                    "y1": 260,
                    "x2": 290,
                    "y2": 280,
                    "confidence": 0.88
                },
                "frame_number": 45
            }
        }


class TrackStatus(str, Enum):
    """Status of a pallet track."""

    ACTIVE = "ACTIVE"  # Currently being tracked
    COMPLETED = "COMPLETED"  # Successfully tracked and processed
    LOST = "LOST"  # Lost tracking


class PalletTrack(BaseModel):
    """Complete tracking information for a single pallet.

    Attributes:
        track_id: Unique tracking identifier
        detections: List of all detections for this pallet
        document_regions: List of document detections on this pallet
        ocr_results: List of OCR results from documents
        status: Current tracking status
        first_seen_frame: First frame where pallet was detected
        last_seen_frame: Most recent frame where pallet was detected
    """

    track_id: int = Field(..., ge=0)
    detections: List[PalletDetection] = Field(default_factory=list)
    document_regions: List[DocumentDetection] = Field(default_factory=list)
    ocr_results: List[OCRResult] = Field(default_factory=list)
    status: TrackStatus = TrackStatus.ACTIVE
    first_seen_frame: int = Field(..., ge=0)
    last_seen_frame: int = Field(..., ge=0)

    @field_validator('last_seen_frame')
    @classmethod
    def last_seen_after_first(cls, v: int, info) -> int:
        """Validate last_seen_frame >= first_seen_frame."""
        if 'first_seen_frame' in info.data and v < info.data['first_seen_frame']:
            raise ValueError('last_seen_frame must be >= first_seen_frame')
        return v

    def duration_frames(self) -> int:
        """Calculate number of frames this pallet was tracked.

        Returns:
            Number of frames from first to last detection
        """
        return self.last_seen_frame - self.first_seen_frame + 1

    def get_best_ocr_text(self) -> str:
        """Get concatenated OCR text sorted by confidence.

        Returns:
            All OCR text joined by newlines, highest confidence first
        """
        sorted_results = sorted(
            self.ocr_results,
            key=lambda x: x.confidence,
            reverse=True
        )
        return "\n".join(r.text for r in sorted_results)

    class Config:
        json_schema_extra = {
            "example": {
                "track_id": 7,
                "detections": [],
                "document_regions": [],
                "ocr_results": [],
                "status": "ACTIVE",
                "first_seen_frame": 42,
                "last_seen_frame": 98
            }
        }


class ExtractedShippingData(BaseModel):
    """Structured shipping data extracted from OCR results.

    Attributes:
        track_id: ID of the pallet track this data came from
        document_type: Type of shipping document
        tracking_number: Shipping tracking number
        weight: Package weight in pounds
        destination_address: Full destination address
        destination_zip: Destination ZIP code
        origin_address: Full origin address
        items: List of items with descriptions and quantities
        confidence_score: Overall confidence in the extraction (0-1)
        needs_review: Whether this data needs manual review
    """

    track_id: int = Field(..., ge=0)
    document_type: str
    tracking_number: Optional[str] = None
    weight: Optional[float] = Field(default=None, gt=0)
    destination_address: Optional[str] = None
    destination_zip: Optional[str] = None
    origin_address: Optional[str] = None
    items: List[Dict[str, Any]] = Field(default_factory=list)
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    needs_review: bool = False

    class Config:
        json_schema_extra = {
            "example": {
                "track_id": 7,
                "document_type": "SHIPPING_LABEL",
                "tracking_number": "1Z999AA10123456784",
                "weight": 45.5,
                "destination_address": "123 Main St, Springfield, IL 62701",
                "destination_zip": "62701",
                "origin_address": "456 Factory Rd, Chicago, IL 60601",
                "items": [
                    {"description": "Widget A", "quantity": 100},
                    {"description": "Widget B", "quantity": 50}
                ],
                "confidence_score": 0.89,
                "needs_review": False
            }
        }
