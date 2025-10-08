"""Frame processing modules for PalleTrack CV pipeline."""

from .document_detector import DocumentDetector
from .document_utils import DocumentAssociator, DocumentRegionExtractor
from .extraction import (
    AddressExtractor,
    CarrierDetector,
    CompanyNameExtractor,
    DocumentTypeClassifier,
    ItemListExtractor,
    PackingListSummaryExtractor,
    RegexFieldExtractor,
    ServiceLevelDetector,
    ShippingDataExtractor,
)
from .frame_quality import FrameQualityScorer, calculate_angle_score, calculate_sharpness, calculate_size_score
from .frame_sampler import AdaptiveFrameSampler
from .frame_selection import FrameSelectionStrategy
from .frame_selector import BestFrameSelector
from .movement_analysis import MovementAnalyzer
from .ocr import DocumentOCR, MultiFrameOCRAggregator, OCRPostProcessor, OCRPreprocessor
from .pallet_detector import DetectionPostProcessor, DetectionVisualizer, PalletDetector
from .pallet_tracker import PalletTracker, TrackManager
from .track_utils import TrackVisualizer, calculate_track_velocity, calculate_track_size_change, get_track_summary

__all__ = [
    "FrameQualityScorer",
    "BestFrameSelector",
    "calculate_sharpness",
    "calculate_size_score",
    "calculate_angle_score",
    "AdaptiveFrameSampler",
    "FrameSelectionStrategy",
    "MovementAnalyzer",
    "PalletDetector",
    "DetectionPostProcessor",
    "DetectionVisualizer",
    "DocumentDetector",
    "DocumentAssociator",
    "DocumentRegionExtractor",
    "PalletTracker",
    "TrackManager",
    "TrackVisualizer",
    "calculate_track_velocity",
    "calculate_track_size_change",
    "get_track_summary",
    "OCRPreprocessor",
    "DocumentOCR",
    "MultiFrameOCRAggregator",
    "OCRPostProcessor",
    "DocumentTypeClassifier",
    "RegexFieldExtractor",
    "CarrierDetector",
    "ServiceLevelDetector",
    "AddressExtractor",
    "CompanyNameExtractor",
    "ItemListExtractor",
    "PackingListSummaryExtractor",
    "ShippingDataExtractor",
]
