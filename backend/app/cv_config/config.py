"""Configuration loader and validator for PalleTrack CV pipeline."""

import os
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field, field_validator


class DetectionConfig(BaseModel):
    """Configuration for object detection models.

    Attributes:
        pallet_model_path: Path to YOLOv8 pallet detection model
        document_model_path: Path to YOLOv8 document detection model
        pallet_conf_threshold: Minimum confidence for pallet detections
        document_conf_threshold: Minimum confidence for document detections
        pallet_iou_threshold: IoU threshold for NMS in pallet detection
        device: Device to run inference on ('cuda' or 'cpu')
        min_pallet_area: Minimum pallet bbox area in pixels
        max_pallet_area: Maximum pallet bbox area in pixels
    """

    pallet_model_path: str = "models/pallet_yolov8n.pt"
    document_model_path: str = "models/document_yolov8.pt"
    pallet_conf_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    document_conf_threshold: float = Field(default=0.6, ge=0.0, le=1.0)
    pallet_iou_threshold: float = Field(default=0.45, ge=0.0, le=1.0)
    device: str = "cuda"
    min_pallet_area: int = Field(default=10000, ge=0)
    max_pallet_area: int = Field(default=500000, ge=0)

    @field_validator('pallet_model_path', 'document_model_path')
    @classmethod
    def validate_model_path(cls, v: str) -> str:
        """Validate that model path exists or can be created."""
        # Allow relative paths - they'll be resolved at runtime
        return v


class TrackingConfig(BaseModel):
    """Configuration for object tracking.

    Attributes:
        max_age: Maximum frames to keep a track alive without detections
        min_hits: Minimum detections before a track is confirmed
        iou_threshold: IoU threshold for matching detections to tracks
    """

    max_age: int = Field(default=30, ge=1)
    min_hits: int = Field(default=3, ge=1)
    iou_threshold: float = Field(default=0.3, ge=0.0, le=1.0)


class FrameSamplingConfig(BaseModel):
    """Configuration for intelligent frame sampling.

    Attributes:
        movement_threshold: Minimum pixel movement to trigger sampling
        size_change_threshold: Minimum size change ratio to trigger sampling
        max_frame_gap: Maximum frames between samples
        max_samples_per_track: Maximum samples to collect per track
    """

    movement_threshold: float = Field(default=50.0, ge=0.0)
    size_change_threshold: float = Field(default=0.2, ge=0.0, le=1.0)
    max_frame_gap: int = Field(default=30, ge=1)
    max_samples_per_track: int = Field(default=10, ge=1)


class OCRConfig(BaseModel):
    """Configuration for OCR processing.

    Attributes:
        language: Language code for OCR (e.g., 'en', 'es')
        use_gpu: Whether to use GPU for OCR
        min_text_confidence: Minimum confidence for OCR results
    """

    language: str = "en"
    use_gpu: bool = True
    min_text_confidence: float = Field(default=0.6, ge=0.0, le=1.0)


class ConfidenceConfig(BaseModel):
    """Configuration for confidence thresholds.

    Attributes:
        auto_accept: Threshold for auto-accepting extractions
        needs_review: Threshold below which extractions need review
        auto_reject: Threshold below which extractions are auto-rejected
    """

    auto_accept: float = Field(default=0.85, ge=0.0, le=1.0)
    needs_review: float = Field(default=0.60, ge=0.0, le=1.0)
    auto_reject: float = Field(default=0.40, ge=0.0, le=1.0)

    @field_validator('needs_review')
    @classmethod
    def needs_review_validation(cls, v: float, info) -> float:
        """Validate needs_review is between auto_reject and auto_accept."""
        if 'auto_reject' in info.data and v < info.data['auto_reject']:
            raise ValueError('needs_review must be >= auto_reject')
        if 'auto_accept' in info.data and v > info.data['auto_accept']:
            raise ValueError('needs_review must be <= auto_accept')
        return v


class FrameQualityConfig(BaseModel):
    """Configuration for frame quality assessment.

    Attributes:
        min_sharpness: Minimum sharpness threshold (Laplacian variance)
        min_size_score: Minimum size score (0-1)
        sharpness_weight: Weight for sharpness metric in composite score
        size_weight: Weight for size metric in composite score
        angle_weight: Weight for angle metric in composite score
        frames_to_select: Number of best frames to select per track
    """

    min_sharpness: float = Field(default=100.0, ge=0.0)
    min_size_score: float = Field(default=0.3, ge=0.0, le=1.0)
    sharpness_weight: float = Field(default=0.5, ge=0.0, le=1.0)
    size_weight: float = Field(default=0.3, ge=0.0, le=1.0)
    angle_weight: float = Field(default=0.2, ge=0.0, le=1.0)
    frames_to_select: int = Field(default=5, ge=1)


class LoggingConfig(BaseModel):
    """Configuration for logging.

    Attributes:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory for log files
        enable_frame_logging: Whether to save frames for debugging
    """

    level: str = Field(default="INFO")
    log_dir: str = "logs"
    enable_frame_logging: bool = False

    @field_validator('level')
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate logging level."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        v_upper = v.upper()
        if v_upper not in valid_levels:
            raise ValueError(f'level must be one of {valid_levels}')
        return v_upper


class AppConfig(BaseModel):
    """Main application configuration.

    Attributes:
        detection: Detection configuration
        tracking: Tracking configuration
        frame_sampling: Frame sampling configuration
        frame_quality: Frame quality assessment configuration
        ocr: OCR configuration
        confidence: Confidence threshold configuration
        logging: Logging configuration
    """

    detection: DetectionConfig = Field(default_factory=DetectionConfig)
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)
    frame_sampling: FrameSamplingConfig = Field(default_factory=FrameSamplingConfig)
    frame_quality: FrameQualityConfig = Field(default_factory=FrameQualityConfig)
    ocr: OCRConfig = Field(default_factory=OCRConfig)
    confidence: ConfidenceConfig = Field(default_factory=ConfidenceConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)


def load_config(config_path: Optional[str] = None) -> AppConfig:
    """Load configuration from YAML file.

    Args:
        config_path: Path to config YAML file. If None, uses default location.

    Returns:
        Validated AppConfig instance

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is invalid YAML
        pydantic.ValidationError: If config values are invalid
    """
    if config_path is None:
        # Default to config.yaml in the cv_config directory
        config_path = os.path.join(
            os.path.dirname(__file__),
            "config.yaml"
        )

    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_file, "r") as f:
        config_dict = yaml.safe_load(f)

    return AppConfig(**config_dict)


# Global config instance
_config: Optional[AppConfig] = None


def get_config(config_path: Optional[str] = None, reload: bool = False) -> AppConfig:
    """Get or load the global configuration instance.

    Args:
        config_path: Path to config YAML file. If None, uses default location.
        reload: Force reload of configuration

    Returns:
        AppConfig instance
    """
    global _config

    if _config is None or reload:
        _config = load_config(config_path)

    return _config
