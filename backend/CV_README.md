# PalleTrack Computer Vision Foundation

This document describes the core data structures and configuration system for the PalleTrack computer vision pipeline.

## Overview

The PalleTrack CV pipeline processes warehouse video feeds to:
1. Detect and track pallets using YOLOv8
2. Detect shipping documents on pallets
3. Extract text using OCR (PaddleOCR)
4. Parse extracted text into structured shipping data

This foundation provides the data models, configuration management, and logging infrastructure that all other components will use.

## Project Structure

```
backend/app/
├── cv_models/          # Core data models
│   ├── __init__.py
│   └── data_models.py
├── cv_config/          # Configuration management
│   ├── __init__.py
│   ├── config.py
│   └── config.yaml
└── cv_utils/           # Utilities
    ├── __init__.py
    └── logging_config.py

backend/tests/
└── test_data_models.py  # Unit tests
```

## Data Models

All data models are Pydantic BaseModel classes that provide:
- Type validation
- JSON serialization
- Clear documentation
- Utility methods

### BoundingBox

Represents a bounding box for detected objects.

```python
from app.cv_models import BoundingBox

bbox = BoundingBox(
    x1=100, y1=200,
    x2=400, y2=600,
    confidence=0.92
)

# Utility methods
area = bbox.area()                    # Calculate area
center_x, center_y = bbox.center()    # Get center point
is_inside = bbox.contains_point(x, y) # Check if point inside
iou_score = bbox.iou(other_bbox)      # Calculate IoU
```

**Key Methods:**
- `area()` - Calculate bounding box area
- `center()` - Get center coordinates
- `contains_point(x, y)` - Check if point is inside
- `iou(other)` - Calculate Intersection over Union
- `width()` - Get width
- `height()` - Get height

### PalletDetection

Single pallet detection in a frame.

```python
from app.cv_models import PalletDetection, BoundingBox

detection = PalletDetection(
    bbox=bbox,
    frame_number=42,
    timestamp=1.4,
    track_id=7  # Optional, assigned by tracker
)
```

### DocumentDetection

Document detected on a pallet.

```python
from app.cv_models import DocumentDetection, DocumentType

doc = DocumentDetection(
    bbox=bbox,
    frame_number=45,
    parent_pallet_track_id=7,
    document_type=DocumentType.SHIPPING_LABEL
)
```

**Document Types:**
- `BOL` - Bill of Lading
- `PACKING_LIST` - Packing List
- `SHIPPING_LABEL` - Shipping Label
- `UNKNOWN` - Unknown document type

### OCRResult

Text extracted from a document region.

```python
from app.cv_models import OCRResult

ocr = OCRResult(
    text="TRACKING: 1Z999AA10123456784",
    confidence=0.94,
    bbox=bbox,
    frame_number=45
)
```

### PalletTrack

Complete tracking information for a pallet across frames.

```python
from app.cv_models import PalletTrack, TrackStatus

track = PalletTrack(
    track_id=7,
    status=TrackStatus.ACTIVE,
    first_seen_frame=42,
    last_seen_frame=98,
    detections=[],          # List of PalletDetection
    document_regions=[],    # List of DocumentDetection
    ocr_results=[]          # List of OCRResult
)

# Utility methods
duration = track.duration_frames()       # Get tracking duration
ocr_text = track.get_best_ocr_text()    # Get OCR text sorted by confidence
```

**Track Status:**
- `ACTIVE` - Currently being tracked
- `COMPLETED` - Successfully tracked and processed
- `LOST` - Lost tracking

### ExtractedShippingData

Structured shipping data parsed from OCR results.

```python
from app.cv_models import ExtractedShippingData

data = ExtractedShippingData(
    track_id=7,
    document_type="SHIPPING_LABEL",
    tracking_number="1Z999AA10123456784",
    weight=45.5,
    destination_address="123 Main St, Springfield, IL 62701",
    destination_zip="62701",
    items=[
        {"description": "Widget A", "quantity": 100}
    ],
    confidence_score=0.89,
    needs_review=False
)
```

## Configuration

Configuration is managed via YAML files with Pydantic validation.

### Loading Configuration

```python
from app.cv_config import get_config

# Load default config
config = get_config()

# Load custom config
config = get_config("path/to/config.yaml")

# Reload config
config = get_config(reload=True)
```

### Configuration Structure

The configuration is organized into sections:

```yaml
detection:
  pallet_model_path: "models/pallet_yolov8.pt"
  document_model_path: "models/document_yolov8.pt"
  pallet_conf_threshold: 0.5
  document_conf_threshold: 0.6

tracking:
  max_age: 30              # Frames to keep track without detections
  min_hits: 3              # Detections needed to confirm track
  iou_threshold: 0.3       # IoU threshold for matching

frame_sampling:
  movement_threshold: 50.0          # Pixel movement to trigger sample
  size_change_threshold: 0.2        # Size change to trigger sample
  max_frame_gap: 30                 # Max frames between samples
  max_samples_per_track: 10         # Max samples per track

ocr:
  language: "en"
  use_gpu: true
  min_text_confidence: 0.6

confidence:
  auto_accept: 0.85        # Auto-accept threshold
  needs_review: 0.60       # Review threshold
  auto_reject: 0.40        # Auto-reject threshold

logging:
  level: "INFO"
  log_dir: "logs"
  enable_frame_logging: false
```

### Accessing Config Values

```python
config = get_config()

# Access nested values
pallet_threshold = config.detection.pallet_conf_threshold
max_age = config.tracking.max_age
log_level = config.logging.level
```

## Logging

The logging system provides structured logging with rotation, per-track logging, and performance metrics.

### Setup Logging

```python
from app.cv_utils import setup_logging

# Setup with defaults from config
setup_logging()

# Setup with custom parameters
setup_logging(
    log_dir="custom_logs",
    log_level="DEBUG",
    enable_frame_logging=True
)
```

### Using Loggers

```python
from app.cv_utils import get_logger

logger = get_logger(__name__)

logger.info("Processing frame 42")
logger.warning("Track 7 lost")
logger.error("Failed to load model")
```

### Per-Track Logging

Track specific activity for debugging individual pallets:

```python
from app.cv_utils.logging_config import TrackLogger

track_logger = TrackLogger(track_id=7, log_dir="logs")

track_logger.info("Track created")
track_logger.log_detection(
    frame_number=42,
    bbox=(100, 200, 400, 600),
    confidence=0.92
)
track_logger.log_ocr(
    frame_number=45,
    text="TRACKING: 1Z999...",
    confidence=0.94
)
```

Track logs are saved to: `logs/tracks/track_{track_id}.log`

### Performance Metrics

Log performance metrics for monitoring:

```python
from app.cv_utils.logging_config import log_performance_metrics

log_performance_metrics(
    logger=logger,
    fps=30.5,
    latency_ms=15.2,
    frame_number=1000,
    active_tracks=12
)
```

Performance logs are saved separately to: `logs/performance_{timestamp}.log`

### Debug Frame Logging

Save frames for debugging (when enabled in config):

```python
from app.cv_utils.logging_config import save_debug_frame
import cv2

frame = cv2.imread("frame.jpg")
path = save_debug_frame(
    frame=frame,
    frame_number=42,
    track_id=7,
    suffix="detection"
)
# Saves to: logs/frames/frame_000042_track_7_detection.jpg
```

## Log Files

The logging system creates the following structure:

```
logs/
├── palletrack_{timestamp}.log          # Main application log
├── performance_{timestamp}.log         # Performance metrics
├── tracks/                             # Per-track logs
│   ├── track_1.log
│   ├── track_2.log
│   └── ...
└── frames/                             # Debug frames (if enabled)
    ├── frame_000042_track_7.jpg
    └── ...
```

## Running Tests

```bash
cd backend

# Run all tests
pytest tests/test_data_models.py -v

# Run specific test class
pytest tests/test_data_models.py::TestBoundingBox -v

# Run with coverage
pytest tests/test_data_models.py --cov=app.cv_models
```

## Usage Examples

### Complete Pipeline Setup

```python
from app.cv_config import get_config
from app.cv_utils import setup_logging, get_logger
from app.cv_models import PalletTrack, TrackStatus

# 1. Load configuration
config = get_config()

# 2. Setup logging
setup_logging()
logger = get_logger(__name__)

# 3. Create a track
track = PalletTrack(
    track_id=1,
    status=TrackStatus.ACTIVE,
    first_seen_frame=0,
    last_seen_frame=0
)

logger.info(f"Created track {track.track_id}")
```

### Processing Detections

```python
from app.cv_models import PalletDetection, BoundingBox

# Create detection
bbox = BoundingBox(x1=100, y1=200, x2=400, y2=600, confidence=0.92)
detection = PalletDetection(
    bbox=bbox,
    frame_number=42,
    timestamp=1.4,
    track_id=1
)

# Add to track
track.detections.append(detection)
track.last_seen_frame = detection.frame_number

# Log it
logger.info(
    f"Track {track.track_id}: Detection at frame {detection.frame_number} "
    f"with confidence {detection.bbox.confidence:.2f}"
)
```

### Extracting Shipping Data

```python
from app.cv_models import ExtractedShippingData

# After OCR and parsing...
shipping_data = ExtractedShippingData(
    track_id=track.track_id,
    document_type="SHIPPING_LABEL",
    tracking_number="1Z999AA10123456784",
    weight=45.5,
    confidence_score=0.89,
    needs_review=False
)

# Serialize to JSON
json_output = shipping_data.model_dump_json(indent=2)

# Save or send to API
with open("output.json", "w") as f:
    f.write(json_output)
```

## Type Hints

All models use comprehensive type hints for IDE support:

```python
from typing import List, Optional
from app.cv_models import PalletTrack, PalletDetection

def process_track(track: PalletTrack) -> Optional[str]:
    """Process a track and return tracking number if found."""
    detections: List[PalletDetection] = track.detections
    # ... your code with full IDE autocomplete
```

## Next Steps

With the foundation in place, the next components to build are:

1. **Video Processing** (`services/video_processing.py`)
   - Video I/O and frame extraction
   - Frame preprocessing

2. **Pallet Detection** (`services/pallet_detection.py`)
   - YOLOv8 pallet detection
   - ByteTrack integration

3. **OCR Processing** (`services/ocr.py`)
   - PaddleOCR integration
   - Text extraction and confidence scoring

4. **Data Parsing** (`services/data_parser.py`)
   - Parse OCR text into structured data
   - Confidence scoring

5. **API Endpoints** (`api/routes/video.py`)
   - Video upload endpoint
   - Processing status endpoint
   - Results retrieval endpoint

## Contributing

When adding new models or config options:

1. Add the model/config to the appropriate file
2. Add validation using Pydantic validators
3. Add docstrings with examples
4. Add unit tests
5. Update this README

## License

See main project LICENSE file.
