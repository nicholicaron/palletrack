# PalletTrack Computer Vision Pipeline

**Automated Pallet Document Scanner for Warehouse Operations**

A complete computer vision pipeline that automatically detects pallets, tracks them through video feeds, locates shipping documents, performs OCR, and extracts structured shipping data with confidence scoring and quality assurance.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Pipeline Components](#pipeline-components)
- [Configuration](#configuration)
- [CLI Reference](#cli-reference)
- [Programmatic Usage](#programmatic-usage)
- [Testing](#testing)
- [Performance Tuning](#performance-tuning)
- [Troubleshooting](#troubleshooting)
- [Examples](#examples)

---

## Overview

PalletTrack processes warehouse video feeds to automatically extract shipping information from documents attached to pallets. The system:

1. **Detects & Tracks** pallets using YOLOv8 object detection and ByteTrack multi-object tracking
2. **Locates Documents** on tracked pallets with spatial association
3. **Intelligently Samples** frames based on movement, quality, and temporal spacing
4. **Performs OCR** on high-quality frames using PaddleOCR
5. **Extracts Data** into structured shipping information
6. **Scores Confidence** using multi-factor analysis
7. **Routes Results** for auto-acceptance, manual review, or rejection

### Key Benefits

- **Automated Processing**: Reduce manual data entry time by 90%+
- **High Accuracy**: Multi-frame OCR aggregation and quality scoring
- **Real-time Capable**: Process live video streams and RTSP feeds
- **Production Ready**: Comprehensive error handling, logging, and monitoring
- **Flexible Deployment**: CLI tool, Python API, or integration via REST

---

## Features

### Detection & Tracking
- YOLOv8-based pallet and document detection
- ByteTrack multi-object tracking with configurable thresholds
- Spatial association of documents to parent pallets
- Track lifecycle management (active → completed/lost)

### Intelligent Frame Processing
- Adaptive frame sampling based on movement and size changes
- Quality assessment (sharpness, size, viewing angle)
- Frame selection for optimal OCR results
- Configurable sampling strategies

### OCR & Text Extraction
- PaddleOCR integration with GPU/CPU support
- Preprocessing pipeline (CLAHE, bilateral filtering, glare removal)
- Multi-frame aggregation (voting, confidence weighting)
- Post-processing (error correction, whitespace normalization)

### Data Extraction & Classification
- Document type classification (BOL, Packing List, Shipping Label)
- Pattern-based field extraction (tracking numbers, addresses, weights, POs)
- Address parsing with city/state/zip extraction
- Line item extraction for packing lists

### Quality Assurance
- Multi-factor confidence scoring
- Three-tier routing (AUTO_ACCEPT, NEEDS_REVIEW, AUTO_REJECT)
- Review queue management with priority ordering
- Quality metrics tracking

### Integration & Export
- JSON, CSV, and WMS-formatted exports
- Review queue export for manual verification
- Real-time processing statistics
- Comprehensive logging and monitoring

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Video Input                               │
│              (File, Live Stream, RTSP)                          │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Frame-by-Frame Processing                      │
├─────────────────────────────────────────────────────────────────┤
│  1. Pallet Detection (YOLOv8)                                   │
│  2. Pallet Tracking (ByteTrack)                                 │
│  3. Document Detection (YOLOv8)                                 │
│  4. Document-Pallet Association                                 │
│  5. Adaptive Frame Sampling                                     │
│  6. Quality Assessment & Frame Selection                        │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                  Track Completion Processing                     │
├─────────────────────────────────────────────────────────────────┤
│  7. OCR Processing (PaddleOCR)                                  │
│     - Preprocessing (CLAHE, filtering, glare removal)           │
│     - Multi-frame aggregation (voting/confidence)               │
│     - Post-processing (error correction)                        │
│  8. Document Classification                                     │
│  9. Field Extraction & Parsing                                  │
│ 10. Data Validation                                             │
│ 11. Confidence Scoring                                          │
│ 12. Result Routing                                              │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Output & Integration                          │
├─────────────────────────────────────────────────────────────────┤
│  • Auto-Accepted Results → Export (JSON/CSV/WMS)                │
│  • Review Queue Items → Manual Verification                     │
│  • Rejected Items → Logging & Analysis                          │
│  • Performance Metrics → Monitoring Dashboard                   │
└─────────────────────────────────────────────────────────────────┘
```

### Project Structure

```
backend/app/
├── cv_models/                    # Data models & type definitions
│   ├── __init__.py
│   └── data_models.py            # BoundingBox, PalletTrack, OCRResult, etc.
│
├── cv_config/                    # Configuration management
│   ├── __init__.py
│   ├── config.py                 # Config loading & validation
│   └── config.yaml               # Main configuration file
│
├── cv_utils/                     # Utilities
│   ├── __init__.py
│   └── logging_config.py         # Structured logging setup
│
├── cv_processing/                # Core CV components
│   ├── __init__.py
│   ├── pallet_detector.py        # YOLOv8 pallet detection
│   ├── pallet_tracker.py         # ByteTrack wrapper
│   ├── document_detector.py      # Document detection & association
│   ├── document_utils.py         # Spatial relationship helpers
│   ├── movement_analysis.py      # Track movement analysis
│   ├── frame_sampler.py          # Adaptive frame sampling
│   ├── frame_selection.py        # Best frames selection
│   ├── frame_quality.py          # Quality assessment
│   ├── track_utils.py            # Track utilities
│   │
│   ├── ocr/                      # OCR processing
│   │   ├── __init__.py
│   │   ├── document_ocr.py       # Main OCR engine
│   │   ├── preprocessor.py       # Image preprocessing
│   │   ├── aggregator.py         # Multi-frame aggregation
│   │   └── post_processor.py     # OCR post-processing
│   │
│   ├── extraction/               # Data extraction
│   │   ├── __init__.py
│   │   ├── data_extractor.py     # Main extraction orchestrator
│   │   ├── document_classifier.py # Document type classification
│   │   ├── field_extractors.py   # Field extraction patterns
│   │   ├── address_extractor.py  # Address parsing
│   │   └── item_extractor.py     # Line item extraction
│   │
│   └── qa/                       # Quality assurance
│       ├── __init__.py
│       ├── confidence_calculator.py # Multi-factor confidence
│       ├── validators.py         # Data validation
│       ├── review_queue.py       # Review queue management
│       └── metrics_tracker.py    # Quality metrics tracking
│
├── pipeline/                     # Pipeline integration
│   ├── __init__.py
│   ├── main_pipeline.py          # Main orchestrator
│   ├── video_processor.py        # Video I/O & processing
│   ├── visualizer.py             # Frame annotation & visualization
│   ├── monitor.py                # Performance monitoring & logging
│   └── exporter.py               # Results export (JSON/CSV/WMS)
│
└── cli.py                        # Command-line interface

backend/tests/
├── test_data_models.py           # Data models tests
├── test_detection.py             # Detection tests
├── test_document_detection.py    # Document detection tests
├── test_pallet_tracker.py        # Tracking tests
├── test_frame_sampling.py        # Frame sampling tests
├── test_ocr.py                   # OCR tests
├── test_extraction.py            # Data extraction tests
└── test_integration.py           # Integration tests
```

---

## Installation

### Prerequisites

- Python 3.10+
- CUDA 11.8+ (optional, for GPU acceleration)

### Install Dependencies

```bash
cd backend

# CPU-only installation (default)
uv pip install -e .

# GPU-enabled installation (with PaddlePaddle GPU support)
uv pip install -e ".[gpu]"
```

### Download Models

Download YOLOv8 models and place them in the `models/` directory:

```bash
mkdir -p models

# Pallet detection model (train your own or use pretrained)
# Place at: models/pallet_yolov8n.pt

# Document detection model (train your own or use pretrained)
# Place at: models/document_yolov8.pt
```

### Verify Installation

```bash
# Run tests
pytest tests/ -v

# Check CLI
python -m app.cli --help
```

---

## Quick Start

### Process a Video File

```bash
# Basic processing with visualization
python -m app.cli input_video.mp4 --output-video annotated.mp4

# Process and export results to JSON
python -m app.cli input_video.mp4 \
  --output-json results.json \
  --output-video annotated.mp4

# Fast processing without visualization
python -m app.cli input_video.mp4 \
  --no-viz \
  --output-json results.json \
  --output-csv results.csv
```

### Process Live Stream

```bash
# Webcam (device 0)
python -m app.cli --live-stream 0 --duration 300

# RTSP stream
python -m app.cli --live-stream rtsp://camera-ip/stream \
  --output-json results.json
```

### Export Review Queue

```bash
python -m app.cli input_video.mp4 \
  --output-json auto_accepted.json \
  --export-review-queue review_queue.json \
  --export-wms wms_payloads.json
```

### View Statistics

```bash
python -m app.cli input_video.mp4 \
  --output-json results.json \
  --stats \
  --save-report detailed_report.json
```

---

## Pipeline Components

### 1. Pallet Detection (`pallet_detector.py`)

YOLOv8-based pallet detection with configurable confidence thresholds.

**Key Features:**
- GPU/CPU inference
- Confidence and size filtering
- Non-maximum suppression
- Efficient batch processing

**Configuration:**
```yaml
detection:
  pallet_model_path: "models/pallet_yolov8n.pt"
  pallet_conf_threshold: 0.5
  pallet_iou_threshold: 0.45
  device: "cuda"  # or "cpu"
  min_pallet_area: 10000
  max_pallet_area: 500000
```

### 2. Pallet Tracking (`pallet_tracker.py`)

ByteTrack-based multi-object tracking for pallet lifecycle management.

**Key Features:**
- ByteTrack algorithm integration
- Track state management (ACTIVE, LOST, COMPLETED)
- Configurable track confirmation and termination
- Track history and trail visualization

**Configuration:**
```yaml
tracking:
  max_age: 30              # Frames without detection before lost
  min_hits: 3              # Detections needed to confirm track
  iou_threshold: 0.3       # IoU threshold for association
  min_track_length: 10     # Min frames before processing
  frames_between_ocr: 15   # Min gap between OCR attempts
```

### 3. Document Detection (`document_detector.py`)

YOLOv8-based document detection with spatial association to parent pallets.

**Key Features:**
- Document type classification
- Containment-based association (document inside pallet bbox)
- Proximity-based fallback association
- Confidence scoring based on spatial relationship

**Configuration:**
```yaml
detection:
  document_model_path: "models/document_yolov8.pt"
  document_conf_threshold: 0.6
  max_association_distance: 100
  containment_confidence: 0.9
  proximity_confidence: 0.7
```

### 4. Adaptive Frame Sampling (`frame_sampler.py`)

Intelligent frame sampling based on movement, size changes, and temporal spacing.

**Key Features:**
- Movement-based triggers (track center displacement)
- Size-based triggers (bbox area changes)
- Velocity-adaptive sampling (faster for moving pallets)
- Periodic forced sampling (max gap enforcement)

**Configuration:**
```yaml
frame_sampling:
  movement_threshold: 50.0          # Pixels
  size_change_threshold: 0.2        # 20% change
  max_frame_gap: 30                 # Force sample every N frames
  max_samples_per_track: 10
  min_temporal_gap: 10              # Min frames between samples
  high_velocity_threshold: 20.0     # pixels/frame
  high_velocity_sample_rate: 5      # Sample every N frames
```

### 5. Frame Quality Assessment (`frame_quality.py`)

Quality scoring based on sharpness, size, and viewing angle.

**Key Features:**
- Laplacian variance sharpness measurement
- Relative size scoring
- Viewing angle estimation
- Weighted composite quality score

**Configuration:**
```yaml
frame_quality:
  min_sharpness: 100.0              # Laplacian variance threshold
  min_size_score: 0.3               # 30% of frame area
  sharpness_weight: 0.5
  size_weight: 0.3
  angle_weight: 0.2
  frames_to_select: 5               # Best N frames per track
```

### 6. OCR Processing (`ocr/`)

PaddleOCR-based text extraction with preprocessing and multi-frame aggregation.

**Components:**
- **Preprocessor**: CLAHE, bilateral filtering, glare removal
- **DocumentOCR**: Main OCR engine with GPU/CPU support
- **Aggregator**: Multi-frame result aggregation (voting, confidence weighting)
- **PostProcessor**: Error correction and normalization

**Configuration:**
```yaml
ocr:
  language: "en"
  use_gpu: true
  min_text_confidence: 0.6

  preprocessing:
    apply_clahe: true
    clahe_clip_limit: 2.0
    bilateral_filter: true
    remove_glare: true
    glare_threshold: 240

  aggregation:
    method: "voting"                    # voting, highest_confidence, longest
    min_frames_for_consensus: 2

  post_processing:
    fix_common_errors: true             # O→0, I→1, etc.
    normalize_whitespace: true
```

### 7. Data Extraction (`extraction/`)

Structured data extraction from OCR text.

**Components:**
- **DocumentClassifier**: Classify document type (BOL, Packing List, Shipping Label)
- **FieldExtractors**: Pattern-based extraction for tracking numbers, weights, POs
- **AddressExtractor**: Parse addresses with city/state/zip
- **ItemExtractor**: Extract line items from packing lists

**Configuration:**
```yaml
data_extraction:
  classification:
    min_confidence: 0.6
    require_all_keywords: false

  field_extraction:
    extract_items: true
    max_items: 100

  validation:
    require_tracking_number: true
    require_weight: false
    min_weight: 1.0
    max_weight: 5000.0
    validate_addresses: true
```

### 8. Quality Assurance (`qa/`)

Multi-factor confidence scoring and result routing.

**Components:**
- **ConfidenceCalculator**: Weighted multi-factor confidence scoring
- **Validators**: Data validation (format, range, consistency)
- **ReviewQueueManager**: Three-tier routing system
- **QualityMetricsTracker**: Quality metrics and reporting

**Confidence Factors:**
1. Detection confidence (bbox confidence scores)
2. OCR confidence (text recognition confidence)
3. Field completeness (percentage of required fields filled)
4. Cross-frame consistency (agreement across frames)
5. Data validation (format and range checks)

**Configuration:**
```yaml
confidence:
  auto_accept: 0.85                     # Auto-accept threshold
  needs_review: 0.60                    # Review threshold
  auto_reject: 0.40                     # Auto-reject threshold

  weights:
    detection_conf: 0.15
    ocr_conf: 0.25
    field_completeness: 0.25
    cross_frame_consistency: 0.20
    data_validation: 0.15

  validation:
    validate_tracking_format: true
    validate_zip_format: true
    detect_garbage_text: true
```

### 9. Review Queue (`qa/review_queue.py`)

Manage items requiring manual verification.

**Key Features:**
- Priority ordering (confidence, timestamp, document type)
- Frame saving for review context
- Queue size limits
- Export functionality

**Configuration:**
```yaml
review_queue:
  priority_order: "confidence"          # confidence, timestamp, importance
  max_queue_size: 1000
  save_frames: true
  frame_save_path: "review_frames"
```

### 10. Pipeline Integration (`pipeline/`)

Complete pipeline orchestration and integration.

**Components:**
- **PalletScannerPipeline**: Main orchestrator integrating all components
- **VideoStreamProcessor**: Video file and live stream processing
- **FrameAnnotator**: Visualization and annotation
- **PipelineMonitor**: Logging and performance monitoring
- **ResultsExporter**: Export results in multiple formats

---

## Configuration

### Configuration File

The main configuration file is `app/cv_config/config.yaml`. It contains all pipeline settings organized by component.

### Loading Configuration

```python
from app.cv_config import get_config

# Load default config
config = get_config()

# Load custom config
config = get_config("path/to/custom_config.yaml")

# Reload config
config = get_config(reload=True)
```

### Environment-Specific Configurations

Create environment-specific configs:

```bash
config/
├── config.yaml           # Default/production
├── config.dev.yaml       # Development
└── config.test.yaml      # Testing
```

Load with:
```bash
python -m app.cli video.mp4 --config config/config.dev.yaml
```

### Key Configuration Sections

See `app/cv_config/config.yaml` for the complete configuration template with detailed comments.

---

## CLI Reference

### Basic Usage

```bash
python -m app.cli [VIDEO_PATH] [OPTIONS]
```

### Input Options

- `video_path`: Path to input video file (positional, optional if using --live-stream)
- `--live-stream URL`: Live stream URL (RTSP, webcam index, etc.)
- `--duration SECONDS`: Duration to process live stream

### Configuration

- `--config PATH`: Path to configuration file (default: `app/cv_config/config.yaml`)

### Output Options

- `--output-video PATH`: Save annotated output video
- `--output-json PATH`: Export extracted data as JSON
- `--output-csv PATH`: Export extracted data as CSV
- `--export-review-queue PATH`: Export review queue to JSON
- `--export-wms PATH`: Export WMS-formatted payloads

### Processing Options

- `--no-viz`: Disable visualization (faster processing)
- `--display`: Display video during processing (for debugging)
- `--skip-frames N`: Process every Nth frame (0 = all frames)

### Logging Options

- `--log-dir PATH`: Directory for log files (default: `logs`)
- `--log-level LEVEL`: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

### Statistics & Reporting

- `--stats`: Print detailed statistics after processing
- `--save-report PATH`: Save detailed processing report to JSON

### Examples

```bash
# Process with all outputs
python -m app.cli warehouse_feed.mp4 \
  --output-video annotated.mp4 \
  --output-json results.json \
  --output-csv results.csv \
  --export-review-queue review.json \
  --stats

# Fast processing (no visualization)
python -m app.cli warehouse_feed.mp4 \
  --no-viz \
  --output-json results.json \
  --skip-frames 2

# Live stream processing
python -m app.cli --live-stream rtsp://192.168.1.100/stream \
  --duration 3600 \
  --output-json hourly_results.json \
  --log-level DEBUG

# Debug mode with display
python -m app.cli test_video.mp4 \
  --display \
  --log-level DEBUG \
  --save-report debug_report.json
```

---

## Programmatic Usage

### Basic Pipeline Usage

```python
from app.pipeline import (
    PalletScannerPipeline,
    VideoStreamProcessor,
    PipelineMonitor,
    ResultsExporter
)

# Initialize components
pipeline = PalletScannerPipeline(config_path="app/cv_config/config.yaml")
monitor = PipelineMonitor(log_dir="logs", log_level="INFO")
processor = VideoStreamProcessor(pipeline, monitor)

# Process video
summary = processor.process_video_file(
    video_path="warehouse_feed.mp4",
    output_video_path="annotated.mp4",
    visualize=True,
    display=False
)

# Export results
extractions = pipeline.completed_extractions
ResultsExporter.export_to_json(extractions, "results.json")
ResultsExporter.export_to_csv(extractions, "results.csv")

# Get review queue
review_queue = pipeline.get_review_queue()
ResultsExporter.export_review_queue(review_queue, "review_queue.json")

# Print statistics
monitor.print_summary()
```

### Frame-by-Frame Processing

```python
import cv2
from app.pipeline import PalletScannerPipeline

pipeline = PalletScannerPipeline(config_path="app/cv_config/config.yaml")

cap = cv2.VideoCapture("warehouse_feed.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)
frame_number = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    timestamp = frame_number / fps

    # Process single frame
    result = pipeline.process_frame(frame, frame_number, timestamp)

    # Check for completed extractions
    if result['completed_extractions']:
        for extraction in result['completed_extractions']:
            print(f"Extracted: {extraction.tracking_number}")

    frame_number += 1

cap.release()

# Finalize pipeline (process remaining tracks)
pipeline.finalize()

# Get all results
all_extractions = pipeline.completed_extractions
print(f"Total extractions: {len(all_extractions)}")
```

### Using Individual Components

```python
from app.cv_processing import PalletDetector, PalletTracker, DocumentDetector
from app.cv_processing.ocr import DocumentOCR
from app.cv_config import get_config

# Load config
config = get_config()

# Initialize components
pallet_detector = PalletDetector(
    model_path=config.detection.pallet_model_path,
    conf_threshold=config.detection.pallet_conf_threshold,
    device=config.detection.device
)

tracker = PalletTracker(
    max_age=config.tracking.max_age,
    min_hits=config.tracking.min_hits,
    iou_threshold=config.tracking.iou_threshold
)

doc_detector = DocumentDetector(
    model_path=config.detection.document_model_path,
    conf_threshold=config.detection.document_conf_threshold,
    device=config.detection.device
)

ocr_engine = DocumentOCR(config=config.ocr)

# Process frame
import cv2
frame = cv2.imread("frame.jpg")

# Detect pallets
pallet_detections = pallet_detector.detect(frame)

# Update tracker
active_tracks = tracker.update(pallet_detections)

# Detect documents
document_detections = doc_detector.detect_and_associate(frame, active_tracks)

# Perform OCR on document region
for doc_det in document_detections:
    x1, y1, x2, y2 = int(doc_det.bbox.x1), int(doc_det.bbox.y1), int(doc_det.bbox.x2), int(doc_det.bbox.y2)
    doc_region = frame[y1:y2, x1:x2]

    ocr_result = ocr_engine.process_region(doc_region)
    print(f"OCR Text: {ocr_result.text}")
```

### Custom Export Format

```python
from app.pipeline import ResultsExporter
from app.cv_models import ExtractedShippingData

def export_custom_format(extractions: list[ExtractedShippingData], output_path: str):
    """Custom export format example."""
    custom_data = []

    for extraction in extractions:
        custom_data.append({
            'pallet_id': extraction.track_id,
            'tracking': extraction.tracking_number,
            'weight_lbs': extraction.weight,
            'destination': f"{extraction.destination_city}, {extraction.destination_state}",
            'confidence': extraction.confidence_score,
            'status': 'APPROVED' if not extraction.needs_review else 'REVIEW'
        })

    import json
    with open(output_path, 'w') as f:
        json.dump(custom_data, f, indent=2)

# Use it
extractions = pipeline.completed_extractions
export_custom_format(extractions, "custom_output.json")
```

---

## Testing

### Run All Tests

```bash
cd backend

# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=app.cv_processing --cov=app.pipeline --cov-report=html

# Run specific test file
pytest tests/test_integration.py -v

# Run specific test class
pytest tests/test_pallet_tracker.py::TestPalletTracker -v

# Run specific test method
pytest tests/test_ocr.py::TestDocumentOCR::test_preprocess_image -v
```

### Test Categories

- `test_data_models.py`: Data model validation and utilities
- `test_detection.py`: Pallet detection tests
- `test_document_detection.py`: Document detection and association
- `test_pallet_tracker.py`: Tracking algorithm tests
- `test_frame_sampling.py`: Frame sampling logic
- `test_ocr.py`: OCR preprocessing and aggregation
- `test_extraction.py`: Data extraction and classification
- `test_integration.py`: End-to-end pipeline tests

### Integration Testing

Integration tests use mocked models to avoid requiring actual YOLOv8 weights:

```python
import pytest
from unittest.mock import patch

@patch('app.pipeline.main_pipeline.PalletDetector')
@patch('app.pipeline.main_pipeline.PalletTracker')
def test_pipeline_initialization(mock_tracker, mock_detector, tmp_path):
    pipeline = PalletScannerPipeline(str(config_path))
    assert pipeline.config is not None
```

### Performance Testing

```bash
# Time video processing
time python -m app.cli test_video.mp4 --no-viz --output-json results.json

# Profile with cProfile
python -m cProfile -o profile.stats -m app.cli test_video.mp4 --no-viz

# Analyze profile
python -c "import pstats; p = pstats.Stats('profile.stats'); p.sort_stats('cumulative'); p.print_stats(20)"
```

---

## Performance Tuning

### GPU Acceleration

Enable GPU for detection and OCR:

```yaml
detection:
  device: "cuda"

ocr:
  use_gpu: true
```

Install GPU-enabled PaddlePaddle:
```bash
uv pip install -e ".[gpu]"
```

### Frame Skipping

Process fewer frames for faster throughput:

```bash
python -m app.cli video.mp4 --skip-frames 2  # Process every 2nd frame
```

Or adjust sampling parameters in config:
```yaml
frame_sampling:
  max_frame_gap: 15           # Sample less frequently
  max_samples_per_track: 5    # Fewer samples per track
```

### Disable Visualization

Visualization adds ~20-30% overhead:

```bash
python -m app.cli video.mp4 --no-viz --output-json results.json
```

### Optimize OCR Settings

Reduce OCR preprocessing for speed:

```yaml
ocr:
  preprocessing:
    apply_clahe: false
    bilateral_filter: false
    remove_glare: false
```

### Adjust Tracking Parameters

Reduce tracking overhead:

```yaml
tracking:
  min_track_length: 5         # Process tracks sooner
  frames_between_ocr: 10      # OCR more frequently (fewer samples needed)
```

### Buffer Management

Limit memory usage:

```yaml
performance:
  max_frame_buffer_size: 20   # Reduce buffer size
  cleanup_interval: 50        # Clean up more frequently
```

### Expected Performance

**GPU Configuration (RTX 3080):**
- Video processing: 30-60 FPS (1080p)
- OCR per document: ~50-100ms
- Total pipeline latency: ~100-200ms per frame

**CPU Configuration (8-core Intel i7):**
- Video processing: 10-20 FPS (1080p)
- OCR per document: ~200-400ms
- Total pipeline latency: ~300-500ms per frame

---

## Troubleshooting

### Common Issues

#### 1. Models Not Found

**Error:** `FileNotFoundError: models/pallet_yolov8n.pt`

**Solution:**
```bash
# Ensure models directory exists
mkdir -p models

# Download or copy your trained models
cp /path/to/trained/model.pt models/pallet_yolov8n.pt
```

Update config paths:
```yaml
detection:
  pallet_model_path: "models/pallet_yolov8n.pt"
  document_model_path: "models/document_yolov8.pt"
```

#### 2. CUDA Out of Memory

**Error:** `RuntimeError: CUDA out of memory`

**Solution:**
- Reduce batch size (currently 1, already minimal)
- Use CPU instead: `device: "cpu"`
- Close other GPU applications
- Use smaller YOLOv8 model (yolov8n instead of yolov8x)

#### 3. PaddleOCR Import Error

**Error:** `ImportError: cannot import name 'PaddleOCR'`

**Solution:**
```bash
# Reinstall PaddleOCR
pip uninstall paddleocr paddlepaddle -y
uv pip install paddlepaddle>=2.6.0 paddleocr>=2.7.0
```

#### 4. Low Confidence Scores

**Issue:** Most extractions end up in review queue

**Solution:**
- Check OCR quality with `--display` flag
- Adjust preprocessing settings:
  ```yaml
  ocr:
    preprocessing:
      apply_clahe: true
      remove_glare: true
  ```
- Lower confidence thresholds temporarily:
  ```yaml
  confidence:
    needs_review: 0.50  # Lower from 0.60
  ```
- Improve lighting in video feed
- Retrain document detection model

#### 5. No Documents Detected

**Issue:** Pallets tracked but no documents found

**Solution:**
- Verify document model path
- Lower document confidence threshold:
  ```yaml
  detection:
    document_conf_threshold: 0.4  # Lower from 0.6
  ```
- Check if documents are visible in frames
- Retrain document detection model with more diverse data

#### 6. Poor OCR Results

**Issue:** OCR extracting gibberish or incorrect text

**Solution:**
- Enable all preprocessing:
  ```yaml
  ocr:
    preprocessing:
      apply_clahe: true
      bilateral_filter: true
      remove_glare: true
  ```
- Increase quality thresholds:
  ```yaml
  frame_quality:
    min_sharpness: 150.0      # Increase from 100
    min_size_score: 0.4       # Increase from 0.3
  ```
- Use more frames for aggregation:
  ```yaml
  frame_quality:
    frames_to_select: 8       # Increase from 5
  ```
- Check video resolution and document size

### Logging & Debugging

Enable debug logging:

```bash
python -m app.cli video.mp4 --log-level DEBUG --display
```

Check log files:
```bash
# Main application log
tail -f logs/pipeline_*.log

# Performance metrics
tail -f logs/performance_*.log
```

Save detailed report:
```bash
python -m app.cli video.mp4 --save-report debug_report.json
```

Inspect review queue:
```bash
python -m app.cli video.mp4 --export-review-queue review.json
```

### Performance Issues

Profile the pipeline:

```bash
# Time each component
python -m cProfile -o profile.stats -m app.cli video.mp4 --no-viz

# View top time consumers
python -c "import pstats; p = pstats.Stats('profile.stats'); p.sort_stats('cumulative'); p.print_stats(30)"
```

Common bottlenecks:
- **OCR preprocessing**: Disable unnecessary steps
- **Frame sampling**: Reduce `max_samples_per_track`
- **Visualization**: Use `--no-viz`
- **Model inference**: Use GPU or smaller models

---

## Examples

### Example 1: Batch Video Processing

```python
"""Process multiple videos in batch."""
from pathlib import Path
from app.pipeline import PalletScannerPipeline, VideoStreamProcessor, PipelineMonitor

def batch_process_videos(video_dir: str, output_dir: str):
    """Process all videos in directory."""
    video_dir = Path(video_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Initialize pipeline once
    pipeline = PalletScannerPipeline("app/cv_config/config.yaml")
    monitor = PipelineMonitor(log_dir="logs/batch", log_level="INFO")
    processor = VideoStreamProcessor(pipeline, monitor)

    # Process each video
    for video_path in video_dir.glob("*.mp4"):
        print(f"Processing: {video_path.name}")

        # Process video
        summary = processor.process_video_file(
            video_path=str(video_path),
            visualize=False
        )

        # Export results
        output_json = output_dir / f"{video_path.stem}_results.json"
        ResultsExporter.export_to_json(
            pipeline.completed_extractions,
            str(output_json)
        )

        print(f"  → {len(pipeline.completed_extractions)} extractions")

        # Reset pipeline for next video
        pipeline.active_tracks.clear()
        pipeline.completed_extractions.clear()

# Run batch processing
batch_process_videos("videos/", "results/")
```

### Example 2: Real-time Stream Monitoring

```python
"""Monitor live stream and send alerts."""
import cv2
from app.pipeline import PalletScannerPipeline

def monitor_live_stream(stream_url: str, alert_callback):
    """Monitor live stream and trigger alerts."""
    pipeline = PalletScannerPipeline("app/cv_config/config.yaml")

    cap = cv2.VideoCapture(stream_url)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_number = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        timestamp = frame_number / fps
        result = pipeline.process_frame(frame, frame_number, timestamp)

        # Check for new extractions
        for extraction in result['completed_extractions']:
            if extraction.needs_review:
                # Alert for manual review
                alert_callback({
                    'type': 'needs_review',
                    'track_id': extraction.track_id,
                    'confidence': extraction.confidence_score,
                    'tracking_number': extraction.tracking_number
                })
            else:
                # Auto-accepted
                alert_callback({
                    'type': 'auto_accepted',
                    'track_id': extraction.track_id,
                    'tracking_number': extraction.tracking_number
                })

        frame_number += 1

        # Break on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    pipeline.finalize()

def alert_handler(alert_data):
    """Handle alerts."""
    if alert_data['type'] == 'needs_review':
        print(f"⚠️  Review needed: Track {alert_data['track_id']}")
        # Send email, webhook, etc.
    else:
        print(f"✅ Auto-accepted: {alert_data['tracking_number']}")

# Monitor stream
monitor_live_stream("rtsp://camera/stream", alert_handler)
```

### Example 3: WMS Integration

```python
"""Integrate with warehouse management system."""
import requests
from app.pipeline import PalletScannerPipeline, ResultsExporter

def process_and_send_to_wms(video_path: str, wms_api_url: str, api_key: str):
    """Process video and send results to WMS API."""
    from app.pipeline import VideoStreamProcessor, PipelineMonitor

    # Process video
    pipeline = PalletScannerPipeline("app/cv_config/config.yaml")
    monitor = PipelineMonitor()
    processor = VideoStreamProcessor(pipeline, monitor)

    summary = processor.process_video_file(video_path, visualize=False)

    # Get auto-accepted results
    auto_accepted = [
        e for e in pipeline.completed_extractions
        if not e.needs_review
    ]

    print(f"Sending {len(auto_accepted)} items to WMS...")

    # Send each item to WMS
    for extraction in auto_accepted:
        # Create WMS payload
        payload = ResultsExporter.create_wms_payload(extraction)

        # Send to WMS API
        response = requests.post(
            wms_api_url,
            json=payload,
            headers={'Authorization': f'Bearer {api_key}'}
        )

        if response.status_code == 200:
            print(f"  ✅ Track {extraction.track_id}: {extraction.tracking_number}")
        else:
            print(f"  ❌ Track {extraction.track_id}: Failed ({response.status_code})")

    # Save review queue for manual processing
    review_queue = pipeline.get_review_queue()
    ResultsExporter.export_review_queue(review_queue, "wms_review_queue.json")
    print(f"Review queue saved: {len(review_queue)} items")

# Run integration
process_and_send_to_wms(
    "warehouse_feed.mp4",
    "https://wms.example.com/api/v1/pallet-scan",
    "your-api-key"
)
```

### Example 4: Quality Metrics Dashboard

```python
"""Generate quality metrics report."""
from app.pipeline import PalletScannerPipeline, VideoStreamProcessor, PipelineMonitor

def generate_quality_report(video_path: str):
    """Generate comprehensive quality report."""
    pipeline = PalletScannerPipeline("app/cv_config/config.yaml")
    monitor = PipelineMonitor(log_dir="logs", log_level="INFO")
    processor = VideoStreamProcessor(pipeline, monitor)

    # Process video
    summary = processor.process_video_file(video_path, visualize=False)

    # Get quality report
    quality_report = pipeline.get_quality_report()

    # Print formatted report
    print("\n" + "=" * 70)
    print("QUALITY METRICS REPORT")
    print("=" * 70)

    summary_data = quality_report['summary']
    print(f"\nTotal Processed: {summary_data['total_processed']}")
    print(f"Auto-Accept Rate: {summary_data['auto_acceptance_rate']:.1%}")
    print(f"Review Rate: {summary_data['review_rate']:.1%}")
    print(f"Rejection Rate: {summary_data['rejection_rate']:.1%}")

    performance = quality_report['performance']
    print(f"\nAverage Confidence: {performance['avg_confidence']:.2f}")
    print(f"Confidence Range: {performance['min_confidence']:.2f} - {performance['max_confidence']:.2f}")

    if summary_data['review_rate'] > 0.3:
        print("\n⚠️  WARNING: High review rate (>30%)")
        print("Consider:")
        print("  - Improving video quality")
        print("  - Retraining models")
        print("  - Adjusting confidence thresholds")

    # Save detailed report
    monitor.save_detailed_report("quality_report.json")
    print("\nDetailed report saved to: quality_report.json")

generate_quality_report("warehouse_feed.mp4")
```

---

## Data Models Reference

### Key Models

See `app/cv_models/data_models.py` for complete model definitions.

**BoundingBox:**
- `x1, y1, x2, y2`: Coordinates
- `confidence`: Detection confidence (0-1)
- Methods: `area()`, `center()`, `iou()`, `contains_point()`

**PalletDetection:**
- `bbox`: BoundingBox
- `frame_number`: Frame index
- `timestamp`: Video timestamp
- `track_id`: Assigned by tracker

**DocumentDetection:**
- `bbox`: BoundingBox
- `frame_number`: Frame index
- `parent_pallet_track_id`: Associated pallet track
- `document_type`: DocumentType enum
- `association_confidence`: Spatial association confidence

**PalletTrack:**
- `track_id`: Unique identifier
- `status`: TrackStatus (ACTIVE, COMPLETED, LOST)
- `first_seen_frame`, `last_seen_frame`: Lifecycle
- `detections`: List[PalletDetection]
- `document_regions`: List[DocumentDetection]
- `ocr_results`: List[OCRResult]

**OCRResult:**
- `text`: Extracted text
- `confidence`: OCR confidence
- `bbox`: Text bounding box
- `frame_number`: Source frame

**ExtractedShippingData:**
- `track_id`: Source track
- `document_type`: Classified type
- `tracking_number`, `po_number`: Identifiers
- `weight`, `declared_value`: Numeric fields
- `destination_*`, `origin_*`: Address fields
- `carrier`, `service_type`: Shipping info
- `items`: Line items (for packing lists)
- `confidence_score`: Overall confidence
- `needs_review`: Routing flag
- `timestamp`: Extraction time

---

## Contributing

### Development Setup

```bash
# Clone repository
git clone https://github.com/your-org/palletrack.git
cd palletrack/backend

# Install with dev dependencies
uv pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### Code Style

- Follow PEP 8
- Use type hints
- Add docstrings to all public methods
- Maximum line length: 100 characters

### Adding New Components

1. Create component in appropriate directory (`cv_processing/`, `extraction/`, etc.)
2. Add Pydantic models in `cv_models/` if needed
3. Add configuration parameters to `config.yaml`
4. Write unit tests in `tests/`
5. Update this README
6. Submit pull request

### Running Checks

```bash
# Type checking
mypy app/

# Linting
ruff check app/

# Format code
ruff format app/

# Run tests
pytest tests/ -v --cov
```

---

## License

See main project LICENSE file.

---

## Support

- **Issues**: https://github.com/your-org/palletrack/issues
- **Documentation**: https://docs.palletrack.io
- **Email**: support@palletrack.io

---

## Acknowledgments

- **YOLOv8**: Ultralytics (https://github.com/ultralytics/ultralytics)
- **ByteTrack**: ByteDance (https://github.com/ifzhang/ByteTrack)
- **PaddleOCR**: PaddlePaddle (https://github.com/PaddlePaddle/PaddleOCR)
- **Supervision**: Roboflow (https://github.com/roboflow/supervision)

---

**Built with ❤️ for warehouse automation**
