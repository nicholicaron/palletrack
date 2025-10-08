"""Main pipeline orchestrator that integrates all CV components."""

import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import yaml

from ..cv_models import ExtractedShippingData, PalletTrack, TrackStatus
from ..cv_processing import (
    AdaptiveFrameSampler,
    DocumentDetector,
    DocumentOCR,
    PalletDetector,
    PalletTracker,
    ShippingDataExtractor,
)
from ..cv_processing.qa import ConfidenceCalculator, ReviewQueueManager, QualityMetricsTracker


class PalletScannerPipeline:
    """Main orchestrator for the complete pallet scanning pipeline."""

    def __init__(self, config_path: str = "app/cv_config/config.yaml"):
        """Initialize all pipeline components.

        Args:
            config_path: Path to configuration YAML file
        """
        self.config = self._load_config(config_path)
        self._initialize_components()

        # State management
        self.frame_buffer: Dict[int, Dict[int, np.ndarray]] = defaultdict(dict)
        self.document_frame_buffer: Dict[int, Dict[int, np.ndarray]] = defaultdict(dict)
        self.active_tracks: Dict[int, PalletTrack] = {}
        self.completed_extractions: List[ExtractedShippingData] = []

        # OCR results storage (track_id -> list of frame OCR results)
        self.track_ocr_results: Dict[int, List] = defaultdict(list)

        # Track when we last processed OCR for each track
        self.last_ocr_frame: Dict[int, int] = {}

        # Statistics
        self.stats = {
            'total_frames_processed': 0,
            'total_pallets_tracked': 0,
            'total_documents_detected': 0,
            'total_ocr_runs': 0,
            'total_extractions': 0,
        }

    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file.

        Args:
            config_path: Path to YAML configuration file

        Returns:
            Configuration dictionary
        """
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config

    def _initialize_components(self):
        """Initialize all pipeline components."""
        # Detection & Tracking
        self.pallet_detector = PalletDetector(self.config)
        self.pallet_tracker = PalletTracker(self.config)
        self.document_detector = DocumentDetector(self.config)

        # Frame Sampling
        self.frame_sampler = AdaptiveFrameSampler(self.config)

        # OCR & Extraction
        self.ocr_processor = DocumentOCR(self.config)
        self.data_extractor = ShippingDataExtractor(self.config)

        # QA & Confidence
        self.confidence_calculator = ConfidenceCalculator(self.config)
        self.review_queue_manager = ReviewQueueManager(self.config)
        self.metrics_tracker = QualityMetricsTracker()

        # Pipeline parameters from config
        tracking_config = self.config.get('tracking', {})
        self.min_track_length = tracking_config.get('min_track_length', 10)
        self.frames_between_ocr = tracking_config.get('frames_between_ocr', 15)

    def process_frame(
        self, frame: np.ndarray, frame_number: int, timestamp: float
    ) -> Dict:
        """Process single video frame through complete pipeline.

        Pipeline steps:
        1. Detect pallets
        2. Update tracker
        3. Detect documents on tracked pallets
        4. Decide if frame should be processed for OCR (adaptive sampling)
        5. If yes: extract document regions, run OCR
        6. Check if any tracks are complete and ready for data extraction
        7. Extract data from completed tracks
        8. Calculate confidence and route to appropriate queue

        Args:
            frame: Input frame (BGR format)
            frame_number: Frame index in video
            timestamp: Timestamp in seconds

        Returns:
            Dictionary with processing results and statistics
        """
        result = {
            'frame_number': frame_number,
            'timestamp': timestamp,
            'pallet_detections': [],
            'active_tracks': {},
            'document_detections': [],
            'ocr_processed': False,
            'completed_extractions': [],
        }

        # Step 1 & 2: Detect and track pallets
        pallet_detections, active_tracks = self._detect_and_track(
            frame, frame_number, timestamp
        )
        result['pallet_detections'] = pallet_detections
        result['active_tracks'] = active_tracks

        # Step 3: Detect documents on tracked pallets
        document_detections = self._detect_documents(frame, frame_number, active_tracks)
        result['document_detections'] = document_detections

        # Step 4 & 5: Adaptive frame sampling and OCR processing
        ocr_results = self._process_ocr_frames(
            frame, frame_number, active_tracks, document_detections
        )
        if ocr_results:
            result['ocr_processed'] = True

        # Step 6 & 7: Process completed tracks
        completed_extractions = self._process_completed_tracks()
        result['completed_extractions'] = completed_extractions

        # Update statistics
        self.stats['total_frames_processed'] += 1

        return result

    def _detect_and_track(
        self, frame: np.ndarray, frame_number: int, timestamp: float
    ) -> tuple:
        """Detection and tracking substep.

        Args:
            frame: Input frame
            frame_number: Frame index
            timestamp: Timestamp in seconds

        Returns:
            Tuple of (pallet_detections, active_tracks)
        """
        # Detect pallets
        pallet_detections = self.pallet_detector.detect(frame)

        # Update tracker
        active_tracks = self.pallet_tracker.update(
            pallet_detections, frame_number, timestamp
        )

        # Store active tracks
        self.active_tracks = {track.track_id: track for track in active_tracks}

        # Update stats
        self.stats['total_pallets_tracked'] = len(self.active_tracks)

        # Store frames for later OCR processing
        for track in active_tracks:
            if track.status == TrackStatus.TRACKED:
                self._store_frame_for_track(track.track_id, frame_number, frame)

        return pallet_detections, self.active_tracks

    def _detect_documents(
        self, frame: np.ndarray, frame_number: int, active_tracks: Dict[int, PalletTrack]
    ) -> List:
        """Document detection substep.

        Args:
            frame: Input frame
            frame_number: Frame index
            active_tracks: Dictionary of active pallet tracks

        Returns:
            List of document detections
        """
        if not active_tracks:
            return []

        # Get list of pallet tracks for document detection
        track_list = list(active_tracks.values())

        # Detect documents
        document_detections = self.document_detector.detect_documents(
            frame, track_list, frame_number
        )

        # Update stats
        self.stats['total_documents_detected'] += len(document_detections)

        return document_detections

    def _process_ocr_frames(
        self,
        frame: np.ndarray,
        frame_number: int,
        active_tracks: Dict[int, PalletTrack],
        document_detections: List,
    ) -> Dict:
        """OCR processing substep (only for sampled frames).

        Args:
            frame: Input frame
            frame_number: Frame index
            active_tracks: Dictionary of active tracks
            document_detections: List of document detections

        Returns:
            Dictionary of OCR results by track_id
        """
        ocr_results_by_track = {}

        for track_id, track in active_tracks.items():
            # Check if track meets minimum length requirement
            if len(track.detections) < self.min_track_length:
                continue

            # Check if enough frames have passed since last OCR
            last_ocr = self.last_ocr_frame.get(track_id, -999)
            if frame_number - last_ocr < self.frames_between_ocr:
                continue

            # Check if this frame should be sampled for OCR
            should_sample = self.frame_sampler.should_sample_frame(track, frame_number)

            if should_sample:
                # Find document detection for this track
                doc_detection = None
                for doc in document_detections:
                    if doc.parent_pallet_track_id == track_id:
                        doc_detection = doc
                        break

                if doc_detection:
                    # Extract document region
                    bbox = doc_detection.bbox
                    x1, y1 = int(bbox.x1), int(bbox.y1)
                    x2, y2 = int(bbox.x2), int(bbox.y2)

                    # Ensure coordinates are within frame bounds
                    h, w = frame.shape[:2]
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)

                    if x2 > x1 and y2 > y1:
                        document_region = frame[y1:y2, x1:x2]

                        # Store document frame for later best-frame selection
                        self._store_document_frame(track_id, frame_number, document_region)

                        # Run OCR
                        ocr_results = self.ocr_processor.process_document(document_region)

                        # Store OCR results for this track
                        self.track_ocr_results[track_id].append(ocr_results)

                        ocr_results_by_track[track_id] = ocr_results

                        # Update last OCR frame
                        self.last_ocr_frame[track_id] = frame_number

                        # Update stats
                        self.stats['total_ocr_runs'] += 1

        return ocr_results_by_track

    def _process_completed_tracks(self) -> List[ExtractedShippingData]:
        """Extract data from tracks that have exited frame or are lost.

        Returns:
            List of newly completed extractions
        """
        completed_extractions = []

        # Check for tracks that have been marked as lost
        tracks_to_process = []
        for track_id, track in list(self.active_tracks.items()):
            if track.status == TrackStatus.LOST:
                tracks_to_process.append(track_id)

        # Process each completed track
        for track_id in tracks_to_process:
            track = self.active_tracks[track_id]

            # Check if we have OCR results for this track
            ocr_results = self.track_ocr_results.get(track_id, [])

            if not ocr_results:
                # No OCR results - skip this track
                del self.active_tracks[track_id]
                continue

            # Extract shipping data
            extracted_data = self.data_extractor.extract_shipping_data(
                track, ocr_results
            )

            # Calculate confidence
            overall_confidence, confidence_breakdown = (
                self.confidence_calculator.calculate_confidence(
                    track, extracted_data, ocr_results
                )
            )

            # Update extracted data with confidence
            extracted_data.confidence_score = overall_confidence

            # Route extraction based on confidence
            route = self.review_queue_manager.route_extraction(
                extracted_data, confidence_breakdown
            )

            # Add to review queue if needed
            if route == 'NEEDS_REVIEW':
                # Get best frame for review
                best_frame = self._get_best_document_frame(track_id)
                self.review_queue_manager.add_to_review_queue(
                    extracted_data, confidence_breakdown, best_frame
                )
                extracted_data.needs_review = True

            # Record metrics
            processing_time = 0.0  # Placeholder - could track actual time
            self.metrics_tracker.record_extraction(
                route, overall_confidence, processing_time, extracted_data.dict()
            )

            # Store completed extraction
            self.completed_extractions.append(extracted_data)
            completed_extractions.append(extracted_data)

            # Cleanup
            del self.active_tracks[track_id]
            if track_id in self.frame_buffer:
                del self.frame_buffer[track_id]
            if track_id in self.document_frame_buffer:
                del self.document_frame_buffer[track_id]
            if track_id in self.track_ocr_results:
                del self.track_ocr_results[track_id]
            if track_id in self.last_ocr_frame:
                del self.last_ocr_frame[track_id]

            # Update stats
            self.stats['total_extractions'] += 1

        return completed_extractions

    def _store_frame_for_track(
        self, track_id: int, frame_number: int, frame: np.ndarray
    ):
        """Store frame for later best-frame selection.

        Args:
            track_id: Track ID
            frame_number: Frame index
            frame: Frame image
        """
        # Store copy of frame
        self.frame_buffer[track_id][frame_number] = frame.copy()

        # Limit buffer size (keep last N frames)
        max_buffer_size = 50
        if len(self.frame_buffer[track_id]) > max_buffer_size:
            # Remove oldest frame
            oldest_frame = min(self.frame_buffer[track_id].keys())
            del self.frame_buffer[track_id][oldest_frame]

    def _store_document_frame(
        self, track_id: int, frame_number: int, document_region: np.ndarray
    ):
        """Store document region for later best-frame selection.

        Args:
            track_id: Track ID
            frame_number: Frame index
            document_region: Document region image
        """
        self.document_frame_buffer[track_id][frame_number] = document_region.copy()

    def _get_best_document_frame(self, track_id: int) -> Optional[np.ndarray]:
        """Get best quality document frame for a track.

        Args:
            track_id: Track ID

        Returns:
            Best document frame, or None if not available
        """
        if track_id not in self.document_frame_buffer:
            return None

        frames = self.document_frame_buffer[track_id]
        if not frames:
            return None

        # For now, return the most recent frame
        # TODO: Could use frame quality assessment here
        latest_frame_num = max(frames.keys())
        return frames[latest_frame_num]

    def finalize(self):
        """Finalize processing - process any remaining active tracks."""
        # Mark all active tracks as lost to trigger extraction
        for track_id in list(self.active_tracks.keys()):
            self.active_tracks[track_id].status = TrackStatus.LOST

        # Process remaining tracks
        self._process_completed_tracks()

    def get_statistics(self) -> Dict:
        """Get pipeline statistics.

        Returns:
            Dictionary with processing statistics
        """
        queue_stats = self.review_queue_manager.get_queue_stats()
        quality_report = self.metrics_tracker.generate_quality_report()

        return {
            'frames_processed': self.stats['total_frames_processed'],
            'pallets_tracked': self.stats['total_pallets_tracked'],
            'documents_detected': self.stats['total_documents_detected'],
            'ocr_runs': self.stats['total_ocr_runs'],
            'extractions_completed': self.stats['total_extractions'],
            'queue_stats': queue_stats,
            'quality_report': quality_report,
        }

    def get_review_queue(self) -> List[Dict]:
        """Get items needing review.

        Returns:
            List of review items
        """
        return self.review_queue_manager.get_review_queue()

    def get_auto_accepted(self) -> List[Dict]:
        """Get auto-accepted extractions.

        Returns:
            List of auto-accepted extractions
        """
        return self.review_queue_manager.auto_accepted

    def get_auto_rejected(self) -> List[Dict]:
        """Get auto-rejected extractions.

        Returns:
            List of auto-rejected extractions
        """
        return self.review_queue_manager.auto_rejected
