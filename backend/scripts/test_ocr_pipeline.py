"""Integration test script for complete OCR pipeline.

This script demonstrates the full OCR workflow:
1. Load video/camera feed
2. Detect and track pallets (Chunks 3-4)
3. Detect documents on pallets (Chunk 5)
4. Select best frames for OCR (Chunk 6)
5. Extract text with preprocessing (Chunk 7)
6. Aggregate multi-frame results
7. Post-process and validate text

Usage:
    python scripts/test_ocr_pipeline.py --video path/to/video.mp4
    python scripts/test_ocr_pipeline.py --camera 0
    python scripts/test_ocr_pipeline.py --images path/to/images/*.jpg
"""

import argparse
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import yaml

from app.cv_models import BoundingBox, PalletTrack
from app.cv_processing import (
    DocumentAssociator,
    DocumentDetector,
    DocumentOCR,
    DocumentRegionExtractor,
    FrameSelectionStrategy,
    MultiFrameOCRAggregator,
    OCRPostProcessor,
    OCRPreprocessor,
    PalletDetector,
    TrackManager,
)


class OCRPipelineTester:
    """Test complete OCR pipeline on video/camera feed."""

    def __init__(self, config_path: str):
        """Initialize OCR pipeline components.

        Args:
            config_path: Path to configuration YAML file
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        print("Initializing pipeline components...")

        # Initialize detection and tracking
        self.pallet_detector = PalletDetector(self.config)
        self.track_manager = TrackManager(self.config)
        self.document_detector = DocumentDetector(self.config)

        # Initialize frame selection
        self.frame_selector = FrameSelectionStrategy(self.config)

        # Initialize OCR components
        self.ocr_engine = DocumentOCR(self.config)
        self.preprocessor = OCRPreprocessor()
        self.post_processor = OCRPostProcessor()

        # Track storage
        self.completed_tracks: Dict[int, PalletTrack] = {}
        self.frame_storage: Dict[int, np.ndarray] = {}

        # OCR results per track
        self.track_ocr_results: Dict[int, List[List]] = defaultdict(list)

        print("✓ Pipeline initialized")

    def process_frame(self, frame: np.ndarray, frame_number: int):
        """Process single frame through pipeline.

        Args:
            frame: Input frame
            frame_number: Frame number
        """
        # Step 1: Detect pallets
        pallet_detections = self.pallet_detector.detect(frame, frame_number)

        # Step 2: Update tracks
        self.track_manager.update(pallet_detections, frame_number)

        # Step 3: Get active tracks
        active_tracks = self.track_manager.get_active_tracks()

        # Step 4: Detect documents on pallets
        if active_tracks:
            document_detections = self.document_detector.detect(
                frame, frame_number, active_tracks
            )

            # Associate documents to pallets
            associated_docs = DocumentAssociator.associate_documents_to_pallets(
                document_detections,
                active_tracks,
                max_distance=self.config['detection']['max_association_distance']
            )

            # Add documents to tracks
            for doc in associated_docs:
                if doc.track_id in active_tracks:
                    active_tracks[doc.track_id].document_regions.append(doc)

        # Step 5: Store frame for OCR processing
        self.frame_storage[frame_number] = frame.copy()

        # Step 6: Check for completed tracks
        completed = self.track_manager.get_completed_tracks()
        for track in completed:
            if track.track_id not in self.completed_tracks:
                self.completed_tracks[track.track_id] = track
                print(f"\n✓ Track {track.track_id} completed - preparing for OCR")
                self.process_completed_track(track)

        return frame, active_tracks

    def process_completed_track(self, track: PalletTrack):
        """Process completed track for OCR.

        Args:
            track: Completed PalletTrack
        """
        print(f"  Track {track.track_id}: {len(track.detections)} frames")
        print(f"  Documents detected: {len(track.document_regions)}")

        if not track.document_regions:
            print(f"  ⚠ No documents found on track {track.track_id}")
            return

        # Step 1: Select best frames for OCR
        selected_frames = self.frame_selector.select_frames_for_track(
            track,
            self.frame_storage
        )

        if not selected_frames:
            print(f"  ⚠ No suitable frames selected for track {track.track_id}")
            return

        print(f"  Selected {len(selected_frames)} frames for OCR: {selected_frames}")

        # Step 2: Extract document regions from selected frames
        extractor = DocumentRegionExtractor(padding=10)
        frame_ocr_results = []

        for frame_num in selected_frames:
            if frame_num not in self.frame_storage:
                continue

            frame = self.frame_storage[frame_num]

            # Get documents for this frame
            frame_docs = [
                doc for doc in track.document_regions
                if doc.frame_number == frame_num
            ]

            if not frame_docs:
                continue

            # Process each document region
            for doc in frame_docs:
                # Extract region
                region, metadata = extractor.extract_region(
                    frame,
                    doc.bbox,
                    return_metadata=True
                )

                if region is None or region.size == 0:
                    continue

                print(f"    Processing frame {frame_num}, doc confidence: {doc.confidence:.2f}")

                # Preprocess
                preprocessed = self.preprocessor.preprocess(region, self.config)

                # Save preprocessed image for debugging
                debug_path = Path("debug_ocr")
                debug_path.mkdir(exist_ok=True)
                cv2.imwrite(
                    str(debug_path / f"track_{track.track_id}_frame_{frame_num}_preprocessed.jpg"),
                    preprocessed
                )

                # Extract text
                ocr_results = self.ocr_engine.extract_text(
                    preprocessed,
                    preprocess=False,  # Already preprocessed
                    frame_number=frame_num
                )

                if ocr_results:
                    frame_ocr_results.append(ocr_results)
                    print(f"      Extracted {len(ocr_results)} text regions")

                    # Display extracted text
                    for result in ocr_results:
                        print(f"        '{result.text}' (conf: {result.confidence:.2f})")

        # Step 3: Aggregate results across frames
        if frame_ocr_results:
            aggregation_method = self.config['ocr']['aggregation']['method']
            aggregated_text = MultiFrameOCRAggregator.aggregate_ocr_results(
                frame_ocr_results,
                method=aggregation_method
            )

            # Calculate consensus confidence
            consensus_conf = MultiFrameOCRAggregator.calculate_consensus_confidence(
                frame_ocr_results
            )

            print(f"\n  Aggregated text ({aggregation_method}): '{aggregated_text}'")
            print(f"  Consensus confidence: {consensus_conf:.2%}")

            # Step 4: Post-process text
            cleaned_text = self.post_processor.clean_text(aggregated_text, self.config)
            print(f"  Cleaned text: '{cleaned_text}'")

            # Extract tracking numbers if present
            tracking_numbers = self.post_processor.extract_tracking_numbers(cleaned_text)
            if tracking_numbers:
                print(f"  📦 Tracking numbers found: {tracking_numbers}")

            # Calculate quality score
            quality_score = self.post_processor.calculate_text_quality_score(cleaned_text)
            print(f"  Quality score: {quality_score:.2f}")

            # Store results
            self.track_ocr_results[track.track_id] = {
                'raw_text': aggregated_text,
                'cleaned_text': cleaned_text,
                'consensus_confidence': consensus_conf,
                'quality_score': quality_score,
                'tracking_numbers': tracking_numbers,
                'num_frames_processed': len(frame_ocr_results)
            }

    def visualize_frame(
        self,
        frame: np.ndarray,
        active_tracks: Dict[int, PalletTrack]
    ) -> np.ndarray:
        """Visualize detections and tracking.

        Args:
            frame: Input frame
            active_tracks: Active tracks dictionary

        Returns:
            Annotated frame
        """
        vis_frame = frame.copy()

        # Draw pallet tracks
        for track_id, track in active_tracks.items():
            if not track.detections:
                continue

            latest_det = track.detections[-1]
            bbox = latest_det.bbox

            # Draw pallet bbox
            cv2.rectangle(
                vis_frame,
                (int(bbox.x_min), int(bbox.y_min)),
                (int(bbox.x_max), int(bbox.y_max)),
                (0, 255, 0),
                2
            )

            # Draw track ID
            cv2.putText(
                vis_frame,
                f"Track {track_id}",
                (int(bbox.x_min), int(bbox.y_min) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

            # Draw document regions if present
            for doc in track.document_regions:
                if doc.frame_number == latest_det.frame_number:
                    doc_bbox = doc.bbox
                    cv2.rectangle(
                        vis_frame,
                        (int(doc_bbox.x_min), int(doc_bbox.y_min)),
                        (int(doc_bbox.x_max), int(doc_bbox.y_max)),
                        (255, 0, 0),
                        2
                    )

                    cv2.putText(
                        vis_frame,
                        f"Doc {doc.confidence:.2f}",
                        (int(doc_bbox.x_min), int(doc_bbox.y_min) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (255, 0, 0),
                        1
                    )

        return vis_frame

    def process_video(self, video_path: str, max_frames: int = None):
        """Process video file.

        Args:
            video_path: Path to video file
            max_frames: Maximum frames to process (None = all)
        """
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"\nProcessing video: {video_path}")
        print(f"FPS: {fps:.2f}, Total frames: {total_frames}")

        frame_number = 0
        start_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if max_frames and frame_number >= max_frames:
                break

            # Process frame
            vis_frame, active_tracks = self.process_frame(frame, frame_number)

            # Visualize
            vis_frame = self.visualize_frame(vis_frame, active_tracks)

            # Display
            cv2.imshow("OCR Pipeline", vis_frame)

            # Progress
            if frame_number % 30 == 0:
                elapsed = time.time() - start_time
                fps_actual = frame_number / elapsed if elapsed > 0 else 0
                print(f"Frame {frame_number}/{total_frames if not max_frames else max_frames} "
                      f"({fps_actual:.1f} FPS)")

            frame_number += 1

            # Exit on 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

        # Print final summary
        self.print_summary()

    def process_camera(self, camera_id: int = 0):
        """Process camera feed.

        Args:
            camera_id: Camera device ID
        """
        cap = cv2.VideoCapture(camera_id)

        if not cap.isOpened():
            print(f"Error: Could not open camera {camera_id}")
            return

        print(f"\nProcessing camera {camera_id}")
        print("Press 'q' to quit")

        frame_number = 0
        start_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error reading frame")
                break

            # Process frame
            vis_frame, active_tracks = self.process_frame(frame, frame_number)

            # Visualize
            vis_frame = self.visualize_frame(vis_frame, active_tracks)

            # Display FPS
            elapsed = time.time() - start_time
            fps = frame_number / elapsed if elapsed > 0 else 0
            cv2.putText(
                vis_frame,
                f"FPS: {fps:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            # Display
            cv2.imshow("OCR Pipeline - Live", vis_frame)

            frame_number += 1

            # Exit on 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

        # Print final summary
        self.print_summary()

    def print_summary(self):
        """Print pipeline summary."""
        print("\n" + "="*80)
        print("OCR PIPELINE SUMMARY")
        print("="*80)

        print(f"\nCompleted tracks: {len(self.completed_tracks)}")
        print(f"Tracks with OCR results: {len(self.track_ocr_results)}")

        for track_id, results in self.track_ocr_results.items():
            print(f"\n--- Track {track_id} ---")
            print(f"  Cleaned text: '{results['cleaned_text']}'")
            print(f"  Consensus confidence: {results['consensus_confidence']:.2%}")
            print(f"  Quality score: {results['quality_score']:.2f}")
            print(f"  Frames processed: {results['num_frames_processed']}")

            if results['tracking_numbers']:
                print(f"  📦 Tracking numbers: {results['tracking_numbers']}")

        # Overall statistics
        if self.track_ocr_results:
            avg_confidence = np.mean([
                r['consensus_confidence'] for r in self.track_ocr_results.values()
            ])
            avg_quality = np.mean([
                r['quality_score'] for r in self.track_ocr_results.values()
            ])

            print(f"\n--- Overall Statistics ---")
            print(f"  Average consensus confidence: {avg_confidence:.2%}")
            print(f"  Average quality score: {avg_quality:.2f}")

        print("\n" + "="*80)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Test complete OCR pipeline")

    parser.add_argument(
        '--config',
        type=str,
        default='app/cv_config/config.yaml',
        help='Path to configuration file'
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--video',
        type=str,
        help='Path to video file'
    )
    input_group.add_argument(
        '--camera',
        type=int,
        help='Camera device ID'
    )

    parser.add_argument(
        '--max-frames',
        type=int,
        default=None,
        help='Maximum frames to process (video only)'
    )

    args = parser.parse_args()

    # Initialize pipeline
    tester = OCRPipelineTester(args.config)

    # Process input
    if args.video:
        tester.process_video(args.video, max_frames=args.max_frames)
    elif args.camera is not None:
        tester.process_camera(args.camera)


if __name__ == "__main__":
    main()
