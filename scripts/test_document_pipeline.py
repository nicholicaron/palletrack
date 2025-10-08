#!/usr/bin/env python3
"""Integration test for complete pallet tracking + document detection pipeline.

This script demonstrates the full end-to-end pipeline:
1. Detect pallets with YOLOv8
2. Track pallets with ByteTrack
3. Detect documents on pallets
4. Associate documents to pallets
5. Extract document regions for OCR

Usage:
    python scripts/test_document_pipeline.py --video /path/to/video.mp4
    python scripts/test_document_pipeline.py --video /path/to/video.mp4 --output /path/to/output.mp4
    python scripts/test_document_pipeline.py --camera 0  # Use webcam
"""

import argparse
import sys
from pathlib import Path

import cv2
import yaml

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend" / "app"))

from cv_models import PalletDetection
from cv_processing import (
    DocumentAssociator,
    DocumentDetector,
    DocumentRegionExtractor,
    PalletDetector,
    TrackManager,
    TrackVisualizer,
)


def load_config(config_path: str = None) -> dict:
    """Load configuration from YAML file.

    Args:
        config_path: Path to config file. If None, uses default.

    Returns:
        Configuration dictionary
    """
    if config_path is None:
        config_path = Path(__file__).parent.parent / "backend" / "app" / "cv_config" / "config.yaml"

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config


def draw_document_detections(frame, documents, color=(0, 255, 255)):
    """Draw document bounding boxes on frame.

    Args:
        frame: Input frame
        documents: List of DocumentDetection objects
        color: Box color (default: yellow)

    Returns:
        Annotated frame
    """
    annotated = frame.copy()

    for doc in documents:
        bbox = doc.bbox
        pt1 = (int(bbox.x1), int(bbox.y1))
        pt2 = (int(bbox.x2), int(bbox.y2))

        # Draw bounding box
        cv2.rectangle(annotated, pt1, pt2, color, 2)

        # Draw label
        label = f"Doc: {bbox.confidence:.2f}"
        if doc.parent_pallet_track_id is not None:
            label += f" (P{doc.parent_pallet_track_id})"

        # Text background
        font = cv2.FONT_HERSHEY_SIMPLEX
        (text_width, text_height), baseline = cv2.getTextSize(label, font, 0.4, 1)
        bg_pt1 = (int(bbox.x1), int(bbox.y1) - text_height - baseline - 5)
        bg_pt2 = (int(bbox.x1) + text_width + 5, int(bbox.y1))
        cv2.rectangle(annotated, bg_pt1, bg_pt2, color, -1)

        # Draw text
        text_pt = (int(bbox.x1) + 2, int(bbox.y1) - 5)
        cv2.putText(annotated, label, text_pt, font, 0.4, (0, 0, 0), 1)

    return annotated


def process_video(video_path: str, config: dict, output_path: str = None, save_regions: bool = False):
    """Process video with full document detection pipeline.

    Args:
        video_path: Path to input video
        config: Configuration dictionary
        output_path: Optional path to save output video
        save_regions: Whether to save extracted document regions
    """
    # Initialize components
    print("Initializing pipeline components...")
    pallet_detector = PalletDetector(config)
    track_manager = TrackManager(config)
    document_detector = DocumentDetector(config)

    # Open video
    print(f"Opening video: {video_path}")
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video: {width}x{height} @ {fps:.2f} fps, {total_frames} frames")

    # Setup output video if requested
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        print(f"Saving output to: {output_path}")

    # Setup directory for saving document regions
    if save_regions:
        regions_dir = Path("document_regions")
        regions_dir.mkdir(exist_ok=True)
        print(f"Saving document regions to: {regions_dir}")

    frame_number = 0
    total_documents_detected = 0
    total_documents_associated = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Calculate timestamp
            timestamp = frame_number / fps

            # Step 1: Detect pallets
            pallet_detections = pallet_detector.detect(frame, frame_number, timestamp)

            # Step 2: Update tracker
            active_tracks = track_manager.update(pallet_detections, frame_number)

            # Step 3: Detect documents on pallets
            documents = document_detector.detect(frame, frame_number, active_tracks)
            total_documents_detected += len(documents)

            # Step 4: Associate documents to pallets
            max_distance = config.get('detection', {}).get('max_association_distance', 100)
            associated_documents = DocumentAssociator.associate_documents_to_pallets(
                documents, active_tracks, max_distance=max_distance
            )

            # Count successfully associated documents
            associated_count = sum(
                1 for doc in associated_documents if doc.parent_pallet_track_id is not None
            )
            total_documents_associated += associated_count

            # Step 5: Extract document regions (if saving)
            if save_regions and associated_documents:
                regions = DocumentRegionExtractor.extract_multiple_regions(
                    frame, associated_documents, padding=10
                )

                # Save regions
                for idx, region in regions.items():
                    doc = associated_documents[idx]
                    pallet_id = doc.parent_pallet_track_id or "unknown"
                    region_path = regions_dir / f"frame_{frame_number:06d}_pallet_{pallet_id}_doc_{idx}.jpg"
                    cv2.imwrite(str(region_path), region)

            # Visualize results
            # First, draw pallet tracks
            vis_frame = TrackVisualizer.draw_tracks(
                frame,
                active_tracks,
                show_trail=True,
                show_info=True
            )

            # Then, draw document detections on top
            vis_frame = draw_document_detections(vis_frame, associated_documents)

            # Add statistics
            completed_count = len(track_manager.tracker.completed_tracks)
            vis_frame = TrackVisualizer.draw_track_stats(
                vis_frame,
                active_count=len(active_tracks),
                completed_count=completed_count,
                frame_number=frame_number
            )

            # Add document stats
            cv2.putText(
                vis_frame,
                f"Documents: {len(associated_documents)} ({associated_count} associated)",
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2
            )

            # Write output
            if writer:
                writer.write(vis_frame)

            # Display
            cv2.imshow('Document Detection Pipeline', vis_frame)

            # Check for exit key
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Quit requested by user")
                break
            elif key == ord(' '):
                # Pause on spacebar
                cv2.waitKey(0)
            elif key == ord('s'):
                # Save current frame on 's' key
                screenshot_path = f"frame_{frame_number:06d}.jpg"
                cv2.imwrite(screenshot_path, vis_frame)
                print(f"Saved screenshot: {screenshot_path}")

            # Progress
            if frame_number % 30 == 0:
                print(f"Frame {frame_number}/{total_frames} - "
                      f"Pallets: {len(active_tracks)}, "
                      f"Documents: {len(associated_documents)} ({associated_count} associated)")

            frame_number += 1

    finally:
        # Cleanup
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()

    # Finalize all tracks
    print("\nFinalizing tracks...")
    track_manager.finalize_all_tracks()

    # Get tracks ready for extraction
    ready_tracks = track_manager.get_tracks_ready_for_extraction()

    # Print summary
    print("\n" + "=" * 60)
    print("PIPELINE SUMMARY")
    print("=" * 60)
    print(f"Total frames processed: {frame_number}")
    print(f"Total pallet tracks: {len(track_manager.tracker.completed_tracks)}")
    print(f"Tracks ready for OCR: {len(ready_tracks)}")
    print(f"Total documents detected: {total_documents_detected}")
    print(f"Total documents associated: {total_documents_associated}")

    if ready_tracks:
        print("\nPallet Tracks with Documents:")
        for track in ready_tracks[:10]:  # Show first 10
            doc_count = len(track.document_regions)
            print(f"  Track {track.track_id}: "
                  f"{len(track.detections)} detections, "
                  f"{doc_count} documents, "
                  f"frames {track.first_seen_frame}-{track.last_seen_frame}")


def process_camera(camera_id: int, config: dict, save_regions: bool = False):
    """Process live camera feed with document detection.

    Args:
        camera_id: Camera device ID (usually 0 for default camera)
        config: Configuration dictionary
        save_regions: Whether to save extracted document regions
    """
    # Initialize components
    print("Initializing pipeline components...")
    pallet_detector = PalletDetector(config)
    track_manager = TrackManager(config)
    document_detector = DocumentDetector(config)

    # Open camera
    print(f"Opening camera {camera_id}...")
    cap = cv2.VideoCapture(camera_id)

    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_id}")
        return

    # Setup directory for saving document regions
    if save_regions:
        regions_dir = Path("document_regions")
        regions_dir.mkdir(exist_ok=True)
        print(f"Saving document regions to: {regions_dir}")

    frame_number = 0
    fps_estimate = 30.0  # Estimate for timestamp calculation

    print("Press 'q' to quit, 'space' to pause, 's' to save screenshot")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            timestamp = frame_number / fps_estimate

            # Pipeline: detect → track → detect documents → associate
            pallet_detections = pallet_detector.detect(frame, frame_number, timestamp)
            active_tracks = track_manager.update(pallet_detections, frame_number)

            documents = document_detector.detect(frame, frame_number, active_tracks)
            max_distance = config.get('detection', {}).get('max_association_distance', 100)
            associated_documents = DocumentAssociator.associate_documents_to_pallets(
                documents, active_tracks, max_distance=max_distance
            )

            # Save regions if requested
            if save_regions and associated_documents:
                regions = DocumentRegionExtractor.extract_multiple_regions(
                    frame, associated_documents, padding=10
                )
                for idx, region in regions.items():
                    doc = associated_documents[idx]
                    pallet_id = doc.parent_pallet_track_id or "unknown"
                    region_path = regions_dir / f"frame_{frame_number:06d}_pallet_{pallet_id}_doc_{idx}.jpg"
                    cv2.imwrite(str(region_path), region)

            # Visualize
            vis_frame = TrackVisualizer.draw_tracks(
                frame, active_tracks, show_trail=True, show_info=True
            )
            vis_frame = draw_document_detections(vis_frame, associated_documents)

            completed_count = len(track_manager.tracker.completed_tracks)
            vis_frame = TrackVisualizer.draw_track_stats(
                vis_frame,
                active_count=len(active_tracks),
                completed_count=completed_count,
                frame_number=frame_number
            )

            associated_count = sum(
                1 for doc in associated_documents if doc.parent_pallet_track_id is not None
            )
            cv2.putText(
                vis_frame,
                f"Documents: {len(associated_documents)} ({associated_count} associated)",
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2
            )

            cv2.imshow('Live Document Detection', vis_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):
                cv2.waitKey(0)
            elif key == ord('s'):
                screenshot_path = f"camera_frame_{frame_number:06d}.jpg"
                cv2.imwrite(screenshot_path, vis_frame)
                print(f"Saved screenshot: {screenshot_path}")

            frame_number += 1

    finally:
        cap.release()
        cv2.destroyAllWindows()

    print(f"\nProcessed {frame_number} frames")
    print(f"Total tracks: {len(track_manager.tracker.active_tracks) + len(track_manager.tracker.completed_tracks)}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Test complete document detection pipeline on video or camera feed"
    )
    parser.add_argument(
        '--video',
        type=str,
        help='Path to input video file'
    )
    parser.add_argument(
        '--camera',
        type=int,
        help='Camera device ID (default: 0 for default camera)'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Path to save output video (only for --video mode)'
    )
    parser.add_argument(
        '--config',
        type=str,
        help='Path to config YAML file (default: backend/app/cv_config/config.yaml)'
    )
    parser.add_argument(
        '--save-regions',
        action='store_true',
        help='Save extracted document regions to document_regions/ directory'
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.video and args.camera is None:
        parser.error("Must specify either --video or --camera")

    if args.video and args.camera is not None:
        parser.error("Cannot specify both --video and --camera")

    # Load configuration
    print("Loading configuration...")
    config = load_config(args.config)

    # Check if models exist
    pallet_model_path = config['detection']['pallet_model_path']
    document_model_path = config['detection']['document_model_path']

    if not Path(pallet_model_path).exists():
        print(f"\nWARNING: Pallet model not found at {pallet_model_path}")
        print("Please train or download a pallet detection model.")
        return

    if not Path(document_model_path).exists():
        print(f"\nWARNING: Document model not found at {document_model_path}")
        print("Please train or download a document detection model.")
        return

    # Process input
    if args.video:
        if not Path(args.video).exists():
            print(f"Error: Video file not found: {args.video}")
            return
        process_video(args.video, config, args.output, args.save_regions)
    else:
        process_camera(args.camera, config, args.save_regions)


if __name__ == "__main__":
    main()
