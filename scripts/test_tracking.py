#!/usr/bin/env python3
"""Integration test for pallet detection and tracking.

This script demonstrates the full detection + tracking pipeline using
a sample video or image sequence.

Usage:
    python scripts/test_tracking.py --video /path/to/video.mp4
    python scripts/test_tracking.py --images /path/to/images/*.jpg
    python scripts/test_tracking.py --camera 0  # Use webcam
"""

import argparse
import sys
from pathlib import Path

import cv2
import yaml

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend" / "app"))

from cv_models import PalletDetection
from cv_processing import PalletDetector, TrackManager, TrackVisualizer


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


def process_video(video_path: str, config: dict, output_path: str = None):
    """Process video with detection and tracking.

    Args:
        video_path: Path to input video
        config: Configuration dictionary
        output_path: Optional path to save output video
    """
    # Initialize detector and tracker
    print("Initializing detector and tracker...")
    detector = PalletDetector(config)
    track_manager = TrackManager(config)
    visualizer = TrackVisualizer()

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

    frame_number = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Calculate timestamp
            timestamp = frame_number / fps

            # Detect pallets
            detections = detector.detect(frame, frame_number, timestamp)

            # Update tracker
            active_tracks = track_manager.update(detections, frame_number)

            # Visualize
            vis_frame = TrackVisualizer.draw_tracks(
                frame,
                active_tracks,
                show_trail=True,
                show_info=True
            )

            # Add statistics
            completed_count = len(track_manager.tracker.completed_tracks)
            vis_frame = TrackVisualizer.draw_track_stats(
                vis_frame,
                active_count=len(active_tracks),
                completed_count=completed_count,
                frame_number=frame_number
            )

            # Write output
            if writer:
                writer.write(vis_frame)

            # Display
            cv2.imshow('Pallet Tracking', vis_frame)

            # Check for exit key
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Quit requested by user")
                break
            elif key == ord(' '):
                # Pause on spacebar
                cv2.waitKey(0)

            # Progress
            if frame_number % 30 == 0:
                print(f"Frame {frame_number}/{total_frames} - "
                      f"Active: {len(active_tracks)}, "
                      f"Completed: {completed_count}")

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
    print("\n" + "=" * 50)
    print("TRACKING SUMMARY")
    print("=" * 50)
    print(f"Total frames processed: {frame_number}")
    print(f"Total tracks created: {len(track_manager.tracker.completed_tracks)}")
    print(f"Tracks ready for extraction: {len(ready_tracks)}")

    if ready_tracks:
        print("\nTrack Details:")
        for track in ready_tracks[:10]:  # Show first 10
            print(f"  Track {track.track_id}: "
                  f"{len(track.detections)} detections, "
                  f"frames {track.first_seen_frame}-{track.last_seen_frame}")


def process_camera(camera_id: int, config: dict):
    """Process live camera feed with detection and tracking.

    Args:
        camera_id: Camera device ID (usually 0 for default camera)
        config: Configuration dictionary
    """
    # Initialize detector and tracker
    print("Initializing detector and tracker...")
    detector = PalletDetector(config)
    track_manager = TrackManager(config)
    visualizer = TrackVisualizer()

    # Open camera
    print(f"Opening camera {camera_id}...")
    cap = cv2.VideoCapture(camera_id)

    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_id}")
        return

    frame_number = 0
    fps_estimate = 30.0  # Estimate for timestamp calculation

    print("Press 'q' to quit, 'space' to pause")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            timestamp = frame_number / fps_estimate

            # Detect and track
            detections = detector.detect(frame, frame_number, timestamp)
            active_tracks = track_manager.update(detections, frame_number)

            # Visualize
            vis_frame = TrackVisualizer.draw_tracks(
                frame,
                active_tracks,
                show_trail=True,
                show_info=True
            )

            completed_count = len(track_manager.tracker.completed_tracks)
            vis_frame = TrackVisualizer.draw_track_stats(
                vis_frame,
                active_count=len(active_tracks),
                completed_count=completed_count,
                frame_number=frame_number
            )

            cv2.imshow('Live Pallet Tracking', vis_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):
                cv2.waitKey(0)

            frame_number += 1

    finally:
        cap.release()
        cv2.destroyAllWindows()

    print(f"\nProcessed {frame_number} frames")
    print(f"Total tracks: {len(track_manager.tracker.active_tracks) + len(track_manager.tracker.completed_tracks)}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Test pallet detection and tracking on video or camera feed"
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

    args = parser.parse_args()

    # Validate arguments
    if not args.video and args.camera is None:
        parser.error("Must specify either --video or --camera")

    if args.video and args.camera is not None:
        parser.error("Cannot specify both --video and --camera")

    # Load configuration
    print("Loading configuration...")
    config = load_config(args.config)

    # Check if model exists
    model_path = config['detection']['pallet_model_path']
    if not Path(model_path).exists():
        print(f"\nWARNING: Model not found at {model_path}")
        print("This script requires a trained YOLOv8 pallet detection model.")
        print("Please train a model or download a pre-trained one.")
        print("\nFor testing purposes, you can:")
        print("  1. Train a model: python backend/scripts/train_detector.py")
        print("  2. Or use a generic YOLOv8 model (will detect all objects, not just pallets)")
        return

    # Process input
    if args.video:
        if not Path(args.video).exists():
            print(f"Error: Video file not found: {args.video}")
            return
        process_video(args.video, config, args.output)
    else:
        process_camera(args.camera, config)


if __name__ == "__main__":
    main()
