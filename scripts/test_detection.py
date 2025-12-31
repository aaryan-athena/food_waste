"""
Test Steel Utensil Detection on Video
======================================

Quick test script to verify steel utensil detection works on the new dataset.

Usage:
    python scripts/test_detection.py [video_path]
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.vision_steel import (
    detect_steel_utensil,
    segment_food_in_steel_utensil,
    process_frame_steel_utensil,
    draw_detection_overlay,
)


def test_on_video(video_path: Path, output_dir: Path = None, show_preview: bool = True):
    """
    Test steel utensil detection on a video file.
    
    Args:
        video_path: Path to video file
        output_dir: Directory to save test frames (optional)
        show_preview: Show preview window (requires display)
    """
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"Video: {video_path.name}")
    print(f"Total frames: {total_frames}, FPS: {fps:.2f}")
    print("-" * 50)
    
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
    
    frame_idx = 0
    detections = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process every 30th frame
        if frame_idx % 30 == 0:
            # Run detection
            result = process_frame_steel_utensil(
                frame,
                known_dimension_mm=200,  # Assume 200mm utensil width
                assumed_depth_mm=30,
                debug=False
            )
            
            # Draw overlay
            display = draw_detection_overlay(frame, result)
            
            if result['utensil_detected']:
                detections += 1
                status = f"Detected: {result['shape_type']}, Fill: {result['percent_fill']:.1f}%"
            else:
                status = "No utensil detected"
            
            print(f"Frame {frame_idx}: {status}")
            
            # Save sample frames
            if output_dir and frame_idx % 90 == 0:
                output_path = output_dir / f"{video_path.stem}_frame_{frame_idx:05d}.jpg"
                cv2.imwrite(str(output_path), display)
            
            # Show preview
            if show_preview:
                try:
                    # Resize for display
                    h, w = display.shape[:2]
                    scale = min(800 / w, 600 / h)
                    display_resized = cv2.resize(display, None, fx=scale, fy=scale)
                    
                    cv2.imshow('Steel Utensil Detection', display_resized)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord(' '):
                        # Pause on spacebar
                        cv2.waitKey(0)
                except cv2.error as e:
                    if 'imshow' in str(e).lower() or 'not implemented' in str(e).lower():
                        print("\nWarning: GUI display not available (opencv-python-headless doesn't support cv2.imshow)")
                        print("Continuing without preview. Images will still be saved to output directory.")
                        show_preview = False  # Disable for remaining frames
                    else:
                        raise
        
        frame_idx += 1
    
    cap.release()
    if show_preview:
        try:
            cv2.destroyAllWindows()
        except:
            pass
    
    print("-" * 50)
    print(f"Processed {frame_idx} frames")
    print(f"Detections: {detections}/{frame_idx // 30} sampled frames")


def main():
    parser = argparse.ArgumentParser(
        description="Test steel utensil detection on video"
    )
    parser.add_argument(
        "video",
        type=str,
        nargs="?",
        default=None,
        help="Path to video file (default: first video in data/)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="test_output",
        help="Directory to save test frames",
    )
    parser.add_argument(
        "--no-preview",
        action="store_true",
        default=True,  # Default to no preview since opencv-headless doesn't support it
        help="Disable preview window (default: True for opencv-headless)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Test on all videos in data folder",
    )
    
    args = parser.parse_args()
    
    output_dir = PROJECT_ROOT / args.output_dir
    show_preview = not args.no_preview
    
    if args.all:
        # Test all videos
        data_dir = PROJECT_ROOT / "data"
        videos = list(data_dir.glob("*.mp4"))
        
        if not videos:
            print(f"No videos found in {data_dir}")
            sys.exit(1)
        
        print(f"Testing {len(videos)} videos...")
        for video_path in sorted(videos):
            test_on_video(video_path, output_dir / video_path.stem, show_preview=False)
            print()
    else:
        # Test single video
        if args.video:
            video_path = Path(args.video)
            if not video_path.is_absolute():
                video_path = PROJECT_ROOT / video_path
        else:
            # Default to first video in data folder
            data_dir = PROJECT_ROOT / "data"
            videos = list(data_dir.glob("*.mp4"))
            if not videos:
                print(f"No videos found in {data_dir}")
                print("Usage: python scripts/test_detection.py <video_path>")
                sys.exit(1)
            video_path = videos[0]
        
        if not video_path.exists():
            print(f"Video not found: {video_path}")
            sys.exit(1)
        
        test_on_video(video_path, output_dir, show_preview)


if __name__ == "__main__":
    main()
