"""
Video Frame Extraction Script
=============================
Extracts frames from MP4 videos in the data folder to create training images
for the food classification model.

Usage:
    python scripts/extract_frames.py [--input-dir data] [--output-dir training_data] [--frame-interval 30]
"""

import argparse
import os
import sys
from pathlib import Path

import cv2

# Project root for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def extract_frames_from_video(
    video_path: Path,
    output_dir: Path,
    frame_interval: int = 30,
    max_frames: int = None,
) -> int:
    """
    Extract frames from a video file.
    
    Args:
        video_path: Path to the input video file
        output_dir: Directory to save extracted frames
        frame_interval: Extract every Nth frame (default: 30, ~1 frame/sec at 30fps)
        max_frames: Maximum number of frames to extract (None = no limit)
    
    Returns:
        Number of frames extracted
    """
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return 0
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"  Video: {video_path.name}")
    print(f"    Total frames: {total_frames}, FPS: {fps:.2f}")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    frame_count = 0
    extracted_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Extract frame at specified interval
        if frame_count % frame_interval == 0:
            # Generate filename with video name prefix and frame number
            filename = f"{video_path.stem}_frame_{frame_count:06d}.jpg"
            output_path = output_dir / filename
            
            # Save frame
            cv2.imwrite(str(output_path), frame)
            extracted_count += 1
            
            if max_frames and extracted_count >= max_frames:
                break
        
        frame_count += 1
    
    cap.release()
    print(f"    Extracted {extracted_count} frames")
    return extracted_count


def organize_by_class(output_base: Path):
    """
    Organize extracted frames into class folders based on video name (food type).
    This creates the folder structure needed for training:
    
    training_data/
    ├── chicken_curry/
    │   ├── chicken_curry_frame_000000.jpg
    │   └── ...
    ├── rice/
    │   ├── rice_frame_000000.jpg
    │   └── ...
    └── ...
    """
    frames_dir = output_base / "all_frames"
    
    if not frames_dir.exists():
        print("No frames directory found. Run extraction first.")
        return
    
    print("\nOrganizing frames by class...")
    
    for frame_file in frames_dir.glob("*.jpg"):
        # Extract class name from filename (e.g., "chicken_curry_frame_000000.jpg" -> "chicken_curry")
        parts = frame_file.stem.rsplit("_frame_", 1)
        if len(parts) != 2:
            continue
        
        class_name = parts[0]
        class_dir = output_base / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        
        # Move file to class directory (overwrite if exists)
        dest = class_dir / frame_file.name
        if dest.exists():
            dest.unlink()
        frame_file.rename(dest)
    
    # Remove empty all_frames directory
    if frames_dir.exists() and not any(frames_dir.iterdir()):
        frames_dir.rmdir()
    
    # Print summary
    print("\nDataset structure:")
    for class_dir in sorted(output_base.iterdir()):
        if class_dir.is_dir():
            count = len(list(class_dir.glob("*.jpg")))
            print(f"  {class_dir.name}: {count} images")


def main():
    parser = argparse.ArgumentParser(
        description="Extract frames from videos for food classification training"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="data",
        help="Directory containing MP4 video files (default: data)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="training_data",
        help="Output directory for extracted frames (default: training_data)",
    )
    parser.add_argument(
        "--frame-interval",
        type=int,
        default=30,
        help="Extract every Nth frame (default: 30, ~1 frame/sec at 30fps)",
    )
    parser.add_argument(
        "--max-frames-per-video",
        type=int,
        default=None,
        help="Maximum frames to extract per video (default: no limit)",
    )
    parser.add_argument(
        "--organize",
        action="store_true",
        help="Organize frames into class folders after extraction",
    )
    
    args = parser.parse_args()
    
    input_dir = PROJECT_ROOT / args.input_dir
    output_dir = PROJECT_ROOT / args.output_dir
    
    if not input_dir.exists():
        print(f"Error: Input directory {input_dir} does not exist")
        sys.exit(1)
    
    # Find all MP4 videos
    videos = list(input_dir.glob("*.mp4"))
    
    if not videos:
        print(f"No MP4 videos found in {input_dir}")
        sys.exit(1)
    
    print(f"Found {len(videos)} videos in {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Frame interval: every {args.frame_interval} frames")
    print("-" * 50)
    
    # Extract frames from each video
    total_extracted = 0
    all_frames_dir = output_dir / "all_frames"
    
    for video_path in sorted(videos):
        count = extract_frames_from_video(
            video_path,
            all_frames_dir,
            frame_interval=args.frame_interval,
            max_frames=args.max_frames_per_video,
        )
        total_extracted += count
    
    print("-" * 50)
    print(f"Total frames extracted: {total_extracted}")
    
    # Organize by class if requested
    if args.organize:
        organize_by_class(output_dir)
    else:
        print(f"\nRun with --organize flag to organize frames into class folders")
        print(f"Or manually organize the frames in {all_frames_dir}")


if __name__ == "__main__":
    main()
