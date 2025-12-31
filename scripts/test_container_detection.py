#!/usr/bin/env python3
"""Test container detection and volume estimation with exact dimensions."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import cv2
from src.vision_steel import (
    process_frame_steel_utensil,
    BIG_CONTAINER,
    SMALL_CONTAINER,
    identify_container_type,
)

def test_video(video_path: str):
    """Test detection on a video and show container info."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open {video_path}")
        return
    
    print(f"\nVideo: {Path(video_path).name}")
    print("=" * 60)
    print(f"Container Dimensions:")
    print(f"  Big (square):     {BIG_CONTAINER['width_mm']/10:.1f} x {BIG_CONTAINER['length_mm']/10:.1f} x {BIG_CONTAINER['height_mm']/10:.1f} cm")
    print(f"                    Max volume: {BIG_CONTAINER['volume_ml']:.0f} ml")
    print(f"  Small (rectangle): {SMALL_CONTAINER['width_mm']/10:.1f} x {SMALL_CONTAINER['length_mm']/10:.1f} x {SMALL_CONTAINER['height_mm']/10:.1f} cm")
    print(f"                    Max volume: {SMALL_CONTAINER['volume_ml']:.0f} ml")
    print("-" * 60)
    
    frame_num = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_num % 60 == 0:  # Every 60 frames
            result = process_frame_steel_utensil(frame, debug=True)
            
            if result['utensil_detected']:
                print(f"\nFrame {frame_num}:")
                print(f"  Shape: {result['shape_type']}")
                print(f"  Container: {result.get('container_type', 'unknown')}")
                
                if 'debug_info' in result and 'volume_estimation' in result['debug_info']:
                    vol_info = result['debug_info']['volume_estimation']
                    print(f"  Container dims: {vol_info.get('container_dims', 'N/A')}")
                    print(f"  Detected aspect ratio: {vol_info.get('detected_aspect_ratio', 0):.2f}")
                    print(f"  mm/px: {vol_info.get('mm_per_px', 0):.3f}")
                
                print(f"  Fill: {result['percent_fill']:.1f}%")
                if result['volume_ml']:
                    print(f"  Volume: {result['volume_ml']:.0f} ml")
        
        frame_num += 1
    
    cap.release()
    print("\n" + "=" * 60)

def main():
    data_dir = PROJECT_ROOT / "data"
    videos = list(data_dir.glob("*.mp4"))
    
    if not videos:
        print("No videos found!")
        return
    
    # Test first video
    test_video(str(videos[0]))
    
    # Also test aspect ratio detection
    print("\nAspect Ratio Container Detection Test:")
    print("-" * 40)
    test_ratios = [1.0, 1.1, 1.5, 1.9, 2.0]
    for ar in test_ratios:
        container = identify_container_type(ar)
        print(f"  AR {ar:.1f} → {container['name']}")

if __name__ == "__main__":
    main()
