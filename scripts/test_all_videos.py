#!/usr/bin/env python3
"""
Quick test script for steel utensil detection.
Tests on all videos in the data folder and shows detection stats.
"""

import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
from src.vision_steel import detect_steel_utensil, segment_food_in_steel_utensil, estimate_volume_from_contour

def test_video(video_path: str, sample_every: int = 30):
    """Test detection on a video file."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  Error: Cannot open {video_path}")
        return None
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    shapes = []
    fills = []
    frame_num = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_num % sample_every == 0:
            contour, mask, info = detect_steel_utensil(frame)
            
            if contour is not None and mask is not None:
                shapes.append(info.get('shape_type', 'unknown'))
                
                # Get fill percentage
                food_result = segment_food_in_steel_utensil(frame, contour, mask)
                if food_result is not None:
                    # Handle both tuple and single mask returns
                    if isinstance(food_result, tuple):
                        food_mask = food_result[0]
                    else:
                        food_mask = food_result
                    
                    if food_mask is not None and isinstance(food_mask, np.ndarray):
                        utensil_area = cv2.contourArea(contour)
                        food_area = np.count_nonzero(food_mask)
                        fill_pct = (food_area / utensil_area * 100) if utensil_area > 0 else 0
                        fills.append(fill_pct)
                    else:
                        fills.append(0)
                else:
                    fills.append(0)
            else:
                shapes.append('not_detected')
                fills.append(0)
        
        frame_num += 1
    
    cap.release()
    
    return {
        'total_frames': total_frames,
        'sampled_frames': len(shapes),
        'shapes': shapes,
        'fills': fills
    }

def main():
    data_dir = PROJECT_ROOT / "data"
    
    if not data_dir.exists():
        print("Data directory not found!")
        sys.exit(1)
    
    videos = list(data_dir.glob("*.mp4"))
    
    if not videos:
        print("No MP4 videos found in data directory!")
        sys.exit(1)
    
    print("=" * 60)
    print("Steel Utensil Detection Test Summary")
    print("=" * 60)
    
    all_shapes = []
    
    for video_path in sorted(videos):
        print(f"\n{video_path.name}:")
        result = test_video(str(video_path))
        
        if result:
            # Count shapes
            from collections import Counter
            shape_counts = Counter(result['shapes'])
            all_shapes.extend(result['shapes'])
            
            # Calculate stats
            valid_fills = [f for f in result['fills'] if f > 0]
            avg_fill = np.mean(valid_fills) if valid_fills else 0
            
            print(f"  Frames: {result['total_frames']}, Sampled: {result['sampled_frames']}")
            print(f"  Shapes: {dict(shape_counts)}")
            print(f"  Fill (non-zero avg): {avg_fill:.1f}%")
    
    # Overall summary
    print("\n" + "=" * 60)
    print("Overall Detection Summary")
    print("=" * 60)
    
    from collections import Counter
    total_shapes = Counter(all_shapes)
    
    print(f"\nTotal samples: {len(all_shapes)}")
    for shape, count in total_shapes.most_common():
        pct = count / len(all_shapes) * 100
        print(f"  {shape}: {count} ({pct:.1f}%)")
    
    # Calculate detection success rate
    detected = len([s for s in all_shapes if s != 'not_detected'])
    rounded_rect = len([s for s in all_shapes if s == 'rounded_rectangle'])
    
    print(f"\nDetection rate: {detected}/{len(all_shapes)} ({detected/len(all_shapes)*100:.1f}%)")
    print(f"Rounded rectangle rate: {rounded_rect}/{len(all_shapes)} ({rounded_rect/len(all_shapes)*100:.1f}%)")

if __name__ == "__main__":
    main()
