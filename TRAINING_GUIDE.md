# Food Waste Detection Training Guide

This guide explains how to train a new food classification model and use the steel utensil detection system.

## Overview

The system now supports:
- **7 Indian food types**: chicken_curry, corn_palak, dal_tadka, gobhi_mattar, kheer, paneer_makhni, rice
- **Steel utensil detection** with various shapes (round, rectangular, irregular)
- **Volume estimation** for non-circular vessels

## Dataset

Videos are in the `data/` folder:
- `chicken_curry.mp4`
- `corn_palak.mp4`
- `dal_tadka.mp4`
- `gobhi_mattar.mp4`
- `kheer.mp4`
- `paneer_makhni.mp4`
- `rice.mp4`

## Quick Start

### Option 1: Run Full Pipeline
```bash
python scripts/run_training_pipeline.py --frame-interval 5
```

### Option 2: Run Steps Individually

#### Step 1: Extract Frames from Videos
```bash
python scripts/extract_frames.py --frame-interval 5 --organize
```

**Options:**
- `--input-dir data` - Directory containing MP4 videos
- `--output-dir training_data` - Output directory for extracted frames
- `--frame-interval 5` - Extract every 5th frame (adjust for more/fewer images)
- `--organize` - Automatically organize frames into class folders

**Output:** Creates `training_data/` with subfolders for each food type containing extracted frames.

#### Step 2: Train the Model
```bash
python scripts/train_model.py --epochs 10 --fine-tune-epochs 5 --batch-size 16
```

**Options:**
- `--data-dir training_data` - Directory containing organized training images
- `--epochs 10` - Number of initial training epochs
- `--fine-tune-epochs 5` - Number of fine-tuning epochs
- `--batch-size 16` - Batch size (reduce if running out of memory)
- `--validation-split 0.2` - Fraction of data for validation

**Output:**
- `keras_model.h5` - Trained model file
- `labels.txt` - Class labels
- `training_history.png` - Training accuracy/loss plots
- `training_metadata.json` - Training information

## Testing Steel Utensil Detection

### Test on Single Video
```bash
python scripts/test_detection.py data/rice.mp4
```

### Test on All Videos
```bash
python scripts/test_detection.py --all
```

**Options:**
- `--output-dir test_output` - Directory to save test frames
- `--no-preview` - Disable preview window (automatically set for opencv-headless)

**Output:** Sample frames with detection overlays saved to `test_output/`

## Model Architecture

- **Base Model:** MobileNetV2 (pretrained on ImageNet)
- **Custom Layers:**
  - GlobalAveragePooling2D
  - Dense(256, relu) + Dropout(0.5)
  - Dense(128, relu) + Dropout(0.3)
  - Dense(7, softmax) - Output layer
- **Training Strategy:**
  - Phase 1: Train only custom layers (frozen base)
  - Phase 2: Fine-tune top 30 layers of base model
- **Data Augmentation:**
  - Rotation (±30°)
  - Width/Height shift (±20%)
  - Shear, Zoom, Horizontal flip
  - Brightness variation (80-120%)

## Steel Utensil Detection

### Key Features

1. **Utensil Detection** (`detect_steel_utensil`)
   - Uses saturation/brightness analysis to find steel surfaces
   - Combines edge detection for utensil boundaries
   - Detects shapes: round, rectangular, square, irregular

2. **Food Segmentation** (`segment_food_in_steel_utensil`)
   - Saturation-based: Food has higher color saturation than steel
   - Color deviation: Measures distance from neutral gray (steel)
   - Texture analysis: Food has more texture than smooth steel
   - Voting system: Pixel is food if 2+ methods agree

3. **Volume Estimation** (`estimate_volume_from_contour`)
   - Calculates percentage of utensil covered by food
   - Estimates volume using: area × assumed_depth
   - Supports calibration with known utensil dimensions

### Usage in Code

```python
from src.vision_steel import process_frame_steel_utensil, draw_detection_overlay

# Process frame
result = process_frame_steel_utensil(
    frame,
    known_dimension_mm=200,  # Known width/diameter of utensil
    assumed_depth_mm=30,      # Average food depth
    utensil_hint='auto'       # or 'round', 'rectangular', 'steel'
)

# Draw overlay
annotated = draw_detection_overlay(frame, result)

# Access results
if result['utensil_detected']:
    print(f"Shape: {result['shape_type']}")
    print(f"Fill: {result['percent_fill']:.1f}%")
    print(f"Volume: {result['volume_ml']:.0f} ml")
```

## Integration with Web App

The web app (`web/app.py`) automatically uses:
- New trained model (`keras_model.h5`) for food classification
- Steel utensil detection via `src/vision.py` (backward compatible)

The `/process` endpoint:
1. Detects utensil (circular or steel)
2. Segments food region
3. Classifies food type
4. Estimates volume
5. Returns annotated image with metrics

## Adjusting Detection Parameters

### For Better Utensil Detection

Edit `src/vision_steel.py` -> `detect_steel_utensil()`:
```python
# Adjust steel detection thresholds
steel_mask = cv2.inRange(hsv, 
    (0, 0, 80),      # Lower: (H, S, V) - increase V for brighter steel
    (180, 60, 255)   # Upper: (H, S, V) - decrease S for less colorful
)
```

### For Better Food Segmentation

Edit `src/vision_steel.py` -> `segment_food_in_steel_utensil()`:
```python
# Adjust saturation threshold
steel_sat_threshold = 40  # Increase if food has low saturation

# Adjust color deviation threshold
color_thresh = 15  # Decrease for more sensitive detection

# Adjust texture threshold
texture_thresh = 30  # Tune based on food texture
```

### For Better Volume Estimation

Edit `src/vision_steel.py` -> `estimate_volume_from_contour()`:
```python
# Adjust depth scaling
effective_depth = assumed_depth_mm * min(percent_fill / 50.0, 1.5)
# Change 50.0 to adjust depth based on coverage percentage
```

## Improving Model Accuracy

### 1. Get More Training Data
- Extract frames at smaller intervals: `--frame-interval 3`
- Record more videos of each food type
- Include variety: different lighting, angles, quantities

### 2. Add Data Augmentation
Edit `scripts/train_model.py` -> `create_data_generators()` to adjust augmentation parameters

### 3. Train Longer
```bash
python scripts/train_model.py --epochs 20 --fine-tune-epochs 10
```

### 4. Adjust Architecture
Edit `scripts/train_model.py` -> `create_model()` to modify layer sizes or add more layers

## Troubleshooting

### Issue: Low Detection Accuracy
- Check test images in `test_output/` to see what's being detected
- Adjust detection thresholds in `src/vision_steel.py`
- Ensure videos have good lighting and clear utensil visibility

### Issue: Wrong Food Classification
- Train with more images per class (currently ~50-70 per class)
- Extract frames at smaller intervals for more training data
- Ensure training images represent variety (start/middle/end of eating)

### Issue: Incorrect Volume Estimation
- Calibrate with known utensil dimensions (`known_dimension_mm`)
- Adjust `assumed_depth_mm` based on actual food depth
- Fine-tune volume calculation in `estimate_volume_from_contour()`

### Issue: Out of Memory During Training
- Reduce batch size: `--batch-size 8`
- Reduce image resolution (currently 224×224)
- Close other applications

## File Structure

```
food_waste/
├── data/                      # Video files (.mp4)
├── training_data/             # Extracted training images
│   ├── chicken_curry/
│   ├── corn_palak/
│   └── ...
├── test_output/              # Test detection output
├── scripts/
│   ├── extract_frames.py     # Video → Images
│   ├── train_model.py        # Train classifier
│   ├── test_detection.py     # Test detection
│   └── run_training_pipeline.py  # Run all steps
├── src/
│   ├── food_classifier.py    # Food classification
│   ├── vision.py             # Main vision module
│   └── vision_steel.py       # Steel utensil detection
├── web/
│   └── app.py                # Flask web application
├── keras_model.h5            # Trained model
├── labels.txt                # Food class labels
└── requirements.txt          # Python dependencies
```

## Current Results

- **Training Accuracy:** 100%
- **Validation Accuracy:** 100%
- **Classes:** 7 Indian food types
- **Training Images:** 429 total (346 train, 83 validation)
- **Model Size:** 13.9 MB

## Next Steps

1. **Collect More Data:** Record additional videos with more variety
2. **Test in Production:** Run the web app and test with live video feed
3. **Fine-tune Detection:** Adjust parameters based on actual performance
4. **Add Empty Plate Detection:** Train an "empty" or "nothing" class
5. **Optimize Performance:** Profile and optimize for real-time detection

## Notes

- The system uses `opencv-python-headless` which doesn't support GUI preview
- All test outputs are saved to disk for inspection
- The steel detection works best with consistent lighting
- Volume estimation is approximate and should be calibrated per utensil type
