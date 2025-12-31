"""
Food Waste Detection Training Pipeline
=======================================

This script orchestrates the entire training pipeline:
1. Extract frames from video files
2. Organize frames into class folders
3. Train the food classification model

Usage:
    python scripts/run_training_pipeline.py [--frame-interval 30] [--epochs 15]
"""

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")
    print(f"Running: {' '.join(cmd)}\n")
    
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    
    if result.returncode != 0:
        print(f"\nError: {description} failed with exit code {result.returncode}")
        sys.exit(result.returncode)
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Run the complete food waste detection training pipeline"
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
        help="Output directory for training data (default: training_data)",
    )
    parser.add_argument(
        "--frame-interval",
        type=int,
        default=30,
        help="Extract every Nth frame (default: 30)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=15,
        help="Number of training epochs (default: 15)",
    )
    parser.add_argument(
        "--fine-tune-epochs",
        type=int,
        default=10,
        help="Number of fine-tuning epochs (default: 10)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for training (default: 32)",
    )
    parser.add_argument(
        "--skip-extraction",
        action="store_true",
        help="Skip frame extraction (use existing training data)",
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip model training (only extract frames)",
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("Food Waste Detection Training Pipeline")
    print("="*60)
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Frame interval: {args.frame_interval}")
    print(f"Epochs: {args.epochs} + {args.fine_tune_epochs} fine-tune")
    
    python_cmd = sys.executable
    
    # Step 1: Extract frames from videos
    if not args.skip_extraction:
        extract_cmd = [
            python_cmd,
            str(PROJECT_ROOT / "scripts" / "extract_frames.py"),
            "--input-dir", args.input_dir,
            "--output-dir", args.output_dir,
            "--frame-interval", str(args.frame_interval),
            "--organize",
        ]
        run_command(extract_cmd, "Step 1: Extracting frames from videos")
    else:
        print("\nSkipping frame extraction (--skip-extraction flag)")
    
    # Step 2: Train model
    if not args.skip_training:
        train_cmd = [
            python_cmd,
            str(PROJECT_ROOT / "scripts" / "train_model.py"),
            "--data-dir", args.output_dir,
            "--epochs", str(args.epochs),
            "--fine-tune-epochs", str(args.fine_tune_epochs),
            "--batch-size", str(args.batch_size),
        ]
        run_command(train_cmd, "Step 2: Training food classification model")
    else:
        print("\nSkipping model training (--skip-training flag)")
    
    # Summary
    print("\n" + "="*60)
    print("Pipeline Complete!")
    print("="*60)
    
    if not args.skip_training:
        model_path = PROJECT_ROOT / "keras_model.h5"
        labels_path = PROJECT_ROOT / "labels.txt"
        
        if model_path.exists():
            print(f"\n✓ Model saved: {model_path}")
        if labels_path.exists():
            print(f"✓ Labels saved: {labels_path}")
        
        print("\nTo use the new model:")
        print("  1. The model and labels are already in the project root")
        print("  2. Restart the web app to load the new model")
        print("  3. Test with the /process endpoint")


if __name__ == "__main__":
    main()
