#!/usr/bin/env python3
"""
Headless Food Waste Detector for Raspberry Pi
==============================================

This script runs food waste detection without any display/GUI.
Designed for Raspberry Pi or any headless Linux system.

Features:
- Captures video from camera (USB webcam or Pi Camera)
- Detects steel utensils and food
- Classifies food type using trained model
- Saves session data to Firebase Firestore
- Logs results to console and file
- Saves annotated frames periodically

Usage:
    python scripts/headless_detector.py [options]

Options:
    --camera 0              Camera device index (default: 0)
    --interval 1.0          Seconds between frame captures (default: 1.0)
    --save-frames           Save annotated frames to disk
    --output-dir output     Directory for saved frames
    --dimension-mm 200      Known utensil dimension in mm for volume calculation
    --depth-mm 30           Assumed food depth in mm
    --log-file detector.log Log file path
    --duration 0            Run for N seconds (0 = run indefinitely until Ctrl+C)
    --video <path>          Process video file instead of camera
    --dry-run               Don't save to Firebase (testing mode)

Example:
    # Run with USB camera, save to Firebase
    python scripts/headless_detector.py --camera 0 --interval 2

    # Process a video file
    python scripts/headless_detector.py --video data/rice.mp4 --dry-run

    # Run for 5 minutes, saving frames
    python scripts/headless_detector.py --duration 300 --save-frames
"""

import argparse
import base64
import json
import logging
import os
import signal
import sys
import time
from collections import Counter
from datetime import datetime, timezone, timedelta
from pathlib import Path
from threading import Event

import cv2
import numpy as np

# Project root for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load environment variables
from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / '.env')

# Import project modules
from src.food_classifier import classify_food
from src.vision_steel import (
    detect_steel_utensil,
    segment_food_in_steel_utensil,
    estimate_volume_from_contour,
    process_frame_steel_utensil,
    draw_detection_overlay,
)

# Firebase imports
import firebase_admin
from firebase_admin import credentials, firestore
from google.api_core import exceptions as google_exceptions


# Constants
IST_TZ = timezone(timedelta(hours=5, minutes=30))
UTC_TZ = timezone.utc
FIRESTORE_COLLECTION = 'utensil_sessions'
EMPTY_FOOD_LABELS = {'nothing', 'none', 'no_food', 'empty', 'empty_plate'}

# Global shutdown event
shutdown_event = Event()


def setup_logging(log_file: str = None, verbose: bool = False):
    """Configure logging for headless operation."""
    level = logging.DEBUG if verbose else logging.INFO
    
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=handlers
    )
    return logging.getLogger(__name__)


def init_firebase():
    """Initialize Firebase connection."""
    if firebase_admin._apps:
        return firestore.client()
    
    # Try different credential sources
    service_account_base64 = os.environ.get('FIREBASE_SERVICE_ACCOUNT_BASE64')
    service_account_json = os.environ.get('FIREBASE_SERVICE_ACCOUNT_JSON')
    credentials_file = os.environ.get('FIREBASE_CREDENTIALS_FILE')
    project_id = os.environ.get('FIREBASE_PROJECT_ID')
    
    cred = None
    
    if service_account_base64:
        try:
            decoded = base64.b64decode(service_account_base64.strip()).decode('utf-8')
            info = json.loads(decoded)
            cred = credentials.Certificate(info)
        except Exception as e:
            raise RuntimeError(f'Invalid FIREBASE_SERVICE_ACCOUNT_BASE64: {e}')
    
    elif service_account_json:
        try:
            info = json.loads(service_account_json.strip())
            cred = credentials.Certificate(info)
        except Exception as e:
            raise RuntimeError(f'Invalid FIREBASE_SERVICE_ACCOUNT_JSON: {e}')
    
    elif credentials_file:
        cred_path = credentials_file.strip()
        if not os.path.isabs(cred_path):
            cred_path = str(PROJECT_ROOT / cred_path)
        if not os.path.exists(cred_path):
            raise RuntimeError(f'FIREBASE_CREDENTIALS_FILE not found: {cred_path}')
        cred = credentials.Certificate(cred_path)
    
    options = {}
    if project_id:
        options['projectId'] = project_id
    
    firebase_admin.initialize_app(credential=cred, options=options or None)
    return firestore.client()


def save_session_to_firebase(db, session_data: dict, logger):
    """Save a detection session to Firebase Firestore."""
    try:
        now = datetime.now(UTC_TZ)
        
        payload = {
            'utensil_type': session_data.get('shape_type', 'unknown'),
            'average_volume_ml': session_data.get('volume_ml'),
            'average_percent_fill': session_data.get('percent_fill'),
            'sample_count': session_data.get('sample_count', 1),
            'detected_food': session_data.get('food_label'),
            'food_confidence': session_data.get('food_confidence'),
            'started_at': session_data.get('started_at', now),
            'ended_at': session_data.get('ended_at', now),
            'created_at': now,
        }
        
        collection = db.collection(FIRESTORE_COLLECTION)
        doc_ref = collection.document()
        doc_ref.set(payload)
        
        logger.info(f"Session saved to Firebase: {doc_ref.id}")
        return doc_ref.id
        
    except google_exceptions.NotFound as e:
        logger.error(f"Firestore database not initialized: {e}")
    except google_exceptions.GoogleAPICallError as e:
        logger.error(f"Firebase API error: {e}")
    except Exception as e:
        logger.error(f"Failed to save to Firebase: {e}")
    
    return None


def save_detection_to_firebase(db, detection_result: dict, logger):
    """Save a single frame detection to Firebase Firestore."""
    try:
        now = datetime.now(UTC_TZ)
        
        payload = {
            'utensil_type': detection_result.get('shape_type', 'unknown'),
            'average_volume_ml': detection_result.get('volume_ml'),
            'average_percent_fill': detection_result.get('percent_fill'),
            'sample_count': 1,
            'detected_food': detection_result.get('food_label'),
            'food_confidence': detection_result.get('food_confidence'),
            'started_at': now,
            'ended_at': now,
            'created_at': now,
            'detection_type': 'single_frame',
        }
        
        collection = db.collection(FIRESTORE_COLLECTION)
        doc_ref = collection.document()
        doc_ref.set(payload)
        
        logger.debug(f"Detection saved to Firebase: {doc_ref.id}")
        return doc_ref.id
        
    except google_exceptions.NotFound as e:
        logger.error(f"Firestore database not initialized: {e}")
    except google_exceptions.GoogleAPICallError as e:
        logger.error(f"Firebase API error: {e}")
    except Exception as e:
        logger.error(f"Failed to save detection to Firebase: {e}")
    
    return None


class SessionTracker:
    """
    Track food detection sessions for aggregation.
    Matches the original UtensilSessionTracker flow from web/app.py:
    
    - When utensil detected with food: accumulate data
    - When utensil detected but empty (empty_detected): after 3 consecutive → finalize & save
    - When no utensil detected: count missed frames, don't finalize yet
    """
    
    def __init__(self, empty_frames_to_finalize: int = 3, tolerance_frames: int = 2):
        self.empty_frames_to_finalize = empty_frames_to_finalize
        self.tolerance_frames = tolerance_frames
        self.reset()
    
    def reset(self):
        self.active = False
        self.started_at = None
        self.last_seen_at = None
        self.volumes = []
        self.percents = []
        self.food_labels = []
        self.food_confidences = []
        self.shape_type = None
        self.empty_streak = 0
        self.missed_frames = 0
        self.frame_count = 0
    
    def update(self, detection_result: dict, food_label: str = None, 
               food_confidence: float = None) -> dict:
        """
        Update session with new detection.
        Returns session data if session should be finalized, else None.
        
        Logic (matching original web app):
        1. utensil_present + has food → accumulate data
        2. utensil_present + empty (no food detected) → count empty_streak, finalize after 3
        3. no utensil → count missed_frames, finalize after tolerance (utensil left frame)
        """
        now = datetime.now(UTC_TZ)
        
        utensil_detected = detection_result.get('utensil_detected', False)
        percent_fill = detection_result.get('percent_fill')
        volume_ml = detection_result.get('volume_ml')
        shape_type = detection_result.get('shape_type')
        
        # Determine if plate is empty:
        # 1. Food classification says empty (nothing, none, empty, etc.)
        # 2. OR fill percentage is very low (< 2%) indicating empty or nearly empty
        food_label_lower = food_label.lower() if food_label else ''
        empty_detected = (
            food_label_lower in EMPTY_FOOD_LABELS or
            (percent_fill is not None and percent_fill < 2.0)
        )
        
        # Start session if utensil detected and not empty
        if utensil_detected and not empty_detected:
            if not self.active:
                # Start new session
                self.active = True
                self.started_at = now
                self.shape_type = shape_type
        
        # If session not active, nothing to track
        if not self.active:
            self.empty_streak = 0
            return None
        
        # Utensil detected
        if utensil_detected:
            self.missed_frames = 0
            self.last_seen_at = now
            self.frame_count += 1
            
            # If empty detected (plate there but no food)
            if empty_detected:
                self.empty_streak += 1
                if self.empty_streak >= self.empty_frames_to_finalize:
                    return self._finalize(now)
                return None
            
            # Reset empty streak - we have food
            self.empty_streak = 0
            
            # Accumulate data
            if percent_fill is not None:
                self.percents.append(percent_fill)
            if volume_ml is not None:
                self.volumes.append(volume_ml)
            if food_label:
                self.food_labels.append(food_label)
                if food_confidence is not None:
                    self.food_confidences.append(food_confidence)
            
            return None
        
        else:
            # Utensil NOT detected - temporarily missing, don't finalize
            self.missed_frames += 1
            
            # If utensil missing for too long, finalize session
            if self.missed_frames > self.tolerance_frames:
                return self._finalize(now)
            
            return None
    
    def _finalize(self, ended_at: datetime) -> dict:
        """Finalize and return session data."""
        if not self.active:
            self.reset()
            return None
        
        # Need at least some volume or percent data to save
        volume_values = [v for v in self.volumes if v is not None]
        percent_values = [p for p in self.percents if p is not None]
        
        if not volume_values and not percent_values:
            self.reset()
            return None
        
        # Calculate averages
        avg_volume = sum(volume_values) / len(volume_values) if volume_values else None
        avg_percent = sum(percent_values) / len(percent_values) if percent_values else None
        
        sample_count = max(self.frame_count, len(volume_values), len(percent_values))
        
        # Most common food label
        food_label = None
        if self.food_labels:
            counts = Counter(self.food_labels)
            food_label = counts.most_common(1)[0][0]
        
        # Average confidence
        avg_confidence = None
        if self.food_confidences:
            avg_confidence = sum(self.food_confidences) / len(self.food_confidences)
        
        session = {
            'shape_type': self.shape_type,
            'volume_ml': avg_volume,
            'percent_fill': avg_percent,
            'food_label': food_label,
            'food_confidence': avg_confidence,
            'sample_count': sample_count,
            'started_at': self.started_at,
            'ended_at': self.last_seen_at or ended_at,
        }
        
        self.reset()
        return session
    
    def force_finalize(self) -> dict:
        """Force finalize current session (e.g., on shutdown)."""
        if self.active:
            return self._finalize(datetime.now(UTC_TZ))
        return None


def process_frame(frame, dimension_mm: float, depth_mm: float, logger) -> dict:
    """Process a single frame and return detection results."""
    result = {
        'utensil_detected': False,
        'shape_type': None,
        'container_type': None,
        'percent_fill': None,
        'volume_ml': None,
        'food_label': None,
        'food_confidence': None,
    }
    
    try:
        # Food classification
        classification = classify_food(frame)
        if classification:
            result['food_label'], result['food_confidence'] = classification
        
        # Steel utensil detection (uses predefined container dimensions)
        # dimension_mm is now optional - containers are auto-detected
        detection = process_frame_steel_utensil(
            frame,
            known_dimension_mm=dimension_mm if dimension_mm > 0 else None,
            assumed_depth_mm=depth_mm,
            utensil_hint='auto'
        )
        
        result['utensil_detected'] = detection.get('utensil_detected', False)
        result['shape_type'] = detection.get('shape_type')
        result['container_type'] = detection.get('container_type')
        result['percent_fill'] = detection.get('percent_fill')
        result['volume_ml'] = detection.get('volume_ml')
        result['detection'] = detection  # Full detection for drawing
        
    except Exception as e:
        logger.error(f"Frame processing error: {e}")
    
    return result


def draw_frame_overlay(frame, result: dict) -> np.ndarray:
    """Draw detection overlay on frame."""
    detection = result.get('detection', {})
    if detection:
        display = draw_detection_overlay(frame, detection)
    else:
        display = frame.copy()
    
    # Add food classification text
    y = 30
    if result.get('food_label'):
        conf = result.get('food_confidence', 0)
        text = f"Food: {result['food_label']} ({conf*100:.0f}%)"
        cv2.putText(display, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.6, (255, 255, 255), 2)
        y += 25
    
    # Add timestamp
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    cv2.putText(display, timestamp, (10, frame.shape[0] - 10), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    return display


def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    print("\nShutdown signal received...")
    shutdown_event.set()


def run_detector(args, logger):
    """Main detection loop."""
    # Initialize Firebase
    db = None
    if not args.dry_run:
        try:
            db = init_firebase()
            logger.info("Firebase initialized successfully")
        except Exception as e:
            logger.error(f"Firebase initialization failed: {e}")
            if not args.dry_run:
                logger.warning("Continuing in dry-run mode (no Firebase saves)")
                args.dry_run = True
    
    # Open video source
    if args.video:
        cap = cv2.VideoCapture(str(args.video))
        source_name = args.video
        is_video_file = True
    else:
        cap = cv2.VideoCapture(args.camera)
        source_name = f"Camera {args.camera}"
        is_video_file = False
    
    if not cap.isOpened():
        logger.error(f"Failed to open {source_name}")
        return 1
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if is_video_file else 0
    
    logger.info(f"Source: {source_name}")
    if is_video_file:
        logger.info(f"Total frames: {total_frames}, FPS: {fps:.1f}")
    logger.info(f"Capture interval: {args.interval}s")
    logger.info(f"Dry run: {args.dry_run}")
    
    # Setup output directory
    output_dir = None
    if args.save_frames:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving frames to: {output_dir}")
    
    # Initialize tracker
    tracker = SessionTracker(empty_frames_to_finalize=3)
    
    # Statistics
    stats = {
        'frames_processed': 0,
        'sessions_saved': 0,
        'start_time': time.time(),
    }
    
    frame_interval = int(fps * args.interval) if is_video_file else 1
    frame_idx = 0
    last_capture_time = 0
    
    logger.info("Starting detection loop... (Press Ctrl+C to stop)")
    
    try:
        while not shutdown_event.is_set():
            # Check duration limit
            if args.duration > 0:
                elapsed = time.time() - stats['start_time']
                if elapsed >= args.duration:
                    logger.info(f"Duration limit reached ({args.duration}s)")
                    break
            
            ret, frame = cap.read()
            if not ret:
                if is_video_file:
                    logger.info("End of video file")
                    break
                else:
                    logger.warning("Failed to capture frame, retrying...")
                    time.sleep(0.5)
                    continue
            
            frame_idx += 1
            
            # Control capture rate
            if is_video_file:
                if frame_idx % frame_interval != 0:
                    continue
            else:
                current_time = time.time()
                if current_time - last_capture_time < args.interval:
                    continue
                last_capture_time = current_time
            
            # Process frame
            result = process_frame(frame, args.dimension_mm, args.depth_mm, logger)
            stats['frames_processed'] += 1
            
            # Log detection
            if result['utensil_detected']:
                fill = result.get('percent_fill') or 0
                food = result.get('food_label') or 'unknown'
                shape = result.get('shape_type') or 'unknown'
                container = result.get('container_type') or 'unknown'
                volume = result.get('volume_ml')
                vol_str = f", Vol: {volume:.0f}ml" if volume else ""
                logger.info(f"Frame {frame_idx}: {shape}, {container}, Fill: {fill:.1f}%{vol_str}, Food: {food}")
            else:
                logger.debug(f"Frame {frame_idx}: No utensil detected")
            
            # Update session tracker
            session = tracker.update(
                result,
                food_label=result.get('food_label'),
                food_confidence=result.get('food_confidence')
            )
            
            # Save session if finalized (session-based tracking)
            if session and not args.dry_run and db:
                doc_id = save_session_to_firebase(db, session, logger)
                if doc_id:
                    stats['sessions_saved'] += 1
                    logger.info(f"Session finalized - Food: {session.get('food_label')}, "
                              f"Avg Fill: {session.get('percent_fill', 0):.1f}%, "
                              f"Samples: {session.get('sample_count')}")
            
            # Save each detection immediately if requested
            if args.save_each_detection and not args.dry_run and db and result['utensil_detected']:
                doc_id = save_detection_to_firebase(db, result, logger)
                if doc_id:
                    stats['sessions_saved'] += 1
            
            # Save frame if requested
            if output_dir and result['utensil_detected']:
                annotated = draw_frame_overlay(frame, result)
                filename = f"frame_{frame_idx:06d}_{datetime.now().strftime('%H%M%S')}.jpg"
                cv2.imwrite(str(output_dir / filename), annotated)
    
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
    
    finally:
        # Finalize any remaining session
        final_session = tracker.force_finalize()
        if final_session and not args.dry_run and db:
            save_session_to_firebase(db, final_session, logger)
            stats['sessions_saved'] += 1
        
        cap.release()
        
        # Print summary
        elapsed = time.time() - stats['start_time']
        logger.info("=" * 50)
        logger.info("Detection Summary")
        logger.info("=" * 50)
        logger.info(f"Run time: {elapsed:.1f}s")
        logger.info(f"Frames processed: {stats['frames_processed']}")
        logger.info(f"Sessions saved: {stats['sessions_saved']}")
        if stats['frames_processed'] > 0:
            logger.info(f"Avg rate: {stats['frames_processed']/elapsed:.2f} frames/sec")
    
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Headless Food Waste Detector for Raspberry Pi",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with USB camera
  python scripts/headless_detector.py --camera 0

  # Process video file (dry run)
  python scripts/headless_detector.py --video data/rice.mp4 --dry-run

  # Run for 5 minutes, saving frames
  python scripts/headless_detector.py --duration 300 --save-frames

  # Raspberry Pi with Pi Camera
  python scripts/headless_detector.py --camera 0 --interval 2 --save-frames
"""
    )
    
    parser.add_argument(
        '--camera', type=int, default=0,
        help='Camera device index (default: 0)'
    )
    parser.add_argument(
        '--video', type=str, default=None,
        help='Process video file instead of camera'
    )
    parser.add_argument(
        '--interval', type=float, default=1.0,
        help='Seconds between frame captures (default: 1.0)'
    )
    parser.add_argument(
        '--duration', type=float, default=0,
        help='Run for N seconds (0 = indefinitely, default: 0)'
    )
    parser.add_argument(
        '--dimension-mm', type=float, default=200,
        help='Known utensil dimension in mm (default: 200)'
    )
    parser.add_argument(
        '--depth-mm', type=float, default=30,
        help='Assumed food depth in mm (default: 30)'
    )
    parser.add_argument(
        '--save-frames', action='store_true',
        help='Save annotated frames to disk'
    )
    parser.add_argument(
        '--output-dir', type=str, default='output',
        help='Directory for saved frames (default: output)'
    )
    parser.add_argument(
        '--log-file', type=str, default=None,
        help='Log file path (default: console only)'
    )
    parser.add_argument(
        '--dry-run', action='store_true',
        help="Don't save to Firebase (testing mode)"
    )
    parser.add_argument(
        '--save-each-detection', action='store_true',
        help="Save each frame's detection to Firebase (instead of waiting for session finalization)"
    )
    parser.add_argument(
        '--verbose', '-v', action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.log_file, args.verbose)
    
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    logger.info("=" * 50)
    logger.info("Food Waste Detector - Headless Mode")
    logger.info("=" * 50)
    
    # Run detector
    return run_detector(args, logger)


if __name__ == '__main__':
    sys.exit(main())
