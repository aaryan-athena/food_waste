"""
Vision Module for Steel Utensil Detection and Food Volume Estimation
=====================================================================

This module provides functions for:
1. Detecting steel utensils (various shapes: round, rectangular, irregular)
2. Segmenting food regions within utensils
3. Estimating food volume based on utensil dimensions

Key differences from the original vision.py:
- Handles steel/metallic utensils instead of white plates
- Supports non-circular utensil shapes (rectangular, oval, irregular)
- Uses edge detection and reflection-based steel detection

Supported Utensil Dimensions:
- Big container (square): 34 x 31 cm, height 7 cm
- Small container (rectangle): 31.5 x 16.5 cm, height 7 cm
"""

import cv2
import numpy as np
from typing import Optional, Tuple, Dict, Any


# ============================================================================
# UTENSIL DIMENSION CONSTANTS (in mm)
# ============================================================================

# Big container - square with rounded corners
BIG_CONTAINER = {
    'name': 'big_square',
    'width_mm': 340,      # 34 cm
    'length_mm': 310,     # 31 cm
    'height_mm': 70,      # 7 cm
    'aspect_ratio': 340 / 310,  # ~1.097 (close to 1:1)
    'area_mm2': 340 * 310,  # 105,400 mm²
    'volume_ml': (340 * 310 * 70) / 1000,  # ~7378 ml max capacity
}

# Small container - rectangle with rounded corners
SMALL_CONTAINER = {
    'name': 'small_rectangle',
    'width_mm': 315,      # 31.5 cm
    'length_mm': 165,     # 16.5 cm
    'height_mm': 70,      # 7 cm
    'aspect_ratio': 315 / 165,  # ~1.91 (rectangular)
    'area_mm2': 315 * 165,  # 51,975 mm²
    'volume_ml': (315 * 165 * 70) / 1000,  # ~3638 ml max capacity
}

# Threshold to distinguish between square and rectangular container
ASPECT_RATIO_THRESHOLD = 1.5  # Below = square, Above = rectangle


def detect_steel_utensil(
    frame: np.ndarray,
    utensil_hint: str = 'auto',
    min_area: int = 10000,
    debug: bool = False
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Dict[str, Any]]:
    """
    Detect steel utensil in frame using multiple techniques.
    
    Steel utensils have characteristic properties:
    - High reflectivity / specular highlights
    - Grayish metallic color
    - Strong edges
    - Often have distinctive rim/border
    
    Args:
        frame: BGR image
        utensil_hint: Hint about utensil type ('round', 'rectangular', 'auto')
        min_area: Minimum contour area to consider
        debug: If True, return debug info
    
    Returns:
        Tuple of (contour, mask, info_dict)
        - contour: The detected utensil contour (numpy array) or None
        - mask: Binary mask of the utensil interior
        - info_dict: Dictionary with detection metadata
    """
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    info = {
        'method': None,
        'shape_type': None,
        'center': None,
        'bounding_rect': None,
        'area': None,
    }
    
    # === Method 1: Edge-based detection for finding utensil boundaries ===
    # This is more reliable for steel utensils than color-based detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.5)
    edges = cv2.Canny(blurred, 30, 100)
    
    # Dilate edges to connect broken lines
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    edges_dilated = cv2.dilate(edges, kernel, iterations=2)
    
    # Close gaps to form complete contours
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    edges_closed = cv2.morphologyEx(edges_dilated, cv2.MORPH_CLOSE, kernel_close, iterations=3)
    
    # Find contours from edges
    contours_edge, _ = cv2.findContours(edges_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # === Method 2: HSV-based detection for steel regions ===
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h_channel, s_channel, v_channel = cv2.split(hsv)
    
    # Steel mask: low saturation, moderate to high value
    steel_mask = cv2.inRange(hsv, (0, 0, 100), (180, 80, 255))
    
    # Clean up the mask
    steel_mask = cv2.morphologyEx(steel_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    steel_mask = cv2.morphologyEx(steel_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Find contours from steel mask
    contours_steel, _ = cv2.findContours(steel_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Combine all contours
    all_contours = list(contours_edge) + list(contours_steel)
    
    if not all_contours:
        return None, None, info
    
    # Score and select best contour
    best_contour = None
    best_score = 0
    
    for contour in all_contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        
        # Calculate contour properties
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue
        
        # Approximate the contour shape
        epsilon = 0.02 * perimeter
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # Get bounding rect and check aspect ratio
        x, y, bw, bh = cv2.boundingRect(contour)
        aspect_ratio = float(bw) / bh if bh > 0 else 0
        
        # Score based on:
        # 1. Area (larger is better, but not too large - shouldn't fill whole frame)
        # 2. Position (centered is better)
        # 3. Aspect ratio (reasonable shapes)
        
        area_score = min(area / (h * w * 0.5), 1.0)  # Prefer up to 50% of frame
        
        cx = x + bw / 2
        cy = y + bh / 2
        center_dist = np.sqrt((cx - w/2)**2 + (cy - h/2)**2)
        center_score = 1.0 - min(center_dist / (min(h, w) * 0.5), 1.0)
        
        # Aspect ratio score (prefer reasonable shapes)
        aspect_score = 1.0 if 0.3 < aspect_ratio < 3.0 else 0.5
        
        # Solidity score (area / convex hull area)
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
        solidity_score = solidity
        
        total_score = area_score * 0.3 + center_score * 0.3 + aspect_score * 0.2 + solidity_score * 0.2
        
        if total_score > best_score:
            best_score = total_score
            best_contour = contour
    
    if best_contour is None:
        return None, None, info
    
    # Create mask from best contour
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(mask, [best_contour], -1, 255, -1)
    
    # Determine shape type
    area = cv2.contourArea(best_contour)
    perimeter = cv2.arcLength(best_contour, True)
    circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
    
    x, y, bw, bh = cv2.boundingRect(best_contour)
    aspect_ratio = float(bw) / bh if bh > 0 else 1.0
    
    # Calculate extent (how much of bounding rect is filled)
    rect_area = bw * bh
    extent = area / rect_area if rect_area > 0 else 0
    
    # Hull solidity
    hull = cv2.convexHull(best_contour)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0
    
    # Approximate contour for corner detection
    epsilon = 0.02 * perimeter
    approx = cv2.approxPolyDP(best_contour, epsilon, True)
    num_corners = len(approx)
    
    # Rounded rectangle detection:
    # - Moderate to high extent (fills bounding rect well): > 0.60
    # - High solidity (convex shape): > 0.85
    # - Not too circular, not too irregular
    # - 4-8 corners typically for rounded rect approximation
    
    if circularity > 0.85:
        shape_type = 'round'
    elif extent > 0.60 and solidity > 0.85 and 0.4 < circularity < 0.85:
        # Rounded rectangle: fills bounding box well but not perfectly circular
        shape_type = 'rounded_rectangle'
    elif num_corners == 4 and extent > 0.8 and 0.85 < aspect_ratio < 1.15:
        shape_type = 'square'
    elif num_corners == 4 and extent > 0.8:
        shape_type = 'rectangular'
    elif extent > 0.55 and solidity > 0.75:
        # Still likely a rounded rectangle with less ideal detection
        shape_type = 'rounded_rectangle'
    else:
        shape_type = 'irregular'
    
    # Calculate center
    M = cv2.moments(best_contour)
    if M['m00'] > 0:
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        info['center'] = (cx, cy)
    
    info['method'] = 'combined_steel_edge'
    info['shape_type'] = shape_type
    info['bounding_rect'] = (x, y, bw, bh)
    info['area'] = area
    info['circularity'] = circularity
    info['extent'] = extent
    info['solidity'] = solidity
    
    return best_contour, mask, info


def segment_food_in_steel_utensil(
    frame: np.ndarray,
    utensil_contour: np.ndarray,
    utensil_mask: np.ndarray,
    debug: bool = False
) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    """
    Segment food regions within a steel utensil.
    
    Steel utensils require different approach than white plates:
    - Steel is grayish/metallic, food is usually colorful
    - Use color saturation to distinguish food from steel
    - Use texture analysis for food with similar brightness
    
    Args:
        frame: BGR image
        utensil_contour: Contour of the detected utensil
        utensil_mask: Binary mask of utensil interior
        debug: Return debug information
    
    Returns:
        Tuple of (food_mask, info_dict)
    """
    info = {}
    
    if utensil_mask is None or np.count_nonzero(utensil_mask) == 0:
        return None, info
    
    # Convert to different color spaces for analysis
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    
    h_channel, s_channel, v_channel = cv2.split(hsv)
    l_channel, a_channel, b_channel = cv2.split(lab)
    
    # === Method 1: Saturation-based segmentation ===
    # Food typically has higher saturation than steel
    # Sample steel color from the rim/border of the utensil
    
    # Create a rim mask (erode the utensil mask to get interior, subtract)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    interior_mask = cv2.erode(utensil_mask, kernel, iterations=2)
    rim_mask = cv2.subtract(utensil_mask, interior_mask)
    
    # Sample saturation values from rim (should be steel)
    rim_saturation = s_channel[rim_mask > 0]
    if len(rim_saturation) > 100:
        steel_sat_threshold = np.percentile(rim_saturation, 85)
    else:
        steel_sat_threshold = 40  # Default threshold for steel
    
    # Food has higher saturation than steel
    food_by_sat = np.zeros_like(utensil_mask)
    food_by_sat[(s_channel > steel_sat_threshold) & (utensil_mask > 0)] = 255
    
    # === Method 2: Color deviation in LAB space ===
    # Sample the steel color (neutral gray in LAB has a≈128, b≈128)
    rim_a = a_channel[rim_mask > 0]
    rim_b = b_channel[rim_mask > 0]
    
    if len(rim_a) > 100:
        steel_a = np.median(rim_a)
        steel_b = np.median(rim_b)
    else:
        steel_a = 128
        steel_b = 128
    
    # Calculate color deviation from steel
    a_dev = np.abs(a_channel.astype(np.float32) - steel_a)
    b_dev = np.abs(b_channel.astype(np.float32) - steel_b)
    color_dev = np.sqrt(a_dev**2 + b_dev**2)
    
    # Threshold color deviation
    color_thresh = 15  # Pixels with color deviation > 15 are likely food
    food_by_color = np.zeros_like(utensil_mask)
    food_by_color[(color_dev > color_thresh) & (utensil_mask > 0)] = 255
    
    # === Method 3: Texture-based segmentation ===
    # Food often has more texture than smooth steel
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Calculate local variance as texture measure
    blur = cv2.GaussianBlur(gray.astype(np.float32), (5, 5), 0)
    blur_sq = cv2.GaussianBlur(gray.astype(np.float32)**2, (5, 5), 0)
    variance = blur_sq - blur**2
    variance = np.clip(variance, 0, None)
    
    # Normalize variance
    variance_norm = cv2.normalize(variance, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Sample variance from rim
    rim_var = variance_norm[rim_mask > 0]
    if len(rim_var) > 100:
        texture_thresh = np.percentile(rim_var, 75) + 10
    else:
        texture_thresh = 30
    
    food_by_texture = np.zeros_like(utensil_mask)
    food_by_texture[(variance_norm > texture_thresh) & (utensil_mask > 0)] = 255
    
    # === Combine methods ===
    # Use voting: pixel is food if at least 2 methods agree
    combined = (
        (food_by_sat > 0).astype(np.uint8) + 
        (food_by_color > 0).astype(np.uint8) + 
        (food_by_texture > 0).astype(np.uint8)
    )
    food_mask = np.zeros_like(utensil_mask)
    food_mask[combined >= 2] = 255
    
    # Morphological cleanup
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    food_mask = cv2.morphologyEx(food_mask, cv2.MORPH_OPEN, kernel_small, iterations=1)
    food_mask = cv2.morphologyEx(food_mask, cv2.MORPH_CLOSE, kernel_small, iterations=2)
    
    # Remove small noise regions
    contours, _ = cv2.findContours(food_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    food_mask_clean = np.zeros_like(food_mask)
    min_food_area = np.count_nonzero(utensil_mask) * 0.01  # At least 1% of utensil
    
    for contour in contours:
        if cv2.contourArea(contour) >= min_food_area:
            cv2.drawContours(food_mask_clean, [contour], -1, 255, -1)
    
    # Apply GrabCut refinement if we have a reasonable initial mask
    food_pixels = np.count_nonzero(food_mask_clean)
    utensil_pixels = np.count_nonzero(utensil_mask)
    
    if food_pixels > utensil_pixels * 0.05:  # At least 5% coverage
        try:
            gc_mask = np.zeros_like(utensil_mask)
            gc_mask[utensil_mask == 0] = cv2.GC_BGD  # Background
            gc_mask[(utensil_mask > 0) & (food_mask_clean == 0)] = cv2.GC_PR_BGD  # Probable background (steel)
            gc_mask[(utensil_mask > 0) & (food_mask_clean > 0)] = cv2.GC_PR_FGD  # Probable foreground (food)
            
            bgd_model = np.zeros((1, 65), np.float64)
            fgd_model = np.zeros((1, 65), np.float64)
            
            cv2.grabCut(frame, gc_mask, None, bgd_model, fgd_model, 3, cv2.GC_INIT_WITH_MASK)
            
            food_mask_refined = np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
            food_mask_refined[utensil_mask == 0] = 0
            
            # Only use refined mask if it's reasonable
            refined_pixels = np.count_nonzero(food_mask_refined)
            if refined_pixels > food_pixels * 0.3:  # Didn't lose too much
                food_mask_clean = food_mask_refined
        except cv2.error:
            pass  # Keep original mask if GrabCut fails
    
    info['food_pixels'] = np.count_nonzero(food_mask_clean)
    info['utensil_pixels'] = utensil_pixels
    info['coverage_percent'] = (info['food_pixels'] / utensil_pixels * 100) if utensil_pixels > 0 else 0
    
    return food_mask_clean, info


def identify_container_type(aspect_ratio: float) -> Dict[str, Any]:
    """
    Identify which container type based on aspect ratio.
    
    Args:
        aspect_ratio: Width/height ratio of detected bounding box
    
    Returns:
        Container specification dict
    """
    # Normalize aspect ratio (always > 1)
    ar = max(aspect_ratio, 1.0 / aspect_ratio) if aspect_ratio > 0 else 1.0
    
    if ar < ASPECT_RATIO_THRESHOLD:
        # Close to square - big container
        return BIG_CONTAINER.copy()
    else:
        # More rectangular - small container
        return SMALL_CONTAINER.copy()


def estimate_volume_from_contour(
    utensil_contour: np.ndarray,
    utensil_mask: np.ndarray,
    food_mask: np.ndarray,
    utensil_info: Dict[str, Any],
    known_dimension_mm: Optional[float] = None,
    assumed_depth_mm: float = 70.0,  # Default to container height (7cm)
    utensil_hint: str = 'auto'
) -> Tuple[Optional[float], Optional[float], Dict[str, Any]]:
    """
    Estimate food volume based on utensil shape and food coverage.
    
    Uses predefined container dimensions:
    - Big container (square): 34 x 31 cm, height 7 cm
    - Small container (rectangle): 31.5 x 16.5 cm, height 7 cm
    
    The container type is auto-detected based on aspect ratio.
    
    Args:
        utensil_contour: Detected utensil contour
        utensil_mask: Binary mask of utensil
        food_mask: Binary mask of food region
        utensil_info: Info dict from detect_steel_utensil
        known_dimension_mm: Override for known dimension (optional, auto-detected if None)
        assumed_depth_mm: Assumed average food depth in mm (default: 70mm = 7cm)
        utensil_hint: Type hint ('big_square', 'small_rectangle', 'auto')
    
    Returns:
        Tuple of (percent_fill, volume_ml, info_dict)
    """
    info = {}
    
    utensil_pixels = np.count_nonzero(utensil_mask)
    food_pixels = np.count_nonzero(food_mask) if food_mask is not None else 0
    
    if utensil_pixels == 0:
        return None, None, info
    
    # Calculate percent fill
    percent_fill = (food_pixels / utensil_pixels) * 100.0
    info['percent_fill'] = percent_fill
    info['food_pixels'] = food_pixels
    info['utensil_pixels'] = utensil_pixels
    
    # Get bounding rectangle
    x, y, bw, bh = utensil_info.get('bounding_rect', cv2.boundingRect(utensil_contour))
    aspect_ratio = float(bw) / bh if bh > 0 else 1.0
    
    # Identify container type based on aspect ratio or hint
    if utensil_hint == 'big_square':
        container = BIG_CONTAINER.copy()
    elif utensil_hint == 'small_rectangle':
        container = SMALL_CONTAINER.copy()
    else:
        container = identify_container_type(aspect_ratio)
    
    info['container_type'] = container['name']
    info['container_dims'] = f"{container['width_mm']/10:.1f}x{container['length_mm']/10:.1f}x{container['height_mm']/10:.1f} cm"
    
    # Calculate pixel to mm ratio using known container dimensions
    # Use the longer detected side to match with container's longer dimension
    longer_side_px = max(bw, bh)
    shorter_side_px = min(bw, bh)
    
    container_longer = max(container['width_mm'], container['length_mm'])
    container_shorter = min(container['width_mm'], container['length_mm'])
    
    # Use known_dimension_mm if provided, otherwise use container dimensions
    if known_dimension_mm is not None and known_dimension_mm > 0:
        mm_per_px = known_dimension_mm / longer_side_px if longer_side_px > 0 else 0
    else:
        mm_per_px = container_longer / longer_side_px if longer_side_px > 0 else 0
    
    info['mm_per_px'] = mm_per_px
    info['detected_aspect_ratio'] = aspect_ratio
    
    # Volume estimation
    volume_ml = None
    
    if mm_per_px > 0:
        # Food area in mm²
        food_area_mm2 = food_pixels * (mm_per_px ** 2)
        info['food_area_mm2'] = food_area_mm2
        
        # Container area in mm²
        container_area_mm2 = container['area_mm2']
        info['container_area_mm2'] = container_area_mm2
        
        # Calculate fill ratio based on area
        area_fill_ratio = food_area_mm2 / container_area_mm2 if container_area_mm2 > 0 else 0
        info['area_fill_ratio'] = area_fill_ratio
        
        # Use container height for depth calculation
        container_height = container['height_mm']
        
        # Effective depth: assume food depth is proportional to fill
        # - Low fill (<30%): thin layer, ~20% of container height
        # - Medium fill (30-70%): moderate depth, ~40% of container height  
        # - High fill (>70%): deeper, ~60% of container height
        if percent_fill < 30:
            depth_factor = 0.2
        elif percent_fill < 70:
            depth_factor = 0.4
        else:
            depth_factor = 0.6
        
        effective_depth = container_height * depth_factor
        info['effective_depth_mm'] = effective_depth
        
        # Volume = area × depth (mm³)
        volume_mm3 = food_area_mm2 * effective_depth
        info['volume_mm3'] = volume_mm3
        
        # Convert to ml (1 ml = 1000 mm³)
        volume_ml = volume_mm3 / 1000.0
        info['volume_ml'] = volume_ml
        
        # Also calculate as percentage of max container volume
        max_volume_ml = container['volume_ml']
        info['max_container_volume_ml'] = max_volume_ml
        info['volume_percent_of_max'] = (volume_ml / max_volume_ml * 100) if max_volume_ml > 0 else 0
    
    return percent_fill, volume_ml, info


# === Legacy API compatibility functions ===
# These maintain backward compatibility with the original vision.py

def detect_utensil_ellipse(frame, utensil_hint='auto', min_radius=60, max_radius=400, debug=False):
    """
    Legacy function for backward compatibility.
    Detects utensil and returns ellipse format if possible.
    """
    contour, mask, info = detect_steel_utensil(frame, utensil_hint, debug=debug)
    
    if contour is None:
        # Fall back to original circle detection
        return _detect_circle_fallback(frame, min_radius, max_radius, debug)
    
    # Fit ellipse to contour if possible
    if len(contour) >= 5:
        try:
            ellipse = cv2.fitEllipse(contour)
            return ellipse
        except cv2.error:
            pass
    
    # Return bounding ellipse from bounding rect
    x, y, w, h = cv2.boundingRect(contour)
    cx = x + w / 2
    cy = y + h / 2
    return ((cx, cy), (float(w), float(h)), 0.0)


def _detect_circle_fallback(frame, min_radius, max_radius, debug):
    """Original circle detection as fallback."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 1.5)
    
    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=120,
        param1=120, param2=40, minRadius=min_radius, maxRadius=max_radius
    )
    
    if circles is not None and len(circles) > 0:
        circles = np.uint16(np.around(circles))
        h, w = gray.shape[:2]
        cx, cy = w / 2, h / 2
        
        def score(c):
            x, y, r = c
            return np.hypot(x - cx, y - cy) - 0.1 * r
        
        best = sorted(circles[0], key=score)[0]
        x, y, r = best
        return ((float(x), float(y)), (float(2 * r), float(2 * r)), 0.0)
    
    return None


def segment_food_in_utensil(img, ellipse, debug=False):
    """
    Legacy function for backward compatibility.
    Uses new steel utensil segmentation internally.
    """
    if ellipse is None:
        return None, {}
    
    # Create mask from ellipse
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    (cx, cy), (MA, ma), angle = ellipse
    cv2.ellipse(mask, (int(cx), int(cy)), (int(MA/2), int(ma/2)), angle, 0, 360, 255, -1)
    
    # Find contour from ellipse mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = contours[0] if contours else None
    
    # Use new segmentation
    food_mask, info = segment_food_in_steel_utensil(img, contour, mask, debug)
    
    return food_mask, info


def _ellipse_mask(shape, ellipse):
    """Create binary mask from ellipse."""
    mask = np.zeros(shape[:2], dtype=np.uint8)
    (cx, cy), (MA, ma), angle = ellipse
    cv2.ellipse(mask, (int(cx), int(cy)), (int(MA/2), int(ma/2)), angle, 0, 360, 255, -1)
    return mask


def estimate_area_and_volume(ellipse, seg_mask, diameter_mm, utensil_hint, assumed_height_mm):
    """
    Legacy function for backward compatibility.
    Compute percent fill by area and approximate volume.
    """
    if ellipse is None:
        return None, None
    
    (cx, cy), (MA, ma), angle = ellipse
    full_mask = _ellipse_mask(seg_mask.shape + (1,), ellipse)
    interior_px = int(np.count_nonzero(full_mask))
    food_px = int(np.count_nonzero(seg_mask))
    
    if interior_px == 0:
        return None, None
    
    percent_fill = 100.0 * food_px / interior_px
    
    est_volume_ml = None
    if diameter_mm is not None and assumed_height_mm is not None and assumed_height_mm > 0:
        major_axis_px = max(MA, ma)
        if major_axis_px > 0:
            mm_per_px = diameter_mm / major_axis_px
            food_area_mm2 = food_px * (mm_per_px ** 2)
            volume_mm3 = food_area_mm2 * assumed_height_mm
            est_volume_ml = volume_mm3 / 1000.0
    
    return percent_fill, est_volume_ml


# === New API functions ===

def process_frame_steel_utensil(
    frame: np.ndarray,
    known_dimension_mm: Optional[float] = None,
    assumed_depth_mm: float = 70.0,  # Default container height 7cm
    utensil_hint: str = 'auto',
    debug: bool = False
) -> Dict[str, Any]:
    """
    Complete processing pipeline for steel utensil food waste detection.
    
    Container dimensions are auto-detected based on aspect ratio:
    - Big container (square): 34 x 31 cm, height 7 cm
    - Small container (rectangle): 31.5 x 16.5 cm, height 7 cm
    
    Args:
        frame: BGR image frame
        known_dimension_mm: Override for known dimension (auto-detected if None)
        assumed_depth_mm: Assumed food depth in mm (default: 70mm)
        utensil_hint: Hint about utensil ('big_square', 'small_rectangle', 'auto')
        debug: Return debug information
    
    Returns:
        Dictionary with:
        - utensil_detected: bool
        - utensil_contour: numpy array or None
        - utensil_mask: binary mask or None
        - food_mask: binary mask or None
        - percent_fill: float or None
        - volume_ml: float or None
        - shape_type: str or None
        - container_type: str or None (big_square or small_rectangle)
        - debug_info: dict (if debug=True)
    """
    result = {
        'utensil_detected': False,
        'utensil_contour': None,
        'utensil_mask': None,
        'food_mask': None,
        'percent_fill': None,
        'volume_ml': None,
        'shape_type': None,
        'container_type': None,
    }
    
    if debug:
        result['debug_info'] = {}
    
    # Step 1: Detect utensil
    contour, mask, utensil_info = detect_steel_utensil(frame, utensil_hint, debug=debug)
    
    if contour is None:
        return result
    
    result['utensil_detected'] = True
    result['utensil_contour'] = contour
    result['utensil_mask'] = mask
    result['shape_type'] = utensil_info.get('shape_type')
    
    if debug:
        result['debug_info']['utensil_detection'] = utensil_info
    
    # Step 2: Segment food
    food_mask, seg_info = segment_food_in_steel_utensil(frame, contour, mask, debug)
    result['food_mask'] = food_mask
    
    if debug:
        result['debug_info']['food_segmentation'] = seg_info
    
    # Step 3: Estimate volume
    percent_fill, volume_ml, vol_info = estimate_volume_from_contour(
        contour, mask, food_mask, utensil_info,
        known_dimension_mm, assumed_depth_mm, utensil_hint
    )
    
    result['percent_fill'] = percent_fill
    result['volume_ml'] = volume_ml
    result['container_type'] = vol_info.get('container_type')
    
    if debug:
        result['debug_info']['volume_estimation'] = vol_info
    
    return result


def draw_detection_overlay(
    frame: np.ndarray,
    result: Dict[str, Any],
    show_contour: bool = True,
    show_food_mask: bool = True,
    show_info: bool = True
) -> np.ndarray:
    """
    Draw detection results on frame for visualization.
    
    Args:
        frame: Original BGR frame
        result: Result dict from process_frame_steel_utensil
        show_contour: Draw utensil contour
        show_food_mask: Overlay food mask
        show_info: Show text info
    
    Returns:
        Annotated frame
    """
    display = frame.copy()
    
    if not result.get('utensil_detected'):
        if show_info:
            cv2.putText(display, "No utensil detected", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return display
    
    # Draw utensil contour
    if show_contour and result.get('utensil_contour') is not None:
        cv2.drawContours(display, [result['utensil_contour']], -1, (0, 255, 255), 2)
    
    # Overlay food mask
    if show_food_mask and result.get('food_mask') is not None:
        overlay = display.copy()
        overlay[result['food_mask'] > 0] = (0, 0, 255)
        display = cv2.addWeighted(display, 0.7, overlay, 0.3, 0)
    
    # Draw info text
    if show_info:
        y = 30
        texts = []
        
        if result.get('shape_type'):
            texts.append(f"Shape: {result['shape_type']}")
        
        if result.get('container_type'):
            container_name = 'Big (34x31cm)' if result['container_type'] == 'big_square' else 'Small (31.5x16.5cm)'
            texts.append(f"Container: {container_name}")
        
        if result.get('percent_fill') is not None:
            texts.append(f"Fill: {result['percent_fill']:.1f}%")
        
        if result.get('volume_ml') is not None:
            texts.append(f"Volume: {result['volume_ml']:.0f} ml")
        
        for i, text in enumerate(texts):
            cv2.putText(display, text, (10, y + i * 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return display
