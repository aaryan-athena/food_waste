"""Debug shape detection to understand actual contour properties."""
import cv2
import numpy as np
import sys

def analyze_frame(frame):
    """Analyze a single frame and print detailed shape info."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Steel detection - metallic gray colors
    lower_steel = np.array([0, 0, 80])
    upper_steel = np.array([180, 60, 200])
    mask = cv2.inRange(hsv, lower_steel, upper_steel)
    
    # Clean up mask
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        print("No contours found in steel mask")
        return None
    
    # Find largest contour
    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    perimeter = cv2.arcLength(largest, True)
    
    print(f"\n=== Largest Contour Analysis ===")
    print(f"Area: {area:.0f} pixels")
    print(f"Perimeter: {perimeter:.1f} pixels")
    
    if perimeter > 0:
        circularity = 4 * np.pi * area / (perimeter ** 2)
        print(f"Circularity: {circularity:.4f}")
    
    # Bounding rectangle
    x, y, w, h = cv2.boundingRect(largest)
    rect_area = w * h
    extent = area / rect_area if rect_area > 0 else 0
    print(f"Bounding Rect: {w}x{h}, Area: {rect_area}")
    print(f"Extent (contour_area/rect_area): {extent:.4f}")
    
    # Convex hull
    hull = cv2.convexHull(largest)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0
    print(f"Hull Area: {hull_area:.0f}")
    print(f"Solidity (contour_area/hull_area): {solidity:.4f}")
    
    # Aspect ratio
    aspect_ratio = float(w) / h if h > 0 else 0
    print(f"Aspect Ratio (w/h): {aspect_ratio:.3f}")
    
    # Approximate polygon
    epsilon = 0.02 * perimeter
    approx = cv2.approxPolyDP(largest, epsilon, True)
    vertices = len(approx)
    print(f"Approx Polygon Vertices (epsilon=0.02): {vertices}")
    
    # Try different epsilon values
    for eps_mult in [0.01, 0.03, 0.04, 0.05]:
        eps = eps_mult * perimeter
        approx_test = cv2.approxPolyDP(largest, eps, True)
        print(f"  epsilon={eps_mult}: {len(approx_test)} vertices")
    
    # Rotated rectangle
    rect = cv2.minAreaRect(largest)
    box = cv2.boxPoints(rect)
    box = np.int32(box)
    rot_w, rot_h = rect[1]
    if rot_w > 0 and rot_h > 0:
        rot_aspect = max(rot_w, rot_h) / min(rot_w, rot_h)
        print(f"Rotated Rect: {rot_w:.1f}x{rot_h:.1f}, Aspect: {rot_aspect:.3f}")
    
    # Shape classification based on metrics
    print(f"\n=== Shape Classification ===")
    print(f"For ROUNDED_RECTANGLE, need:")
    print(f"  extent > 0.75: {extent:.4f} -> {'YES' if extent > 0.75 else 'NO'}")
    print(f"  solidity > 0.9: {solidity:.4f} -> {'YES' if solidity > 0.9 else 'NO'}")
    print(f"  0.55 < circularity < 0.85: {circularity:.4f} -> {'YES' if 0.55 < circularity < 0.85 else 'NO'}")
    
    # Determine shape
    if circularity > 0.85:
        shape = "circular"
    elif extent > 0.75 and solidity > 0.9 and circularity > 0.55:
        shape = "rounded_rectangle"
    elif extent > 0.70 and solidity > 0.85:
        shape = "rectangle"
    else:
        shape = "irregular"
    
    print(f"\nDetected shape: {shape}")
    
    return largest, mask

def main():
    if len(sys.argv) < 2:
        print("Usage: python debug_shape.py <video_file>")
        sys.exit(1)
    
    video_path = sys.argv[1]
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Cannot open {video_path}")
        sys.exit(1)
    
    # Read first frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Cannot read frame")
        sys.exit(1)
    
    print(f"Frame size: {frame.shape[1]}x{frame.shape[0]}")
    
    result = analyze_frame(frame)
    
    if result:
        contour, mask = result
        # Save debug images
        debug_frame = frame.copy()
        cv2.drawContours(debug_frame, [contour], -1, (0, 255, 0), 3)
        cv2.imwrite("debug_contour.jpg", debug_frame)
        cv2.imwrite("debug_mask.jpg", mask)
        print("\nSaved: debug_contour.jpg, debug_mask.jpg")
    
    cap.release()

if __name__ == "__main__":
    main()
