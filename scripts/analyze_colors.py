"""Analyze video frame colors to find optimal steel utensil detection parameters."""
import cv2
import numpy as np
import sys
import os

def analyze_colors(frame):
    """Analyze color distribution in the frame."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    print("\n=== Color Analysis ===")
    print(f"Frame shape: {frame.shape}")
    
    # Get center region (where utensil likely is)
    h, w = frame.shape[:2]
    center_y, center_x = h // 2, w // 2
    margin = 100
    center_region = frame[center_y-margin:center_y+margin, center_x-margin:center_x+margin]
    center_hsv = hsv[center_y-margin:center_y+margin, center_x-margin:center_x+margin]
    
    print(f"\nCenter region HSV stats:")
    print(f"  H - min: {center_hsv[:,:,0].min()}, max: {center_hsv[:,:,0].max()}, mean: {center_hsv[:,:,0].mean():.1f}")
    print(f"  S - min: {center_hsv[:,:,1].min()}, max: {center_hsv[:,:,1].max()}, mean: {center_hsv[:,:,1].mean():.1f}")
    print(f"  V - min: {center_hsv[:,:,2].min()}, max: {center_hsv[:,:,2].max()}, mean: {center_hsv[:,:,2].mean():.1f}")
    
    # Try different steel detection approaches
    print("\n=== Testing Different Detection Approaches ===")
    
    # Approach 1: Original steel detection
    lower1 = np.array([0, 0, 80])
    upper1 = np.array([180, 60, 200])
    mask1 = cv2.inRange(hsv, lower1, upper1)
    contours1, _ = cv2.findContours(mask1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours1:
        largest1 = max(contours1, key=cv2.contourArea)
        print(f"1. Original (S:0-60, V:80-200): Area={cv2.contourArea(largest1):.0f}")
    
    # Approach 2: Look for metallic shine (high brightness, low saturation)
    lower2 = np.array([0, 0, 120])
    upper2 = np.array([180, 50, 255])
    mask2 = cv2.inRange(hsv, lower2, upper2)
    contours2, _ = cv2.findContours(mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours2:
        largest2 = max(contours2, key=cv2.contourArea)
        print(f"2. High brightness (S:0-50, V:120-255): Area={cv2.contourArea(largest2):.0f}")
    
    # Approach 3: Edge-based detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    kernel = np.ones((5, 5), np.uint8)
    edges_dilated = cv2.dilate(edges, kernel, iterations=2)
    edges_filled = cv2.morphologyEx(edges_dilated, cv2.MORPH_CLOSE, np.ones((20, 20), np.uint8))
    contours3, _ = cv2.findContours(edges_filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours3:
        largest3 = max(contours3, key=cv2.contourArea)
        print(f"3. Edge-based: Area={cv2.contourArea(largest3):.0f}")
    
    # Approach 4: Look for reflective surface (gradient in brightness)
    # Use adaptive thresholding
    gray_blur = cv2.GaussianBlur(gray, (11, 11), 0)
    adaptive = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY, 51, 2)
    contours4, _ = cv2.findContours(adaptive, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours4:
        largest4 = max(contours4, key=cv2.contourArea)
        print(f"4. Adaptive threshold: Area={cv2.contourArea(largest4):.0f}")
    
    # Approach 5: Saturation-based (steel has low saturation)
    _, sat_mask = cv2.threshold(hsv[:,:,1], 40, 255, cv2.THRESH_BINARY_INV)
    # Combined with reasonable brightness
    _, val_mask = cv2.threshold(hsv[:,:,2], 100, 255, cv2.THRESH_BINARY)
    mask5 = cv2.bitwise_and(sat_mask, val_mask)
    kernel = np.ones((7, 7), np.uint8)
    mask5 = cv2.morphologyEx(mask5, cv2.MORPH_CLOSE, kernel)
    mask5 = cv2.morphologyEx(mask5, cv2.MORPH_OPEN, kernel)
    contours5, _ = cv2.findContours(mask5, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours5:
        largest5 = max(contours5, key=cv2.contourArea)
        area5 = cv2.contourArea(largest5)
        hull5 = cv2.convexHull(largest5)
        hull_area5 = cv2.contourArea(hull5)
        solidity5 = area5 / hull_area5 if hull_area5 > 0 else 0
        x, y, w, h = cv2.boundingRect(largest5)
        extent5 = area5 / (w * h) if w * h > 0 else 0
        perimeter5 = cv2.arcLength(largest5, True)
        circ5 = 4 * np.pi * area5 / (perimeter5 ** 2) if perimeter5 > 0 else 0
        print(f"5. Low-saturation + brightness: Area={area5:.0f}, Extent={extent5:.3f}, Solidity={solidity5:.3f}, Circ={circ5:.3f}")
    
    # Save masks for visual inspection
    cv2.imwrite("debug_mask1_original.jpg", mask1)
    cv2.imwrite("debug_mask2_bright.jpg", mask2)
    cv2.imwrite("debug_mask3_edges.jpg", edges_filled)
    cv2.imwrite("debug_mask4_adaptive.jpg", adaptive)
    cv2.imwrite("debug_mask5_lowsat.jpg", mask5)
    cv2.imwrite("debug_center_region.jpg", center_region)
    print("\nSaved debug masks: debug_mask1_original.jpg, debug_mask2_bright.jpg, etc.")
    
    # Return best mask (approach 5 seems promising)
    return mask5, contours5

def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_colors.py <video_file>")
        sys.exit(1)
    
    video_path = sys.argv[1]
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Cannot open {video_path}")
        sys.exit(1)
    
    ret, frame = cap.read()
    if not ret:
        print("Error: Cannot read frame")
        sys.exit(1)
    
    mask, contours = analyze_colors(frame)
    
    if contours:
        largest = max(contours, key=cv2.contourArea)
        debug_frame = frame.copy()
        cv2.drawContours(debug_frame, [largest], -1, (0, 255, 0), 3)
        cv2.imwrite("debug_best_contour.jpg", debug_frame)
        print("Saved: debug_best_contour.jpg")
    
    cap.release()

if __name__ == "__main__":
    main()
