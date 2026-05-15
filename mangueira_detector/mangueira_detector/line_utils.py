"""
Line detection utilities for mangueira detector.
Implements slope/intercept extraction and clustering methods inspired by lanes.py.
"""

import math
import cv2
import numpy as np


def lines_to_slope_intercept(lines):
    """
    Convert HoughLinesP output to list of (slope, intercept, length) tuples.
    
    Args:
        lines: Output from cv2.HoughLinesP, shape (N, 1, 4) or None
    
    Returns:
        List of tuples: [(slope, intercept, length, x1, y1, x2, y2), ...]
    """
    if lines is None or len(lines) == 0:
        return []
    
    result = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        
        # Compute length
        length = math.hypot(x2 - x1, y2 - y1)
        if length < 1.0:  # Skip degenerate lines
            continue
        
        # Compute slope and intercept using polyfit
        # y = mx + b => m = slope, b = intercept
        if x2 - x1 != 0:
            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - slope * x1
        else:
            # Vertical line: use large slope
            slope = float('inf')
            intercept = x1
        
        result.append((slope, intercept, length, x1, y1, x2, y2))
    
    return result


def cluster_lines_by_slope(line_data, slope_threshold=0.5):
    """
    Cluster lines by similar slope values.
    
    Args:
        line_data: List of (slope, intercept, length, ...) tuples
        slope_threshold: Max slope difference to group lines together
    
    Returns:
        List of clusters, each cluster is a list of line_data tuples
    """
    if not line_data:
        return []
    
    # Handle vertical lines separately
    vertical_lines = [l for l in line_data if math.isinf(l[0])]
    non_vertical = [l for l in line_data if not math.isinf(l[0])]
    
    clusters = []
    
    # Cluster non-vertical lines
    used = set()
    for i, line_i in enumerate(non_vertical):
        if i in used:
            continue
        
        cluster = [line_i]
        used.add(i)
        
        slope_i = line_i[0]
        for j in range(i + 1, len(non_vertical)):
            if j in used:
                continue
            line_j = non_vertical[j]
            slope_j = line_j[0]
            
            if abs(slope_i - slope_j) < slope_threshold:
                cluster.append(line_j)
                used.add(j)
        
        clusters.append(cluster)
    
    # Add vertical lines as separate cluster if any
    if vertical_lines:
        clusters.append(vertical_lines)
    
    return clusters


def cluster_lines_by_angle(line_data, angle_tolerance=0.2):
    """
    Cluster lines by similar angle (computed from slope).
    Handles wrap-around and is more robust than slope-based clustering.
    
    Args:
        line_data: List of (slope, intercept, length, x1, y1, x2, y2) tuples
        angle_tolerance: Max angle difference (radians) to group lines
    
    Returns:
        List of clusters
    """
    if not line_data:
        return []
    
    def slope_to_angle(slope):
        """Convert slope to angle in [-pi/2, pi/2]."""
        if math.isinf(slope):
            return 0.0  # Vertical line
        angle = math.atan(slope)
        # Normalize to [-pi/2, pi/2]
        while angle > math.pi / 2:
            angle -= math.pi
        while angle < -math.pi / 2:
            angle += math.pi
        return angle
    
    # Compute angles
    angles = [slope_to_angle(l[0]) for l in line_data]
    
    clusters = []
    used = set()
    
    for i in range(len(line_data)):
        if i in used:
            continue
        
        cluster = [line_data[i]]
        used.add(i)
        angle_i = angles[i]
        
        for j in range(i + 1, len(line_data)):
            if j in used:
                continue
            angle_j = angles[j]
            
            # Compute angle difference with wrap-around handling
            angle_diff = abs(angle_i - angle_j)
            angle_diff = min(angle_diff, math.pi - angle_diff)
            
            if angle_diff < angle_tolerance:
                cluster.append(line_data[j])
                used.add(j)
        
        clusters.append(cluster)
    
    return clusters


def average_cluster(cluster):
    """
    Average a cluster of lines using weighted average (weight = length).
    
    Args:
        cluster: List of (slope, intercept, length, x1, y1, x2, y2) tuples
    
    Returns:
        Tuple: (avg_slope, avg_intercept, total_length, center_x, center_y)
    """
    if not cluster:
        return None
    
    total_length = sum(l[2] for l in cluster)
    if total_length < 1.0:
        return None

    vertical = [l for l in cluster if math.isinf(l[0])]
    non_vertical = [l for l in cluster if not math.isinf(l[0])]
    len_v = sum(l[2] for l in vertical)
    len_nv = sum(l[2] for l in non_vertical)

    # Geometric center of all line endpoints
    all_x = []
    all_y = []
    for line in cluster:
        _, _, _, x1, y1, x2, y2 = line
        all_x.extend([x1, x2])
        all_y.extend([y1, y2])
    
    center_x = np.mean(all_x)
    center_y = np.mean(all_y)

    # Slope/intercept for y = m x + b. For vertical Hough segments, intercept stores x (see lines_to_slope_intercept).
    if not vertical:
        avg_slope = sum(l[0] * l[2] for l in non_vertical) / len_nv
        avg_intercept = sum(l[1] * l[2] for l in non_vertical) / len_nv
    elif not non_vertical:
        avg_slope = float('inf')
        avg_intercept = sum(l[1] * l[2] for l in vertical) / len_v
    else:
        # Mixed vertical + finite-slope: pick representation by dominated edge length
        if len_nv >= len_v:
            avg_slope = sum(l[0] * l[2] for l in non_vertical) / len_nv
            avg_intercept = sum(l[1] * l[2] for l in non_vertical) / len_nv
        else:
            avg_slope = float('inf')
            avg_intercept = sum(l[1] * l[2] for l in vertical) / len_v

    return (avg_slope, avg_intercept, total_length, center_x, center_y)


def line_extremes_through_center(cluster, center_x, center_y, width, height, y_top_frac=0.3):
    """
    Segment along length-weighted mean direction of cluster segments, passing through
    (center_x, center_y), clipped to image. Robust for mixed vertical / near-vertical clusters.
    """
    y_top = int(height * y_top_frac)
    y_bottom = height - 1

    ux = uy = 0.0
    wsum = 0.0
    for line in cluster:
        _, _, length, x1, y1, x2, y2 = line
        ddx = float(x2 - x1)
        ddy = float(y2 - y1)
        nrm = math.hypot(ddx, ddy)
        if nrm < 1e-6:
            continue
        ux += (ddx / nrm) * length
        uy += (ddy / nrm) * length
        wsum += length
    if wsum < 1e-6:
        return None
    ux /= wsum
    uy /= wsum
    nrm = math.hypot(ux, uy)
    if nrm < 1e-6:
        return None
    ux /= nrm
    uy /= nrm

    def clamp_xy(x, y):
        xi = int(round(x))
        yi = int(round(y))
        xi = max(0, min(width - 1, xi))
        yi = max(0, min(height - 1, yi))
        return xi, yi

    if abs(uy) > 1e-6:
        t1 = (y_top - center_y) / uy
        t2 = (y_bottom - center_y) / uy
        x1 = center_x + t1 * ux
        x2 = center_x + t2 * ux
        p1 = clamp_xy(x1, y_top)
        p2 = clamp_xy(x2, y_bottom)
        return (p1[0], p1[1], p2[0], p2[1])

    # Nearly horizontal direction: span by x
    if abs(ux) < 1e-6:
        return (*clamp_xy(center_x, y_top), *clamp_xy(center_x, y_bottom))
    t0 = (0.0 - center_x) / ux
    t1 = (float(width - 1) - center_x) / ux
    y0 = center_y + t0 * uy
    y1 = center_y + t1 * uy
    p0 = clamp_xy(0.0, y0)
    p1 = clamp_xy(float(width - 1), y1)
    return (p0[0], p0[1], p1[0], p1[1])


def make_line_coordinates(slope, intercept, y1, y2):
    """
    Reconstruct (x1, y1, x2, y2) from slope and intercept.
    Matches lanes.py make_coordinates logic.
    
    Args:
        slope: Line slope (m in y = mx + b)
        intercept: Line intercept (b in y = mx + b)
        y1: Starting y coordinate
        y2: Ending y coordinate
    
    Returns:
        Tuple: (x1, y1, x2, y2)
    """
    if math.isinf(slope):
        # Vertical line
        return (int(intercept), int(y1), int(intercept), int(y2))
    
    if abs(slope) < 1e-6:
        # Horizontal line
        x1 = 0
        x2 = 1000
        return (x1, int(y1), x2, int(y2))
    
    # General case: x = (y - b) / m
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    
    return (x1, int(y1), x2, int(y2))


def apply_roi_mask(frame, roi_type='trapezoid', roi_params=None):
    """
    Apply region of interest mask to frame.
    
    Args:
        frame: Input image (grayscale or BGR)
        roi_type: 'trapezoid' or 'rectangle'
        roi_params: dict with ROI parameters:
            - For 'trapezoid': {'top_fraction': 0.3, 'bottom_fraction': 0.95, 
                               'left_fraction': 0.1, 'right_fraction': 0.9}
            - For 'rectangle': {'y_min_frac': 0.5, 'y_max_frac': 1.0, 
                               'x_min_frac': 0.0, 'x_max_frac': 1.0}
    
    Returns:
        Masked frame (same shape and channels as input)
    """
    if roi_params is None:
        roi_params = {}
    
    height, width = frame.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    
    if roi_type == 'trapezoid':
        # Trapezoid ROI: wider at bottom, narrower at top
        top_frac = roi_params.get('top_fraction', 0.3)
        bottom_frac = roi_params.get('bottom_fraction', 1.0)
        left_frac = roi_params.get('left_fraction', 0.1)
        right_frac = roi_params.get('right_fraction', 0.9)
        
        y_top = int(height * top_frac)
        y_bottom = int(height * bottom_frac)
        
        # Trapezoid vertices (narrower at top)
        x_left_top = int(width * (left_frac + (1 - right_frac + left_frac) / 2 * (1 - top_frac)))
        x_right_top = int(width * (right_frac - (1 - right_frac + left_frac) / 2 * (1 - top_frac)))
        x_left_bottom = int(width * left_frac)
        x_right_bottom = int(width * right_frac)
        
        points = np.array([
            [x_left_top, y_top],
            [x_right_top, y_top],
            [x_right_bottom, y_bottom],
            [x_left_bottom, y_bottom]
        ], dtype=np.int32)
        
        cv2.fillPoly(mask, [points], 255)
    
    elif roi_type == 'rectangle':
        # Rectangle ROI
        y_min_frac = roi_params.get('y_min_frac', 0.5)
        y_max_frac = roi_params.get('y_max_frac', 1.0)
        x_min_frac = roi_params.get('x_min_frac', 0.0)
        x_max_frac = roi_params.get('x_max_frac', 1.0)
        
        y_min = int(height * y_min_frac)
        y_max = int(height * y_max_frac)
        x_min = int(width * x_min_frac)
        x_max = int(width * x_max_frac)
        
        mask[y_min:y_max, x_min:x_max] = 255
    
    else:
        # No ROI
        return frame
    
    if len(frame.shape) == 3:
        # Color image
        masked = cv2.bitwise_and(frame, frame, mask=mask)
    else:
        # Grayscale image
        masked = cv2.bitwise_and(frame, mask)
    
    return masked


def circular_mean(angles):
    """
    Compute circular mean of angles (in radians).
    Properly handles angle wrap-around.
    
    Args:
        angles: List of angles in radians
    
    Returns:
        Mean angle in [-pi, pi]
    """
    if not angles:
        return 0.0
    
    sin_sum = sum(math.sin(a) for a in angles)
    cos_sum = sum(math.cos(a) for a in angles)
    
    mean_angle = math.atan2(sin_sum, cos_sum)
    return mean_angle


def smooth_value_deque(value, deque_obj, use_circular=False):
    """
    Add value to deque and return smoothed (averaged) value.
    
    Args:
        value: New value to add
        deque_obj: collections.deque with max length set
        use_circular: If True, use circular mean (for angles)
    
    Returns:
        Smoothed average value
    """
    deque_obj.append(value)
    
    if use_circular:
        return circular_mean(list(deque_obj))
    else:
        return np.mean(list(deque_obj))
