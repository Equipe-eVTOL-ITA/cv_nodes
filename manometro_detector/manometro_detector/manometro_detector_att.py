import rclpy
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Float32, Int16MultiArray, String
from geometry_msgs.msg import Point
from detector.detector import Detector

import cv2 as cv
import numpy as np
import os
import time
from datetime import datetime

# Configurações
TEMPLATE_SIZE = 400          # Side length (px) for the canonical square
DIFF_BLUR_KSIZE = 7          # (Median) Blur kernel for diff cleanup
DIFF_THRESHOLD = 30          # Threshold applied after blur on the diff image
EPSILON_RAMER_DOUGLAS_PEUCKER = 0.04 # Valor do epsilon para o algoritmo de RDP
ROI_RADIUS_RATIO = 0.40      # Define a ROI interna (cobre o ponteiro sem incluir números da borda)
ROI_OFFSET_Y = -20           # Deslocamento vertical do centro da ROI (cabinho da moldura)
ROI_OFFSET_X = 0             # Deslocamento horizontal do centro da ROI
MAX_POINTER_PIXELS = 50000    # Limite máximo de pixels brancos (ruído). Acima disso = Miss Detection
PIVOT_Y_RATIO = 0.43788      # Razão do deslocamento do centro da ROI

# Angle (degrees, math convention: 0=right, CCW positive) observed when the
# pointer is at the 0-pressure and 100-pressure marks.
ANGLE_AT_0   = 240.0
ANGLE_AT_100 = -60.0

def order_corners(pts):
    """
    Order 4 corner points as: top-left, top-right, bottom-right, bottom-left.
    """
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]   # top-left  has smallest x+y
    rect[2] = pts[np.argmax(s)]   # bottom-right has largest x+y
    d = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(d)]   # top-right has smallest x-y
    rect[3] = pts[np.argmax(d)]   # bottom-left has largest x-y
    return rect, (rect[0]+rect[1]+rect[2]+rect[3])


def find_square(binary):
    """
    Find the largest approximate quadrilateral in a binary image.
    Returns the 4 ordered corners or None.
    """
    contours, _ = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    best = None
    best_area = 0
    for cnt in contours:
        area = cv.contourArea(cnt)
        if area < 1000:
            continue
        peri = cv.arcLength(cnt, True)
        approx = cv.approxPolyDP(cnt, EPSILON_RAMER_DOUGLAS_PEUCKER * peri, True)
        if len(approx) == 4 and area > best_area:
            best = approx
            best_area = area

    if best is None:
        return None

    return order_corners(best.reshape(4, 2))


def warp_to_square(gray, corners, size):
    """
    Perspective-warp the region defined by `corners` into a `size x size` square.
    """
    dst = np.array([
        [0,      0],
        [size-1, 0],
        [size-1, size-1],
        [0,      size-1],
    ], dtype="float32")

    M = cv.getPerspectiveTransform(corners.astype("float32"), dst)
    warped = cv.warpPerspective(gray, M, (size, size))
    return warped


def get_circle_params(binary):
    """
    Detect the main circle in a binary image using the isoperimetric inequality.
    Returns (radius, center_tuple) or (None, None).
    """
    contours, _ = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    best_radius = None
    best_center = None
    max_area = 0
    for cnt in contours:
        area = cv.contourArea(cnt)
        if area < 5000:  # Ignore small noise
            continue
        peri = cv.arcLength(cnt, True)
        if peri == 0:
            continue
        circularity = (4 * np.pi * area) / (peri * peri)
        
        if circularity > 0.75 and area > max_area:
            max_area = area
            best_radius = np.sqrt(area / np.pi)
            M = cv.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                best_center = (cx, cy)
    return best_radius, best_center


def adjust_template(template, target_radius, current_radius):
    """
    Resize the template so its circle matches the target radius.
    Pads with white if smaller, crops if larger, to maintain original size.
    """
    if target_radius is None or current_radius is None:
        return template
    
    scale = target_radius / current_radius
    h, w = template.shape
    new_h, new_w = int(h * scale), int(w * scale)
    
    # Avoid resizing to invalid dimensions
    if new_h <= 0 or new_w <= 0:
        return template

    resized = cv.resize(template, (new_w, new_h), 
                       interpolation=cv.INTER_AREA if scale < 1 else cv.INTER_LINEAR)
    
    result = np.ones((h, w), dtype=np.uint8) * 255 # White background
    
    if scale < 1:
        # Pad: place resized template in center of white result
        dy = (h - new_h) // 2
        dx = (w - new_w) // 2
        result[dy:dy+new_h, dx:dx+new_w] = resized
    else:
        # Crop: take center of resized template
        dy = (new_h - h) // 2
        dx = (new_w - w) // 2
        # Ensure we don't go out of bounds if resizing was slightly off
        y_end = min(dy + h, new_h)
        x_end = min(dx + w, new_w)
        actual_h = y_end - dy
        actual_w = x_end - dx
        result[0:actual_h, 0:actual_w] = resized[dy:y_end, dx:x_end]
        
    return result


def isolate_pointer(warped, template):
    """
    Subtract the template from the warped image.
    Returns a binary mask where the pointer pixels are white.
    """
    diff = cv.absdiff(warped, template)
    _, mask = cv.threshold(diff, DIFF_THRESHOLD, 255, cv.THRESH_BINARY)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
    return diff, mask


def pointer_angle_pca(mask):
    """
    Use PCA on the white pixels of `mask` to find the dominant direction.
    Disambiguates the 180° PCA ambiguity by picking the direction that
    points AWAY from the image center (pointer tip extends outward).
    Returns the angle in degrees (math convention: 0=right, CCW positive)
    and the centroid (cx, cy).
    """
    coords = cv.findNonZero(mask)
    if coords is None or len(coords) < 10:
        return None, None

    coords = coords.reshape(-1, 2).astype(np.float64)
    mean, eigenvectors = cv.PCACompute(coords, mean=None)
    cx, cy = mean[0]

    # First eigenvector = direction of maximum variance = pointer axis
    vx, vy = eigenvectors[0]

    # ── Disambiguate 180° ambiguity ──────────────────────────────────────
    # The image center is the hub of the manometer.
    # The pointer TIP is on the side of the centroid that is farther from
    # the image center.  Project each pixel onto the eigenvector; pixels
    # with positive projection are on the +v side, negative on the -v side.
    # The side whose mean is farther from image center is the tip side.
    img_center = np.array([mask.shape[1] / 2.0, mask.shape[0] * PIVOT_Y_RATIO])
    projections = (coords - mean) @ eigenvectors[0]       # scalar per pixel
    pos_mask = projections > 0
    neg_mask = ~pos_mask

    if pos_mask.any() and neg_mask.any():
        mean_pos = coords[pos_mask].mean(axis=0)
        mean_neg = coords[neg_mask].mean(axis=0)
        dist_pos = np.linalg.norm(mean_pos - img_center)
        dist_neg = np.linalg.norm(mean_neg - img_center)
        if dist_neg > dist_pos:
            vx, vy = -vx, -vy  # flip to point toward the tip

    angle_rad = np.arctan2(-vy, vx)  # negate vy because image y-axis is flipped
    angle_deg = np.degrees(angle_rad)

    return angle_deg, (cx, cy)


def pointer_angle_moments(mask):
    """
    Use image moments on the white pixels of `mask` to find the dominant direction.
    """
    M = cv.moments(mask)
    if M["m00"] == 0:
        return None, None
        
    cx = M["m10"] / M["m00"]
    cy = M["m01"] / M["m00"]
    
    mu20 = M["mu20"] / M["m00"]
    mu02 = M["mu02"] / M["m00"]
    mu11 = M["mu11"] / M["m00"]
    
    angle_rad = 0.5 * np.arctan2(2 * mu11, mu20 - mu02)
    vx, vy = np.cos(angle_rad), np.sin(angle_rad)
    
    coords = cv.findNonZero(mask)
    if coords is None:
        return None, None
    coords = coords.reshape(-1, 2).astype(np.float64)
    img_center = np.array([mask.shape[1] / 2.0, mask.shape[0] * PIVOT_Y_RATIO])
    mean = np.array([cx, cy])
    
    projections = (coords - mean) @ np.array([vx, vy])
    pos_mask = projections > 0
    neg_mask = ~pos_mask
    
    if pos_mask.any() and neg_mask.any():
        mean_pos = coords[pos_mask].mean(axis=0)
        mean_neg = coords[neg_mask].mean(axis=0)
        dist_pos = np.linalg.norm(mean_pos - img_center)
        dist_neg = np.linalg.norm(mean_neg - img_center)
        if dist_neg > dist_pos:
            vx, vy = -vx, -vy
            
    angle_rad = np.arctan2(-vy, vx)
    angle_deg = np.degrees(angle_rad)
    
    return angle_deg, (cx, cy)


def pointer_angle_hough(mask):
    """
    Use Probabilistic Hough Transform to find the dominant direction.
    """
    lines = cv.HoughLinesP(mask, 1, np.pi/180, threshold=20, minLineLength=30, maxLineGap=10)
    if lines is None or len(lines) == 0:
        return None, None
        
    longest = 0
    best_line = lines[0][0]
    for line in lines:
        x1, y1, x2, y2 = line[0]
        length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        if length > longest:
            longest = length
            best_line = line[0]
            
    x1, y1, x2, y2 = best_line
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    
    vx = x2 - x1
    vy = y2 - y1
    norm = np.sqrt(vx**2 + vy**2)
    if norm > 0:
        vx, vy = vx/norm, vy/norm
        
    img_center = np.array([mask.shape[1] / 2.0, mask.shape[0] * PIVOT_Y_RATIO])
    p1 = np.array([x1, y1])
    p2 = np.array([x2, y2])
    dist1 = np.linalg.norm(p1 - img_center)
    dist2 = np.linalg.norm(p2 - img_center)
    
    if dist1 > dist2:
        vx = x1 - x2
        vy = y1 - y2
    else:
        vx = x2 - x1
        vy = y2 - y1
        
    angle_rad = np.arctan2(-vy, vx)
    angle_deg = np.degrees(angle_rad)
    
    return angle_deg, (cx, cy)


def correct_angle_homography(angle_deg, warped_bin):
    """
    Uses the contour of the warped image to find the ellipse generated by homography.
    Corrects the angle according to the ellipse axes ratio to reverse perspective distortion.
    """
    contours, _ = cv.findContours(warped_bin, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if not contours:
        return angle_deg
        
    largest_contour = max(contours, key=cv.contourArea)
    if len(largest_contour) < 5:
        return angle_deg
        
    _, (minor_axis, major_axis), _ = cv.fitEllipse(largest_contour)
    
    if major_axis == 0:
        return angle_deg
        
    k = minor_axis / major_axis
    if k == 0 or np.isnan(k):
        return angle_deg
        
    rad = np.radians(angle_deg)
    
    # tan(theta_real) = k * tan(theta_lido)
    vy_real = k * np.sin(rad)
    vx_real = np.cos(rad)
    
    real_rad = np.arctan2(vy_real, vx_real)
    return np.degrees(real_rad)


def angle_to_pressure(angle_deg, angle_at_0=ANGLE_AT_0, angle_at_100=ANGLE_AT_100):
    """
    Map detected pointer angle to pressure value (0–100).
    Uses a linear interpolation between the two calibrated angles.
    Calibration: set angle_at_0 / angle_at_100 via ROS2 params.
    """
    sweep = angle_at_0 - angle_at_100   # total angular sweep
    if sweep == 0:
        return 0.0
    covered_distance = (angle_at_0 - angle_deg) % 360

    if covered_distance > sweep:
        dist_to_0 = 360 - covered_distance
        dist_to_100 = covered_distance - sweep
        if dist_to_0 < dist_to_100:
            covered_distance = 0.0
        else:
            covered_distance = sweep
    fraction = covered_distance / sweep
    pressure = fraction * 100.0
    return np.clip(pressure, 0.0, 100.0)

class ManometroDetector(Detector):
    def __init__(self):
        super().__init__('manometro_detector')  # handles camera sub, bridge, debug params

        self.pressure_pub = self.create_publisher(
            Float32,
            '/measured_pressure',
            10 #QoS
        )

        self.position_manometro = self.create_publisher(
            Int16MultiArray,
            '/position_manometer',
            10 # QoS
        )

        self.debug_pub = self.create_publisher(
            CompressedImage,
            '/manometro_debug/compressed',
            10
        )

        # Normalized pixel error [-1, 1] for XY alignment.
        # Published every frame: NaN when manometer not detected.
        self.error_pub = self.create_publisher(
            Point,
            '/manometer_error',
            10
        )

        # Load template
        assets_dir = os.path.join(os.path.dirname(__file__), 'assets')
        template_path = os.path.join(assets_dir, 'template2.png')
        template = cv.imread(template_path, cv.IMREAD_GRAYSCALE)
        if template is None:
            raise FileNotFoundError(f"Template not found: {template_path}")

        # Resize template
        template = cv.resize(template, (TEMPLATE_SIZE, TEMPLATE_SIZE))
        clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        template = clahe.apply(template)
        _, self.template_bin = cv.threshold(template, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)

        # Detectar um circulo no template
        self.template_radius, _ = get_circle_params(self.template_bin)
        if self.template_radius is None:
            self.get_logger().warn("Circle not found in template. Resize alignment might fail.")

        # Calibration: angular positions of 0% and 100% pressure on the dial.
        # Override via ROS2 params (or YAML) without recompiling.
        self.declare_parameter('angle_at_0',   ANGLE_AT_0)
        self.declare_parameter('angle_at_100', ANGLE_AT_100)
        self._angle_at_0   = self.get_parameter('angle_at_0').value
        self._angle_at_100 = self.get_parameter('angle_at_100').value

        # Additional rotation offset in degrees applied AFTER the best_rot*90 correction.
        # Use 0, 90, 180, or 270 to compensate for camera mounting orientation.
        # Default 180: the debug inset was 180° off from the physical manometer.
        self.declare_parameter('rotation_correction_deg', 0.0)
        self._rot_corr = self.get_parameter('rotation_correction_deg').value

        self.get_logger().info(
            f'Calibration: angle_at_0={self._angle_at_0:.1f}  '
            f'angle_at_100={self._angle_at_100:.1f}  '
            f'rotation_correction={self._rot_corr:.0f}deg')

        self._miss_log_time  = 0.0
        self._skip_log_time  = 0.0
        self._latest_debug_frame = None

        self._save_dir = os.path.expanduser('~/evtol/manometro_readings')
        os.makedirs(self._save_dir, exist_ok=True)
        self.create_subscription(
            String, '/pressure_analysis', self._save_callback, 10)

    def _save_callback(self, msg):
        if not msg.data or self._latest_debug_frame is None:
            return
        is_above = 'above' in msg.data
        label    = 'ACIMA_DO_LIMITE' if is_above else 'DENTRO_DO_LIMITE'
        ts       = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'manometro_{ts}_{label}.jpg'
        path     = os.path.join(self._save_dir, filename)
        cv.imwrite(path, self._latest_debug_frame)
        self.get_logger().info(f'[foto] Salva em {path}')

    def process_frame(self, frame, header):
        try:
            pressure = -1.0

            debug_frame = frame.copy()

            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            gray = clahe.apply(gray)
            _, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)

            centro = None
            angle = None
            pointer_mask_debug = None
            warped_debug = None
            best_rot_debug = 0
            h_img, w_img = frame.shape[:2]
            img_area = h_img * w_img

            result = find_square(binary)
            if result is not None:
                corners, corner_sum = result
                centro_quadrado = (corner_sum / 4)
                cornerCE = corners[0]
                cornerCD = corners[1]
                cornerBD = corners[2]
                cornerBE = corners[3]
                pontoMedioC = (cornerCD + cornerCE)/2
                pontoMedioB = (cornerBD + cornerBE)/2
                vetorBaixoCima = (pontoMedioC - pontoMedioB)
                vetorCorretivo = vetorBaixoCima*0.06212
                candidate_centro = (centro_quadrado + vetorCorretivo).astype(int)

                quad_area = cv.contourArea(corners.astype(np.float32).reshape(4, 1, 2))

                # Always draw candidate so the user can see what find_square found.
                # Orange = candidate being evaluated; color changes later based on outcome.
                cv.polylines(debug_frame, [corners.astype(int)], True, (0, 165, 255), 2)
                cv.circle(debug_frame, tuple(candidate_centro), 4, (0, 165, 255), -1)
                cv.putText(debug_frame, f"A={quad_area:.0f}",
                           (int(corners[0][0]), int(corners[0][1]) - 6),
                           cv.FONT_HERSHEY_SIMPLEX, 0.35, (0, 165, 255), 1)

                # Area filter: 0.5% min (removes tiny noise), 70% max (removes horizon line)
                if quad_area > 0.70 * img_area or quad_area < 0.005 * img_area:
                    now = time.time()
                    if now - self._skip_log_time >= 5.0:
                        self.get_logger().warn(
                            f"find_square: quadrilatero invalido ({quad_area:.0f} px2 vs "
                            f"img {img_area} px2), ignorado")
                        self._skip_log_time = now
                    cv.putText(debug_frame, "REJ",
                               (int(candidate_centro[0]) - 15, int(candidate_centro[1])),
                               cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 200), 1)
                    result = None

            if result is not None:
                corners, corner_sum = result
                centro_quadrado = (corner_sum / 4)
                cornerCE = corners[0]
                cornerCD = corners[1]
                cornerBD = corners[2]
                cornerBE = corners[3]
                pontoMedioC = (cornerCD + cornerCE)/2
                pontoMedioB = (cornerBD + cornerBE)/2
                vetorBaixoCima = (pontoMedioC - pontoMedioB)
                vetorCorretivo = vetorBaixoCima*0.06212
                candidate_centro = (centro_quadrado + vetorCorretivo).astype(int)

                warped = warp_to_square(gray, corners, TEMPLATE_SIZE)
                _, warped_bin = cv.threshold(warped, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)

                # Template subtraction (Radius Alignment + 4 rotations).
                # Select the rotation with MINIMUM diff: correct alignment cancels the
                # background, leaving only the pointer pixels in the diff.
                best_diff_val = float('inf')
                best_mask = None
                best_warped_bin = None
                best_rot = 0  # index of the selected rotation (for display/logging)

                warped_radius, _ = get_circle_params(warped_bin)
                adj_template_bin = adjust_template(self.template_bin, warped_radius, self.template_radius)

                for rot in range(4):
                    rotated_warped = np.rot90(warped_bin, rot)
                    diff, mask = isolate_pointer(rotated_warped, adj_template_bin)
                    diff_val = np.sum(diff)
                    if diff_val < best_diff_val:
                        best_diff_val = diff_val
                        best_mask = mask
                        best_warped_bin = rotated_warped
                        best_rot = rot

                pointer_mask = best_mask

                # Isolate only the area inside the manometer circle
                if best_warped_bin is not None:
                    r_mask, c_mask = get_circle_params(best_warped_bin)
                    if r_mask is not None and c_mask is not None:
                        c_mask_img = np.zeros_like(pointer_mask)
                        h_warped, w_warped = best_warped_bin.shape[:2]
                        pivot_center = (int(w_warped/2.0), int(h_warped * PIVOT_Y_RATIO))
                        cv.circle(c_mask_img, pivot_center, int(r_mask * ROI_RADIUS_RATIO), 255, -1)
                        pointer_mask = cv.bitwise_and(pointer_mask, c_mask_img)

                # CRITICAL: only confirm detection when template subtraction passes.
                # centro stays None for MISS cases → /manometer_error publishes NaN.
                if best_mask is not None:
                    white_pixels = cv.countNonZero(pointer_mask)
                    if white_pixels > MAX_POINTER_PIXELS:
                        now = time.time()
                        if now - self._miss_log_time >= 2.0:
                            self.get_logger().warn(
                                f"MISS DETECTION: {white_pixels} px — nao e manometro")
                            self._miss_log_time = now
                        cv.putText(debug_frame, "MISS",
                                   (int(candidate_centro[0]) - 20, int(candidate_centro[1])),
                                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 200), 1)
                    else:
                        # Valid manometer detection — confirm and draw in green
                        centro = candidate_centro
                        cv.polylines(debug_frame, [corners.astype(int)], True, (0, 255, 0), 3)
                        for pt in corners:
                            cv.circle(debug_frame, (int(pt[0]), int(pt[1])), 7, (0, 220, 0), -1)
                        cv.circle(debug_frame, tuple(centro), 5, (255, 0, 0), -1)
                        cv.putText(debug_frame, "MANOMETRO",
                                   (int(corners[0][0]), int(corners[0][1]) - 10),
                                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                        angle, _ = pointer_angle_pca(pointer_mask)
                        if angle is not None:
                            angle = correct_angle_homography(angle, best_warped_bin)
                            # With correct (minimum-diff) rotation selected, the PCA angle
                            # is already in the template's coordinate frame — no correction needed.
                            angle = ((angle + 180.0) % 360.0) - 180.0
                            pressure = angle_to_pressure(
                                angle, self._angle_at_0, self._angle_at_100)
                            pointer_mask_debug = pointer_mask.copy()
                            warped_debug = best_warped_bin.copy()
                            best_rot_debug = best_rot

                            self.get_logger().info(
                                f'[manometro] pressao={pressure:.1f}  '
                                f'angulo={angle:.1f}deg  rot={best_rot}')

                            side_len = float(np.linalg.norm(
                                corners[0].astype(float) - corners[1].astype(float)))
                            arrow_len = int(side_len * 0.35)
                            # The main frame is best_rot CCW rotations behind the template
                            # frame: angle_main = angle + best_rot * 90
                            angle_main = angle + best_rot * 90.0
                            ax = int(centro[0] + arrow_len * np.cos(np.radians(angle_main)))
                            ay = int(centro[1] - arrow_len * np.sin(np.radians(angle_main)))
                            cv.arrowedLine(debug_frame, tuple(centro), (ax, ay),
                                           (0, 0, 255), 3, tipLength=0.3)

            # ── Pressure + angle overlay (top-left banner) ──────────────────
            h_frame, w_frame = debug_frame.shape[:2]
            cv.rectangle(debug_frame, (0, 0), (340, 60), (0, 0, 0), -1)
            if pressure >= 0:
                cv.putText(debug_frame, f"Pressao: {pressure:.1f}",
                           (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 230, 0), 2)
                cv.putText(debug_frame, f"Angulo:  {angle:.1f} deg",
                           (10, 55), cv.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)
            else:
                cv.putText(debug_frame, "Pressao: --",
                           (10, 38), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 220), 2)

            # ── Mini-inset: binary threshold (bottom-left) ──────────────────
            inset_size = min(160, h_frame // 4, w_frame // 4)
            bin_small = cv.resize(binary, (inset_size, inset_size))
            bin_bgr = cv.cvtColor(bin_small, cv.COLOR_GRAY2BGR)
            cv.putText(bin_bgr, "bin", (2, 12), cv.FONT_HERSHEY_SIMPLEX, 0.4, (100, 255, 100), 1)
            debug_frame[h_frame - inset_size:h_frame, 0:inset_size] = bin_bgr

            # ── Mini-inset: manômetro normalizado + seta do ângulo detectado ──
            # The image is rotated by rotation_correction_deg so it shows the
            # same orientation as the physical manometer (right-side up).
            if warped_debug is not None and angle is not None:
                inset_size = min(160, h_frame // 3, w_frame // 3)
                # Apply the mounting correction to the display image
                extra_k = int(round(self._rot_corr / 90.0)) % 4
                warped_display = np.rot90(warped_debug, extra_k)
                warped_color = cv.cvtColor(
                    cv.resize(warped_display, (inset_size, inset_size)), cv.COLOR_GRAY2BGR)
                cx_i = inset_size // 2
                cy_i = int(inset_size * PIVOT_Y_RATIO)
                line_len = int(inset_size * 0.40)
                # The angle is already in the template (= inset) coordinate frame,
                # so the arrow is drawn directly at `angle`.
                lx = int(cx_i + line_len * np.cos(np.radians(angle)))
                ly = int(cy_i - line_len * np.sin(np.radians(angle)))
                cv.arrowedLine(warped_color, (cx_i, cy_i), (lx, ly), (0, 0, 255), 2, tipLength=0.3)
                cv.putText(warped_color, f"{angle:.0f}d r{best_rot_debug}", (2, 12),
                           cv.FONT_HERSHEY_SIMPLEX, 0.35, (0, 200, 255), 1)
                cv.rectangle(warped_color, (0, 0), (inset_size - 1, inset_size - 1), (0, 0, 200), 2)
                x0 = w_frame - inset_size
                y0 = h_frame - inset_size
                debug_frame[y0:h_frame, x0:w_frame] = warped_color

            # Publish pressure
            msg_out = Float32()
            msg_out.data = float(pressure)
            self.pressure_pub.publish(msg_out)

            # Publish manometer center position (only when detected)
            if centro is not None:
                position = Int16MultiArray()
                position.data = [int(centro[0]), int(centro[1])]
                self.position_manometro.publish(position)

            # Publish normalized XY error for alignment controller
            err_msg = Point()
            if centro is not None:
                err_msg.x = float(centro[0] - w_img / 2.0) / (w_img / 2.0)
                err_msg.y = float(centro[1] - h_img / 2.0) / (h_img / 2.0)
            else:
                err_msg.x = float('nan')
                err_msg.y = float('nan')
            self.error_pub.publish(err_msg)

            self._latest_debug_frame = debug_frame.copy()

            if bool(self.get_parameter('debug_image').value):
                self._pub_debug(self.debug_pub, debug_frame, header)

        except Exception as e:
            self.get_logger().error(f"Erro no callback de imagem: {str(e)}")

def main(args=None):
    rclpy.init(args=args)
    
    manometro_node = ManometroDetector()
    
    try:
        rclpy.spin(manometro_node)
    except KeyboardInterrupt:
        pass
    finally:
        manometro_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
