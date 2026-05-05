import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Float32
from cv_bridge import CvBridge

import cv2 as cv
import numpy as np
import os

# Configurações
TEMPLATE_SIZE = 400          # Side length (px) for the canonical square
DIFF_BLUR_KSIZE = 7          # (Median) Blur kernel for diff cleanup
DIFF_THRESHOLD = 30          # Threshold applied after blur on the diff image
EPSILON_RAMER_DOUGLAS_PEUCKER = 0.04 # Valor do epsilon para o algoritmo de RDP
ROI_RADIUS_RATIO = 0.30      # Define a ROI interna (para ignorar os números da borda)
ROI_OFFSET_Y = -20           # Deslocamento vertical do centro da ROI
ROI_OFFSET_X = 0             # Deslocamento horizontal do centro da ROI
MAX_POINTER_PIXELS = 2500    # Limite máximo de pixels brancos (ruído). Acima disso = Miss Detection

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
    return rect


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
    blurred = cv.medianBlur(diff, DIFF_BLUR_KSIZE)
    _, mask = cv.threshold(blurred, DIFF_THRESHOLD, 255, cv.THRESH_BINARY)
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
    img_center = np.array([mask.shape[1] / 2.0, mask.shape[0] / 2.0])
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
    img_center = np.array([mask.shape[1] / 2.0, mask.shape[0] / 2.0])
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
        
    img_center = np.array([mask.shape[1] / 2.0, mask.shape[0] / 2.0])
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


def angle_to_pressure(angle_deg):
    """
    Map detected pointer angle to pressure value (0–100).
    Uses a linear interpolation between the two calibrated angles.
    """
    # Normalize angles so the sweep is monotonically decreasing
    # from ANGLE_AT_0 (225°) to ANGLE_AT_100 (-45° = 315° CW)
    sweep = ANGLE_AT_0 - ANGLE_AT_100   # total angular sweep (positive)
    fraction = (ANGLE_AT_0 - angle_deg) / sweep
    pressure = fraction * 100.0
    return np.clip(pressure, 0.0, 100.0)

class ManometroDetector(Node):
    def __init__(self):
        super().__init__('manometro_detector')

        self.bridge = CvBridge()

        self.img_sub = self.create_subscription(
            CompressedImage,
            '/vertical_camera/compressed',
            self.callback,
            10 # QoS
        )

        self.pressure_pub = self.create_publisher(
            Float32,
            '/measured_pressure',
            10 #QoS
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

    def callback(self, msg):
        try:
            pressure = -1.0
            
            frame = self.bridge.compressed_imgmsg_to_cv2(msg)

            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            gray = clahe.apply(gray)
            _, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)

            corners = find_square(binary)
            if corners is not None:
                warped = warp_to_square(gray, corners, TEMPLATE_SIZE)
                _, warped_bin = cv.threshold(warped, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)

                # Template subtraction (Radius Alignment + 4 rotations)
                best_diff_val = -1
                best_mask = None
                best_warped_bin = None
                
                # Detect circle in warped image for radius matching
                warped_radius, _ = get_circle_params(warped_bin)
                # Adjust template to match warped image circle size
                adj_template_bin = adjust_template(self.template_bin, warped_radius, self.template_radius)
                
                # Test 0, 90, 180, 270 degrees
                for rot in range(4):
                    rotated_warped = np.rot90(warped_bin, rot)
                    diff, mask = isolate_pointer(rotated_warped, adj_template_bin)
                    
                    # The correct rotation and scale should have the greatest difference
                    diff_val = np.sum(diff)
                    if diff_val > best_diff_val:
                        best_diff_val = diff_val
                        best_mask = mask
                        best_warped_bin = rotated_warped

                pointer_mask = best_mask

                # Isolate only the area inside the manometer circle
                if best_warped_bin is not None:
                    r_mask, c_mask = get_circle_params(best_warped_bin)
                    if r_mask is not None and c_mask is not None:
                        c_mask_img = np.zeros_like(pointer_mask)
                        # Ajusta o centro da ROI usando os offsets configurados
                        adjusted_center = (c_mask[0] + ROI_OFFSET_X, c_mask[1] + ROI_OFFSET_Y)
                        # Aplica o raio da ROI (Region of Interest) para ignorar sujeiras nas bordas
                        cv.circle(c_mask_img, adjusted_center, int(r_mask * ROI_RADIUS_RATIO), 255, -1)
                        pointer_mask = cv.bitwise_and(pointer_mask, c_mask_img)

                # Angle detection
                angle = None
                if best_mask is not None:
                    # Conta a "massa" de pixels brancos (ruído / ponteiro)
                    white_pixels = cv.countNonZero(pointer_mask)
                    
                    if white_pixels > MAX_POINTER_PIXELS:
                        self.get_logger().warn(f"MISS DETECTION: Excesso de ruído absoluto - {white_pixels} px")
                    else:
                        angle, centroid = pointer_angle_pca(pointer_mask)
                        #angle, centroid = pointer_angle_moments(pointer_mask)
                        #angle, centroid = pointer_angle_hough(pointer_mask)
                        
                if angle is not None:
                        # Correcao do angulo
                        angle = correct_angle_homography(angle, best_warped_bin)
                        pressure = angle_to_pressure(angle)
                        
            # Publica a pressao como Float32 (valor real ou -1.0 se falhou)
            msg_out = Float32()
            msg_out.data = float(pressure)
            self.pressure_pub.publish(msg_out)

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