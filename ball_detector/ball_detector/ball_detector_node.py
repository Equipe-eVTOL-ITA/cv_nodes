import rclpy
from rclpy.node import Node
from rclpy.qos import (QoSProfile, QoSReliabilityPolicy,
                       QoSHistoryPolicy, QoSDurabilityPolicy)
from sensor_msgs.msg import CompressedImage
from custom_msgs.msg import BallDetection
from cv_bridge import CvBridge
import cv2
import numpy as np
import math


_DBG_QOS = QoSProfile(
    history=QoSHistoryPolicy.KEEP_LAST, depth=5,
    reliability=QoSReliabilityPolicy.BEST_EFFORT,
    durability=QoSDurabilityPolicy.VOLATILE
)


class BallDetectorNode(Node):
    """
    Detects the orange competition ball in the horizontal camera image.

    Shadow/lighting problem: Gazebo's directional light creates a half-lit ball.
    The unlit side fails the HSV filter → half-moon mask → broken circularity.

    Three complementary strategies handle this:

    Path A  — Canny edges + HoughCircles on grayscale (lighting-independent).
              Finds circular edges regardless of colour. Verifies orange content
              inside the detected circle as confirmation.

    Path B  — CLAHE-enhanced HSV mask + convex-hull filling.
              CLAHE equalises local contrast so the shadowed half gains V/S.
              Convex hull turns any remaining half-moon into a full polygon
              before the circularity / area check.

    Path C  — HoughLines anchor + ROI search (original tertiary path).

    Debug image topics (CompressedImage, BEST_EFFORT):
      ball_detector/mask/compressed     — orange mask after morphology
      ball_detector/hough/compressed    — Path A: Canny circles result
      ball_detector/contours/compressed — Path B: hull+contour result
      ball_detection_image/compressed   — final annotated frame
    """

    def __init__(self):
        super().__init__('ball_detector')

        # ── HSV filter ───────────────────────────────────────────────────
        self.declare_parameter('h_min',  0)
        self.declare_parameter('h_max', 20)
        self.declare_parameter('s_min', 120)
        self.declare_parameter('s_max', 255)
        self.declare_parameter('v_min', 80)    # lower than before to catch shadowed pixels
        self.declare_parameter('v_max', 255)
        self.declare_parameter('h_min2', 165)
        self.declare_parameter('h_max2', 180)
        self.declare_parameter('morph_kernel_size', 5)

        # ── CLAHE (Path B preprocessing) ────────────────────────────────
        # Applied in LAB L-channel before HSV conversion.
        self.declare_parameter('clahe_clip',      2.0)   # clip limit
        self.declare_parameter('clahe_grid',       8)    # tile grid size

        # ── Path A: Canny + HoughCircles ────────────────────────────────
        self.declare_parameter('canny_low',        50)
        self.declare_parameter('canny_high',      150)
        self.declare_parameter('hough_circles_dp',        1.5)
        self.declare_parameter('hough_circles_min_dist',   20)
        self.declare_parameter('hough_circles_param1',     50)
        self.declare_parameter('hough_circles_param2',     20)   # lower = more detections
        self.declare_parameter('ball_min_radius',           5)
        self.declare_parameter('ball_max_radius',         200)
        # Minimum fraction of orange pixels inside a detected circle (0-1)
        self.declare_parameter('hough_orange_min_frac',   0.10)

        # ── Path B: convex hull threshold ────────────────────────────────
        # Minimum ratio of hull_area / enclosing_circle_area to be a ball
        self.declare_parameter('hull_circle_ratio',       0.40)
        self.declare_parameter('min_ball_area',            50)

        # ── Path C: HoughLines anchor ────────────────────────────────────
        self.declare_parameter('roi_padding',       80)
        self.declare_parameter('hough_threshold',   50)
        self.declare_parameter('hough_min_length',  50)
        self.declare_parameter('hough_max_gap',     20)

        # ── Diagnostic ───────────────────────────────────────────────────
        self.declare_parameter('hsv_log_interval', 30)

        # ── Publishers ───────────────────────────────────────────────────
        self.publisher_    = self.create_publisher(BallDetection,   'ball_detection',                    10)
        self.debug_pub_    = self.create_publisher(CompressedImage, 'ball_detection_image/compressed',   _DBG_QOS)
        self.mask_pub_     = self.create_publisher(CompressedImage, 'ball_detector/mask/compressed',     _DBG_QOS)
        self.hough_pub_    = self.create_publisher(CompressedImage, 'ball_detector/hough/compressed',    _DBG_QOS)
        self.contour_pub_  = self.create_publisher(CompressedImage, 'ball_detector/contours/compressed', _DBG_QOS)

        self.subscription = self.create_subscription(
            CompressedImage, '/horizontal_camera/compressed',
            self.image_callback, 10)

        self.br = CvBridge()
        self._frame_count = 0
        self.get_logger().info('Ball Detector Node started.')

    # ------------------------------------------------------------------ #
    #  UTILITIES                                                           #
    # ------------------------------------------------------------------ #

    def _pub_debug(self, pub, img, header):
        try:
            out = self.br.cv2_to_compressed_imgmsg(img)
            out.header = header
            pub.publish(out)
        except Exception as e:
            self.get_logger().error(f'Debug img error: {e}')

    def _apply_clahe(self, frame_bgr):
        """Equalise local contrast in LAB L-channel before HSV conversion."""
        clip  = self.get_parameter('clahe_clip').value
        grid  = int(self.get_parameter('clahe_grid').value)
        lab   = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid))
        l_eq  = clahe.apply(l)
        return cv2.cvtColor(cv2.merge([l_eq, a, b]), cv2.COLOR_LAB2BGR)

    def _build_orange_mask(self, frame_bgr):
        """Return morphologically-closed dual-range orange/red HSV mask."""
        h_min  = self.get_parameter('h_min').value
        h_max  = self.get_parameter('h_max').value
        h_min2 = self.get_parameter('h_min2').value
        h_max2 = self.get_parameter('h_max2').value
        s_min  = self.get_parameter('s_min').value
        s_max  = self.get_parameter('s_max').value
        v_min  = self.get_parameter('v_min').value
        v_max  = self.get_parameter('v_max').value
        k_size = self.get_parameter('morph_kernel_size').value

        hsv   = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv, np.array([h_min,  s_min, v_min]),
                                 np.array([h_max,  s_max, v_max]))
        mask2 = cv2.inRange(hsv, np.array([h_min2, s_min, v_min]),
                                 np.array([h_max2, s_max, v_max]))
        mask  = cv2.bitwise_or(mask1, mask2)
        k     = cv2.getStructuringElement(cv2.MORPH_RECT, (k_size, k_size))
        return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k), hsv

    # ------------------------------------------------------------------ #
    #  PATH A — Canny + HoughCircles (lighting-independent)               #
    # ------------------------------------------------------------------ #

    def _path_a_canny_hough(self, frame_bgr, orange_mask, header):
        """
        Detect circular edge in grayscale regardless of colour/lighting.
        Verify orange content inside each candidate circle to confirm it's the ball.
        Returns (cx, cy, radius, area, 1.0) or None.
        """
        dp       = float(self.get_parameter('hough_circles_dp').value)
        min_dist = int(self.get_parameter('hough_circles_min_dist').value)
        p1       = int(self.get_parameter('hough_circles_param1').value)
        p2       = int(self.get_parameter('hough_circles_param2').value)
        min_r    = int(self.get_parameter('ball_min_radius').value)
        max_r    = int(self.get_parameter('ball_max_radius').value)
        min_frac = float(self.get_parameter('hough_orange_min_frac').value)
        c_low    = int(self.get_parameter('canny_low').value)
        c_high   = int(self.get_parameter('canny_high').value)

        gray    = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        edges   = cv2.Canny(blurred, c_low, c_high)

        hough_dbg = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        result    = None

        circles = cv2.HoughCircles(
            edges, cv2.HOUGH_GRADIENT,
            dp=dp, minDist=min_dist,
            param1=p1, param2=p2,
            minRadius=min_r, maxRadius=max_r
        )

        if circles is not None:
            h, w = frame_bgr.shape[:2]
            candidates = []
            for c in np.round(circles[0, :]).astype(int):
                cx, cy, r = int(c[0]), int(c[1]), int(c[2])
                # Build circular ROI mask and check orange fraction
                roi_mask = np.zeros((h, w), dtype=np.uint8)
                cv2.circle(roi_mask, (cx, cy), r, 255, -1)
                total_px   = math.pi * r * r
                orange_px  = int(np.count_nonzero(cv2.bitwise_and(orange_mask, orange_mask, mask=roi_mask)))
                orange_frac = orange_px / max(total_px, 1)
                if orange_frac >= min_frac:
                    candidates.append((orange_frac, cx, cy, r))
                cv2.circle(hough_dbg, (cx, cy), r,
                           (0, 200, 0) if orange_frac >= min_frac else (60, 60, 60), 1)

            if candidates:
                # Best = highest orange fraction
                candidates.sort(key=lambda x: -x[0])
                _, cx, cy, r = candidates[0]
                area = math.pi * r * r
                cv2.circle(hough_dbg, (cx, cy), r, (0, 255, 0), 2)
                cv2.circle(hough_dbg, (cx, cy), 3, (0, 0, 255), -1)
                cv2.putText(hough_dbg, f'A: r={r} orng={candidates[0][0]:.0%}',
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                result = (cx, cy, r, area, 1.0)
        else:
            cv2.putText(hough_dbg, 'A: no circles', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        self._pub_debug(self.hough_pub_, hough_dbg, header)
        return result

    # ------------------------------------------------------------------ #
    #  PATH B — CLAHE mask + convex hull (half-moon → full circle)        #
    # ------------------------------------------------------------------ #

    def _path_b_hull_contour(self, mask_closed, frame_bgr, header):
        """
        Get contours from the (possibly half-moon) mask, compute the convex hull
        of each contour, and test hull_area / enclosing_circle_area.
        A half-moon's hull approximates a full circle.
        Returns (cx, cy, radius, area, quality) or None.
        """
        min_area  = int(self.get_parameter('min_ball_area').value)
        min_ratio = float(self.get_parameter('hull_circle_ratio').value)

        contours, _ = cv2.findContours(mask_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        dbg = frame_bgr.copy()
        result  = None
        best_q  = 0.0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue

            # Convex hull of the contour (fills the missing shadow side)
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            if hull_area <= 0:
                continue

            # Enclosing circle of the hull
            (cx, cy), radius = cv2.minEnclosingCircle(hull)
            circle_area = math.pi * radius * radius
            ratio = hull_area / circle_area  # 1.0 = perfect circle

            # Draw all hulls for inspection
            cv2.drawContours(dbg, [hull], -1, (0, 200, 255), 1)

            if ratio >= min_ratio and ratio > best_q:
                best_q  = ratio
                result  = (int(cx), int(cy), max(5, int(radius)), hull_area, ratio)
                cv2.circle(dbg, (int(cx), int(cy)), int(radius), (0, 255, 0), 2)
                cv2.putText(dbg, f'B: hull_ratio={ratio:.2f} area={hull_area:.0f}',
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        if result is None:
            cv2.putText(dbg, 'B: no valid hull', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        self._pub_debug(self.contour_pub_, dbg, header)
        return result

    # ------------------------------------------------------------------ #
    #  PATH C — HoughLines anchor (hose-based ROI, original tertiary)     #
    # ------------------------------------------------------------------ #

    def _path_c_hough_lines_anchor(self, mask_closed, width, height):
        pad    = int(self.get_parameter('roi_padding').value)
        h_thr  = int(self.get_parameter('hough_threshold').value)
        h_mlen = int(self.get_parameter('hough_min_length').value)
        h_mgap = int(self.get_parameter('hough_max_gap').value)
        min_area = int(self.get_parameter('min_ball_area').value)

        lines = cv2.HoughLinesP(mask_closed, 1, np.pi / 180,
                                threshold=h_thr, minLineLength=h_mlen,
                                maxLineGap=h_mgap)
        h_lines = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = abs(math.degrees(math.atan2(y2 - y1, x2 - x1)))
                if angle < 15 or angle > 165:
                    h_lines.append(line[0])

        if not h_lines:
            return None

        bx1 = max(0,      min(min(l[0], l[2]) for l in h_lines) - pad)
        bx2 = min(width,  max(max(l[0], l[2]) for l in h_lines) + pad)
        by1 = max(0,      min(min(l[1], l[3]) for l in h_lines) - pad)
        by2 = min(height, max(max(l[1], l[3]) for l in h_lines) + pad)

        roi_mask = mask_closed[by1:by2, bx1:bx2]
        contours, _ = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best = None
        best_q = 0.0
        for cnt in contours:
            if cv2.contourArea(cnt) < min_area:
                continue
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            (cx, cy), r = cv2.minEnclosingCircle(hull)
            ratio = hull_area / max(math.pi * r * r, 1)
            if ratio > best_q:
                best_q = ratio
                best = (int(cx) + bx1, int(cy) + by1, max(5, int(r)), hull_area, ratio)
        return best

    # ------------------------------------------------------------------ #
    #  MAIN CALLBACK                                                       #
    # ------------------------------------------------------------------ #

    def image_callback(self, msg):
        try:
            frame = self.br.compressed_imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'Failed to convert image: {e}')
            return

        self._frame_count += 1
        height, width = frame.shape[:2]

        # ── 1. Preprocessing: CLAHE → equalise local contrast ────────────
        frame_eq = self._apply_clahe(frame)

        # ── 2. Build orange mask on CLAHE-enhanced frame ──────────────────
        mask_closed, hsv_eq = self._build_orange_mask(frame_eq)

        # Also build mask on original frame for Path A colour verification
        mask_orig, _ = self._build_orange_mask(frame)

        # Publish mask debug
        mask_dbg = cv2.cvtColor(mask_closed, cv2.COLOR_GRAY2BGR)
        cv2.putText(mask_dbg, f'orange px={np.count_nonzero(mask_closed)}',
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        self._pub_debug(self.mask_pub_, mask_dbg, msg.header)

        # ── 3. HSV diagnostic ─────────────────────────────────────────────
        log_int = self.get_parameter('hsv_log_interval').value
        if log_int > 0 and self._frame_count % log_int == 0:
            near = cv2.bitwise_or(
                cv2.inRange(hsv_eq, np.array([0, 40, 40]), np.array([40, 255, 255])),
                cv2.inRange(hsv_eq, np.array([155, 40, 40]), np.array([180, 255, 255]))
            )
            n = int(np.count_nonzero(near))
            if n > 0:
                px = hsv_eq[near > 0]
                h, s, v = px[:, 0], px[:, 1], px[:, 2]
                hist_h, _ = np.histogram(h, bins=8, range=(0, 40))
                dom = int(np.argmax(hist_h)) * 5
                self.get_logger().info(
                    f'[HSV] near-orange px={n} | H=[{int(h.min())},{int(h.max())}] '
                    f'dom≈[{dom},{dom+5}] | S mean={int(s.mean())} | V mean={int(v.mean())}'
                )

        # ── 4. Detection pipeline ─────────────────────────────────────────
        det_result = None
        det_method = 'none'

        # Path A: Canny+HoughCircles — lighting-independent edge circles
        det_result = self._path_a_canny_hough(frame, mask_orig, msg.header)
        if det_result:
            det_method = 'Canny+Hough'
            self._pub_debug(self.contour_pub_, frame, msg.header)  # placeholder

        # Path B: CLAHE mask + convex hull — half-moon → circle
        if det_result is None:
            det_result = self._path_b_hull_contour(mask_closed, frame_eq, msg.header)
            if det_result:
                det_method = 'hull-contour'

        # Path C: HoughLines anchor (hose ROI)
        if det_result is None:
            self._pub_debug(self.contour_pub_, frame, msg.header)  # placeholder
            det_result = self._path_c_hough_lines_anchor(mask_closed, width, height)
            if det_result:
                det_method = 'lines-anchor'

        # ── 5. Build output ───────────────────────────────────────────────
        det_msg = BallDetection()
        det_msg.header.stamp    = self.get_clock().now().to_msg()
        det_msg.header.frame_id = msg.header.frame_id
        output = frame.copy()

        if det_result is not None:
            cx, cy, radius, area, _ = det_result
            cv2.circle(output, (cx, cy), radius, (0, 0, 255), -1)
            cv2.drawMarker(output, (cx, cy), (255, 255, 255),
                           cv2.MARKER_CROSS, radius * 2, 2)
            x_err = (cx - width  / 2.0) / (width  / 2.0)
            y_err = (cy - height / 2.0) / (height / 2.0)
            det_msg.is_detected       = True
            det_msg.x_error           = float(x_err)
            det_msg.y_error           = float(y_err)
            det_msg.target_score      = float(area)
            det_msg.tracking_mode     = 0
            det_msg.center_position.x = float(cx)
            det_msg.center_position.y = float(cy)
            det_msg.center_position.z = 0.0
            det_msg.distance_estimate = -1.0
            cv2.putText(output, f'[{det_method}] area={area:.0f}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            det_msg.is_detected       = False
            det_msg.x_error           = 0.0
            det_msg.y_error           = 0.0
            det_msg.target_score      = 0.0
            det_msg.tracking_mode     = 0
            det_msg.center_position.x = 0.0
            det_msg.center_position.y = 0.0
            det_msg.center_position.z = 0.0
            det_msg.distance_estimate = -1.0
            cv2.putText(output, 'nao detectada', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        self.publisher_.publish(det_msg)
        self._pub_debug(self.debug_pub_, output, msg.header)


def main(args=None):
    rclpy.init(args=args)
    node = BallDetectorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
