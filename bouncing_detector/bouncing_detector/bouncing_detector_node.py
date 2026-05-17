import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from custom_msgs.msg import BouncingDetection
from cv_bridge import CvBridge
import cv2
import cv2.aruco as aruco
import numpy as np
from rclpy.qos import (
    QoSProfile,
    QoSReliabilityPolicy,
    QoSHistoryPolicy,
    QoSDurabilityPolicy
)

try:
    import easyocr as _easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False

try:
    import pytesseract
    PYTESSERACT_AVAILABLE = True
except ImportError:
    PYTESSERACT_AVAILABLE = False


class BouncingDetectorNode(Node):
    def __init__(self):
        super().__init__('bouncing_detector_node')

        # Initialise OCR reader
        self._ocr = None
        if EASYOCR_AVAILABLE:
            self.get_logger().info('EasyOCR available — using it for digit reading.')
            self._ocr = _easyocr.Reader(['en'], gpu=False, verbose=False)
        elif PYTESSERACT_AVAILABLE:
            self.get_logger().info('pytesseract available — using it as fallback digit reader.')
        else:
            self.get_logger().warn(
                'Neither EasyOCR nor pytesseract found. '
                'Falling back to contour-based digit classification (less reliable). '
                'Install EasyOCR with: pip install easyocr')

        self.declare_parameter('vertical_camera_topic', '/vertical_camera/compressed')
        camera_topic = self.get_parameter('vertical_camera_topic').get_parameter_value().string_value

        self.declare_parameter('debug_pub_interval', 0.2)
        self._debug_pub_interval = self.get_parameter('debug_pub_interval').get_parameter_value().double_value
        # self._debug_pub_interval = 0.2

        qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE
        )

        self.publisher_  = self.create_publisher(BouncingDetection, 'bouncing_detection', 10)

        self.debug_pub_  = self.create_publisher(CompressedImage, 'bouncing_detection_image/compressed', qos)
        self._debug_last_pub_time = self.get_clock().now()
        self._debug_pub_interval = rclpy.duration.Duration(seconds=self._debug_pub_interval)
        self._debug_max_width = 640
        self._jpeg_quality = 60 # porcentagem de qualidade


        self.subscription = self.create_subscription(
            CompressedImage,
            camera_topic,
            self.image_callback,
            qos
        )

        self.br = CvBridge()
        self.aruco_dict   = aruco.getPredefinedDictionary(aruco.DICT_5X5_100)
        self.aruco_params = aruco.DetectorParameters_create()

        # Latched target: persists across frames so base matching works even
        # when the ArUco is no longer in view (e.g. during SEARCH_BASE).
        self._cached_target_calculated = False
        self._cached_target_base = ''

    
    # Critério de divisibilidade
    def calculate_target_number(self, aruco_id):
        """
        Returna '3', '4', or '5'
        Verifica quais entre eles, em ordem decrescente, é divisor exato do id do ArUco.
        """
        for n in [5, 4, 3]:
            if aruco_id % n == 0:
                return str(n)
        self.get_logger().warn(f'ArUco ID {aruco_id} has no divisor in {{3, 4, 5}}')
        return None

    # Detecção das formas
    def detect_shape(self, contour):
        """
        Returns 'TRIANGULO', 'HEXAGONO', or 'ESTRELA' — the three shapes used
        in the SAE 2026 competition.  All other contours return 'UNKNOWN'.

        Detection strategy:
          - ESTRELA : low solidity (many concavities) + many polygon vertices.
              Stars have a convex-hull area much larger than their own area.
          - TRIANGULO : 3 vertices after polygon approximation.
          - HEXAGONO   : 5–8 vertices (approximation tolerance can collapse some
              hexagon edges, so a range is safer than an exact count).
        """
        peri   = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
        v      = len(approx)

        area      = cv2.contourArea(contour)
        hull      = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity  = area / hull_area if hull_area > 0 else 1.0

        # Stars are non-convex: solidity well below 0.65, vertex count ≥ 8
        if solidity < 0.65 and v >= 8:
            return "ESTRELA"

        if v == 3:
            return "TRIANGULO"

        if 5 <= v <= 8:
            return "HEXAGONO"

        return "UNKNOWN"

    # leitura dos números
    def _classify_digit_345(self, thresh):
        """
        Classifies a binarised digit image (digit=white on black) as ('3'|'4'|'5', conf).

        Returns (digit_str, confidence) where confidence reflects feature quality:
          '4' — detected via enclosed hole (RETR_CCOMP); conf scales with hole completeness
          '5' — full-width top bar; conf scales with bar coverage ratio
          '3' — residual case; fixed conf 0.40 (weakest claim)
        Returns (None, 0.0) on empty image.
        """
        h, w = thresh.shape[:2]
        if h == 0 or w == 0:
            return None, 0.0

        cnts, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_CCOMP,
                                           cv2.CHAIN_APPROX_SIMPLE)
        if hierarchy is None or len(cnts) == 0:
            return None, 0.0

        holes = sum(1 for row in hierarchy[0] if row[3] != -1)
        if holes >= 1:
            # Confidence: saturate at 1 hole (expected for '4')
            conf = min(1.0, 0.55 + holes * 0.2)
            return '4', conf

        top = thresh[:max(1, h // 4), :]
        filled_cols = int(np.sum(np.any(top > 0, axis=0)))
        bar_ratio = filled_cols / max(w, 1)
        if bar_ratio > 0.5:
            conf = min(1.0, 0.45 + bar_ratio * 0.5)
            return '5', conf

        return '3', 0.40

    def read_number_in_contour(self, frame, contour):
        """
        Reads the digit ('3','4','5') inside a detected shape.

        Returns (digit_str, confidence) where confidence ∈ [0, 1]:
          EasyOCR high (conf > 0.7)  → 0.90
          EasyOCR mid  (conf 0.4-0.7) → 0.65
          pytesseract                 → 0.55
          contour classification      → 0.20 – 0.70 (feature-based)
          not readable                → (None, 0.0)
        """
        x, y, w, h = cv2.boundingRect(contour)
        margin = max(5, min(w, h) // 6)
        x1 = max(0, x + margin)
        y1 = max(0, y + margin)
        x2 = min(frame.shape[1], x + w - margin)
        y2 = min(frame.shape[0], y + h - margin)
        if x2 <= x1 or y2 <= y1:
            return None, 0.0

        roi  = frame[y1:y2, x1:x2]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi

        scale = max(1, 80 // min(gray.shape[:2]))
        if scale > 1:
            gray = cv2.resize(gray, None, fx=scale, fy=scale,
                              interpolation=cv2.INTER_CUBIC)

        _, thresh = cv2.threshold(gray, 0, 255,
                                  cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        contour_local = ((contour.reshape(-1, 2) - np.array([x1, y1])) * scale
                         ).reshape(-1, 1, 2).astype(np.int32)
        shape_mask = np.zeros(thresh.shape[:2], dtype=np.uint8)
        cv2.drawContours(shape_mask, [contour_local], -1, 255, cv2.FILLED)
        thresh = cv2.bitwise_and(thresh, thresh, mask=shape_mask)

        # --- 1. EasyOCR (primary) ----------------------------------------
        if self._ocr is not None:
            results = self._ocr.readtext(thresh, allowlist='345',
                                         detail=1, paragraph=False)
            for (_, text, ocr_conf) in results:
                if ocr_conf > 0.4:
                    for c in text.strip():
                        if c in '345':
                            conf = 0.90 if ocr_conf > 0.7 else 0.65
                            return c, conf

        # --- 2. pytesseract (secondary) ----------------------------------
        if PYTESSERACT_AVAILABLE:
            text = pytesseract.image_to_string(
                thresh,
                config='--psm 10 --oem 3 -c tessedit_char_whitelist=345'
            )
            for c in text.strip():
                if c in '345':
                    return c, 0.55

        # --- 3. Contour-based classification (fallback) ------------------
        digit, conf = self._classify_digit_345(thresh)
        return digit, conf


    # Main callback
    def image_callback(self, msg):
        try:
            frame = self.br.compressed_imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'Failed to convert image: {e}')
            return

        height, width = frame.shape[:2]
        output_frame  = frame.copy()

        det_msg = BouncingDetection()
        det_msg.header = msg.header

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ---- 1. ArUco detection ----------------------------------------
        corners, ids, _ = aruco.detectMarkers(gray, self.aruco_dict,
                                              parameters=self.aruco_params)

        if ids is not None and len(ids) > 0:
            aruco.drawDetectedMarkers(output_frame, corners, ids)

            c  = corners[0][0]
            cx = (c[0][0] + c[2][0]) / 2.0
            cy = (c[0][1] + c[2][1]) / 2.0

            det_msg.aruco_detected = True
            det_msg.aruco_id       = int(ids[0][0])
            det_msg.aruco_x_error  = float((cx - width  / 2.0) / (width  / 2.0))
            det_msg.aruco_y_error  = float((cy - height / 2.0) / (height / 2.0))

            # ---- 2. Detect the shape surrounding the ArUco (gabarito) --
            blurred  = cv2.GaussianBlur(gray, (5, 5), 0)
            edges    = cv2.Canny(blurred, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)

            aruco_shape = "UNKNOWN"
            for cnt in sorted(contours, key=cv2.contourArea, reverse=True):
                if cv2.contourArea(cnt) < 500:
                    continue
                shape = self.detect_shape(cnt)
                if shape == "UNKNOWN":
                    continue
                M = cv2.moments(cnt)
                if M['m00'] == 0:
                    continue
                scx = int(M['m10'] / M['m00'])
                scy = int(M['m01'] / M['m00'])
                if abs(scx - cx) < 150 and abs(scy - cy) < 150:
                    aruco_shape = shape
                    cv2.putText(output_frame, f"GABARITO: {shape}",
                                (scx, scy - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 255, 0), 2)
                    break

            det_msg.aruco_shape = aruco_shape

            # ---- 3. Apply divisibility rule → target base ID -----------
            #
            # The correct landing base has:
            #   (a) the same shape as the gabarito  (= aruco_shape)
            #   (b) the number written inside it divides aruco_id
            #
            # target_base encodes both as "SHAPE_NUMBER" (e.g. "HEXAGONO_4")
            # so base detection can match unambiguously.
            if aruco_shape != "UNKNOWN":
                target_number = self.calculate_target_number(det_msg.aruco_id)
                if target_number is not None:
                    det_msg.target_calculated = True
                    det_msg.target_base = f"{aruco_shape}_{target_number}"
                    cv2.putText(output_frame, f"TARGET: {det_msg.target_base}",
                                (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                                1.0, (255, 0, 0), 2)

        # Latch the target so base matching keeps working when ArUco is not visible
        if det_msg.target_calculated:
            self._cached_target_calculated = True
            self._cached_target_base = det_msg.target_base
        elif self._cached_target_calculated:
            det_msg.target_calculated = True
            det_msg.target_base = self._cached_target_base

        # ---- 4. Landing base detection ----------------------------------
        blurred2    = cv2.GaussianBlur(gray, (5, 5), 0)
        edges2      = cv2.Canny(blurred2, 50, 150)
        base_cnts, _ = cv2.findContours(edges2, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)

        visible_bases      = []
        visible_bases_x    = []
        visible_bases_y    = []
        visible_bases_conf = []

        for cnt in base_cnts:
            if cv2.contourArea(cnt) < 500:
                continue
            shape = self.detect_shape(cnt)
            if shape == "UNKNOWN":
                continue

            M = cv2.moments(cnt)
            if M['m00'] == 0:
                continue
            bcx = int(M['m10'] / M['m00'])
            bcy = int(M['m01'] / M['m00'])
            bx_err = float((bcx - width  / 2.0) / (width  / 2.0))
            by_err = float((bcy - height / 2.0) / (height / 2.0))

            # Read the number + detection confidence
            number, num_conf = self.read_number_in_contour(frame, cnt)
            base_id = f"{shape}_{number}" if number else shape

            visible_bases.append(base_id)
            visible_bases_x.append(bx_err)
            visible_bases_y.append(by_err)
            visible_bases_conf.append(float(num_conf) if num_conf else 0.0)

            cv2.circle(output_frame, (bcx, bcy), 10, (255, 255, 0), -1)
            cv2.putText(output_frame,
                        f"BASE {base_id} ({num_conf:.2f})" if num_conf else f"BASE {base_id}",
                        (bcx - 30, bcy - 15), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 0), 2)

            # Match against the target (exact match, then soft match by shape)
            if det_msg.target_calculated and base_id == det_msg.target_base:
                det_msg.target_base_in_sight = True
                det_msg.target_base_x_error  = bx_err
                det_msg.target_base_y_error  = by_err
                cv2.circle(output_frame, (bcx, bcy), 20, (0, 0, 255), 3)
            elif (det_msg.target_calculated
                  and not det_msg.target_base_in_sight
                  and number is None
                  and shape == det_msg.target_base.split('_')[0]):
                same_shape_readable = [b for b in visible_bases
                                       if b.startswith(shape + '_')]
                if not same_shape_readable:
                    self.get_logger().warn(
                        f'Soft match: {shape} seen but number unreadable '
                        f'(target={det_msg.target_base}) — using position anyway')
                    det_msg.target_base_in_sight = True
                    det_msg.target_base_x_error  = bx_err
                    det_msg.target_base_y_error  = by_err
                    cv2.circle(output_frame, (bcx, bcy), 20, (0, 165, 255), 3)

        det_msg.visible_bases            = visible_bases
        det_msg.visible_bases_x_error    = visible_bases_x
        det_msg.visible_bases_y_error    = visible_bases_y
        det_msg.visible_bases_confidence = visible_bases_conf

        self.publisher_.publish(det_msg)

        try:
            debug_msg = self.br.cv2_to_compressed_imgmsg(output_frame)
            debug_msg.header = msg.header
            self.debug_pub_.publish(debug_msg)
            now_s = self.get_clock().now()
            if (now_s - self._debug_last_pub_time) >= self._debug_pub_interval:
                h, w = output_frame.shape[:2]
                if w > self._debug_max_width:
                   scale = float(self._debug_max_width) / float(w)
                   output_small = cv2.resize(output_frame, (int(w*scale), int(h*scale)),interpolation=cv2.INTER_AREA)
                else:
                   output_small = output_frame
                ret, enc = cv2.imencode('.jpg', output_small,
                                        [int(cv2.IMWRITE_JPEG_QUALITY), int(self._jpeg_quality)])
                if not ret:
                    self.get_logger().error('JPEG encoding failed for debug image')
                else:
                    debug_msg = CompressedImage()
                    debug_msg.format = 'jpeg'
                    debug_msg.data = np.array(enc).tobytes()
                    debug_msg.header = msg.header
                    self.debug_pub_.publish(debug_msg)
                    self._debug_last_pub_time = now_s

        except Exception as e:
            self.get_logger().error(f'Failed to publish debug image: {e}')


def main(args=None):
    rclpy.init(args=args)
    node = BouncingDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
