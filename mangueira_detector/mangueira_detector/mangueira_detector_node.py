#!/usr/bin/env python3
"""
Mangueira (hose) line detector using color mask + lanes-inspired detection pipeline.

Detects hose lines using:
1. Red/orange HSV color mask
2. ROI (Region of Interest) cropping (optional)
3. Canny edge detection
4. HoughLinesP line detection
5. Slope-based line clustering and weighted averaging
6. Temporal smoothing (deque-based)

Publishes normalized hose position/angle for drone alignment control.
"""

from collections import deque
import math

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge, CvBridgeError
from rclpy.node import Node
from rclpy.qos import QoSProfile
from sensor_msgs.msg import CompressedImage, Image
from geometry_msgs.msg import PointStamped
from std_msgs.msg import Float64
from vision_msgs.msg import Detection2D, Detection2DArray, BoundingBox2D, ObjectHypothesisWithPose

# Import line utilities
from .line_utils import (
    lines_to_slope_intercept,
    cluster_lines_by_angle,
    average_cluster,
    make_line_coordinates,
    apply_roi_mask,
    circular_mean,
    smooth_value_deque,
)

_DBG_QOS = QoSProfile(depth=5)


class MangueiraDetector(Node):
    def __init__(self):
        super().__init__('mangueira_detector')

        # ============ Image processing parameters ============
        self.declare_parameter('image_topic', '/vertical_camera/compressed')
        self.declare_parameter('resize_width', 600)

        # ============ Edge detection & Hough parameters ============
        self.declare_parameter('canny_low', 50)
        self.declare_parameter('canny_high', 150)
        self.declare_parameter('hough_threshold', 30)
        self.declare_parameter('min_line_length', 30)
        self.declare_parameter('max_line_gap', 10)

        # ============ Red/orange mask parameters ============
        # Handle hue wrap-around with two ranges
        self.declare_parameter('red_lower1_h', 0)
        self.declare_parameter('red_lower1_s', 100)
        self.declare_parameter('red_lower1_v', 40)
        self.declare_parameter('red_upper1_h', 10)
        self.declare_parameter('red_upper1_s', 255)
        self.declare_parameter('red_upper1_v', 255)

        self.declare_parameter('red_lower2_h', 170)
        self.declare_parameter('red_lower2_s', 100)
        self.declare_parameter('red_lower2_v', 40)
        self.declare_parameter('red_upper2_h', 179)
        self.declare_parameter('red_upper2_s', 255)
        self.declare_parameter('red_upper2_v', 255)

        self.declare_parameter('morph_kernel_size', 3)

        # ============ ROI parameters ============
        self.declare_parameter('roi_enable', False)
        self.declare_parameter('roi_type', 'trapezoid')  # 'trapezoid' or 'rectangle'
        self.declare_parameter('roi_top_fraction', 0.2)
        self.declare_parameter('roi_bottom_fraction', 1.0)
        self.declare_parameter('roi_left_fraction', 0.05)
        self.declare_parameter('roi_right_fraction', 0.95)

        # ============ Line clustering & filtering parameters ============
        self.declare_parameter('angle_cluster_tolerance', 0.2)  # radians (~11 deg)
        self.declare_parameter('min_cluster_length', 30.0)  # minimum total length to accept cluster

        # ============ Temporal smoothing parameters ============
        self.declare_parameter('smoothing_window', 5)  # deque max length for temporal averaging
        self.declare_parameter('normalize_method', 'image')  # 'image' (0..1) or 'centered' (-1..1)

        # ============ Debug parameters ============
        self.declare_parameter('debug_mask', True)
        self.declare_parameter('debug_image', True)

        # ============ ROS infrastructure ============
        image_topic = self.get_parameter('image_topic').value
        self.bridge = CvBridge()

        self.pos_pub = self.create_publisher(PointStamped, '/mangueira/position', 10)
        self.angle_pub = self.create_publisher(Float64, '/mangueira/angle', 10)
        self.detections_pub = self.create_publisher(Detection2DArray, '/mangueira/detections', 10)
        self.image_pub = self.create_publisher(Image, '/mangueira_detector/image', 10)
        self.mask_pub = self.create_publisher(Image, '/mangueira_detector/mask', _DBG_QOS)

        self.sub = self.create_subscription(CompressedImage, image_topic, self.image_callback, 10)

        # ============ State for temporal smoothing ============
        smoothing_window = int(self.get_parameter('smoothing_window').value)
        self._pos_x_history = deque(maxlen=smoothing_window)
        self._pos_y_history = deque(maxlen=smoothing_window)
        self._angle_history = deque(maxlen=smoothing_window)

        self.get_logger().info(
            f'Mangueira detector (lanes-inspired) started.\n'
            f'  Image topic: {image_topic}\n'
            f'  ROI enabled: {self.get_parameter("roi_enable").value}\n'
            f'  Smoothing window: {smoothing_window}'
        )

    def _build_red_mask(self, frame_bgr):
        # Build red mask using two HSV ranges to handle hue wrap-around
        l1_h = int(self.get_parameter('red_lower1_h').value)
        l1_s = int(self.get_parameter('red_lower1_s').value)
        l1_v = int(self.get_parameter('red_lower1_v').value)

        u1_h = int(self.get_parameter('red_upper1_h').value)
        u1_s = int(self.get_parameter('red_upper1_s').value)
        u1_v = int(self.get_parameter('red_upper1_v').value)

        l2_h = int(self.get_parameter('red_lower2_h').value)
        l2_s = int(self.get_parameter('red_lower2_s').value)
        l2_v = int(self.get_parameter('red_lower2_v').value)

        u2_h = int(self.get_parameter('red_upper2_h').value)
        u2_s = int(self.get_parameter('red_upper2_s').value)
        u2_v = int(self.get_parameter('red_upper2_v').value)

        kernel_size = int(self.get_parameter('morph_kernel_size').value)

        blurred = cv2.GaussianBlur(frame_bgr, (11, 11), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        lower1 = np.array([l1_h, l1_s, l1_v], dtype=np.uint8)
        upper1 = np.array([u1_h, u1_s, u1_v], dtype=np.uint8)
        lower2 = np.array([l2_h, l2_s, l2_v], dtype=np.uint8)
        upper2 = np.array([u2_h, u2_s, u2_v], dtype=np.uint8)

        mask1 = cv2.inRange(hsv, lower1, upper1)
        mask2 = cv2.inRange(hsv, lower2, upper2)
        mask = cv2.bitwise_or(mask1, mask2)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return mask

    def image_callback(self, msg):
        """Main processing pipeline: mask -> ROI -> Canny -> Hough -> cluster -> smooth -> publish"""
        try:
            frame = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as exc:
            self.get_logger().error(f'Failed to convert image: {exc}')
            return

        # Resize frame
        resize_width = int(self.get_parameter('resize_width').value)
        scale = resize_width / float(frame.shape[1])
        frame = cv2.resize(frame, (resize_width, int(frame.shape[0] * scale)))
        height, width = frame.shape[:2]

        # ============ Step 1: Build red/orange color mask ============
        mask = self._build_red_mask(frame)

        # ============ Step 2: Apply ROI if enabled ============
        if bool(self.get_parameter('roi_enable').value):
            roi_type = self.get_parameter('roi_type').value
            roi_params = {
                'top_fraction': float(self.get_parameter('roi_top_fraction').value),
                'bottom_fraction': float(self.get_parameter('roi_bottom_fraction').value),
                'left_fraction': float(self.get_parameter('roi_left_fraction').value),
                'right_fraction': float(self.get_parameter('roi_right_fraction').value),
            }
            mask = apply_roi_mask(mask, roi_type=roi_type, roi_params=roi_params)

        # Debug: publish mask as Image
        if bool(self.get_parameter('debug_mask').value):
            try:
                mask_dbg = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                cv2.putText(mask_dbg, f'orange px={int(np.count_nonzero(mask))}', (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                mask_msg = self.bridge.cv2_to_imgmsg(mask_dbg, encoding='bgr8')
                mask_msg.header = msg.header
                self.mask_pub.publish(mask_msg)
            except CvBridgeError as e:
                self.get_logger().error(f'Failed to publish mask image: {e}')

        # ============ Step 3: Canny edge detection ============
        canny_low = int(self.get_parameter('canny_low').value)
        canny_high = int(self.get_parameter('canny_high').value)
        edges = cv2.Canny(mask, canny_low, canny_high)

        # ============ Step 4: HoughLinesP line detection ============
        rho = 1
        theta = np.pi / 180.0
        threshold = int(self.get_parameter('hough_threshold').value)
        min_line_length = int(self.get_parameter('min_line_length').value)
        max_line_gap = int(self.get_parameter('max_line_gap').value)

        lines = cv2.HoughLinesP(edges, rho, theta, threshold, minLineLength=min_line_length, maxLineGap=max_line_gap)

        # ============ Step 5: Convert to slope/intercept and cluster ============
        line_data = lines_to_slope_intercept(lines)

        angle_tol = float(self.get_parameter('angle_cluster_tolerance').value)
        clusters = cluster_lines_by_angle(line_data, angle_tolerance=angle_tol)

        # Filter clusters by minimum total length
        min_cluster_len = float(self.get_parameter('min_cluster_length').value)
        clusters = [c for c in clusters if sum(l[2] for l in c) >= min_cluster_len]

        annotated = frame.copy()
        detection_array = Detection2DArray()
        detection_array.header = msg.header

        best_detection = None
        best_confidence = 0.0

        # ============ Step 6: Process best cluster and publish ============
        if clusters:
            # Find best cluster (by total length)
            best_cluster = max(clusters, key=lambda c: sum(l[2] for l in c))

            avg_result = average_cluster(best_cluster)
            if avg_result is not None:
                avg_slope, avg_intercept, total_length, center_x, center_y = avg_result

                # Reconstruct line coordinates
                y_bottom = height - 1
                y_top = int(height * 0.3)  # Extend line upwards for visualization
                line_coords = make_line_coordinates(avg_slope, avg_intercept, y_bottom, y_top)
                x1, y1, x2, y2 = line_coords

                # Clamp to image bounds
                x1 = max(0, min(width - 1, x1))
                x2 = max(0, min(width - 1, x2))
                y1 = max(0, min(height - 1, y1))
                y2 = max(0, min(height - 1, y2))

                # Compute angle (atan2(dx, dy) so 0 = vertical)
                dx = float(x2 - x1)
                dy = float(y2 - y1)
                if dy != 0.0 or dx != 0.0:
                    angle_raw = math.atan2(dx, dy)
                    # Normalize to [-pi/2, pi/2]
                    while angle_raw > math.pi / 2:
                        angle_raw -= math.pi
                    while angle_raw < -math.pi / 2:
                        angle_raw += math.pi
                else:
                    angle_raw = 0.0

                # Compute confidence
                diag = math.hypot(width, height)
                confidence = float(total_length / diag)

                # ============ Step 7: Temporal smoothing ============
                norm_x = float(center_x) / float(width) if self.get_parameter('normalize_method').value == 'image' else 2 * float(center_x) / float(width) - 1.0
                norm_y = float(center_y) / float(height) if self.get_parameter('normalize_method').value == 'image' else 2 * float(center_y) / float(height) - 1.0

                smoothed_x = smooth_value_deque(norm_x, self._pos_x_history, use_circular=False)
                smoothed_y = smooth_value_deque(norm_y, self._pos_y_history, use_circular=False)
                smoothed_angle = smooth_value_deque(angle_raw, self._angle_history, use_circular=True)

                best_detection = {
                    'center_x': center_x,
                    'center_y': center_y,
                    'angle': smoothed_angle,
                    'norm_x': smoothed_x,
                    'norm_y': smoothed_y,
                    'confidence': confidence,
                    'x1': x1,
                    'y1': y1,
                    'x2': x2,
                    'y2': y2,
                    'total_length': total_length,
                }
                best_confidence = confidence

                # ============ Step 8: Publish messages ============
                pos_msg = PointStamped()
                pos_msg.header = msg.header
                pos_msg.point.x = smoothed_x
                pos_msg.point.y = smoothed_y
                pos_msg.point.z = confidence
                self.pos_pub.publish(pos_msg)

                angle_msg = Float64()
                angle_msg.data = float(smoothed_angle)
                self.angle_pub.publish(angle_msg)

                # Build Detection2D message
                det = Detection2D()
                det.header = msg.header
                bbox = BoundingBox2D()
                bbox.center.position.x = smoothed_x
                bbox.center.position.y = smoothed_y
                bbox.size_x = min(0.5, total_length / float(width))
                bbox.size_y = min(0.2, total_length / float(height))
                det.bbox = bbox
                hyp = ObjectHypothesisWithPose()
                hyp.hypothesis.class_id = 'mangueira'
                hyp.hypothesis.score = float(min(0.99, confidence))
                det.results.append(hyp)
                detection_array.detections.append(det)

        # Publish detection array (even if empty)
        self.detections_pub.publish(detection_array)

        # ============ Step 9: Annotate and debug publish ============
        if bool(self.get_parameter('debug_image').value):
            # Draw all detected lines faintly
            if line_data:
                for slope, intercept, length, x1, y1, x2, y2 in line_data:
                    cv2.line(annotated, (int(x1), int(y1)), (int(x2), int(y2)), (100, 100, 255), 1)

            # Draw best cluster lines in yellow
            if clusters and best_detection:
                best_cluster = max(clusters, key=lambda c: sum(l[2] for l in c))
                for slope, intercept, length, x1, y1, x2, y2 in best_cluster:
                    cv2.line(annotated, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)

                # Draw final averaged line in green
                x1, y1, x2, y2 = best_detection['x1'], best_detection['y1'], best_detection['x2'], best_detection['y2']
                cv2.line(annotated, (x1, y1), (x2, y2), (0, 255, 0), 3)

                # Draw arrow pointing along line
                cx, cy = int(best_detection['center_x']), int(best_detection['center_y'])
                arrow_len = int(min(40, best_detection['total_length'] * 0.25))
                angle = best_detection['angle']
                left = (int(x2 + arrow_len * math.sin(angle + 0.5)),
                        int(y2 + arrow_len * math.cos(angle + 0.5)))
                right = (int(x2 + arrow_len * math.sin(angle - 0.5)),
                         int(y2 + arrow_len * math.cos(angle - 0.5)))
                cv2.line(annotated, (x2, y2), left, (0, 255, 0), 2)
                cv2.line(annotated, (x2, y2), right, (0, 255, 0), 2)

                # Draw center circle
                cv2.circle(annotated, (cx, cy), 5, (0, 0, 255), -1)

                # Draw info text
                angle_deg = math.degrees(best_detection['angle'])
                conf = best_detection['confidence']
                cv2.putText(annotated, f'angle={angle_deg:.1f}deg conf={conf:.2f}',
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(annotated, f'pos=({best_detection["norm_x"]:.2f}, {best_detection["norm_y"]:.2f})',
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            else:
                cv2.putText(annotated, 'no line detected', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            try:
                ann_msg = self.bridge.cv2_to_imgmsg(annotated, encoding='bgr8')
                ann_msg.header = msg.header
                self.image_pub.publish(ann_msg)
            except CvBridgeError as e:
                self.get_logger().error(f'Failed to publish annotated image: {e}')


def main(args=None):
    rclpy.init(args=args)
    node = MangueiraDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
