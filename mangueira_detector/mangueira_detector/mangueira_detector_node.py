#!/usr/bin/env python3
"""
Rough-line Mangueira (hose) detector using an orange mask.

Detects rough line segments with HoughLinesP on an orange color mask
and publishes the largest line's normalized position and orientation
so the drone can align in the plane (lateral offset) and yaw.

Inspired by the ball detector's orange mask pipeline.
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


_DBG_QOS = QoSProfile(depth=5)


class MangueiraDetector(Node):
    def __init__(self):
        super().__init__('mangueira_detector')

        # Parameters (some defaults borrowed from ball_detector)
        self.declare_parameter('image_topic', '/vertical_camera/compressed')
        self.declare_parameter('resize_width', 600)
        self.declare_parameter('min_line_length', 30)
        self.declare_parameter('max_line_gap', 10)
        self.declare_parameter('hough_threshold', 30)
        self.declare_parameter('canny_low', 50)
        self.declare_parameter('canny_high', 150)

        # Red mask params (handle hue wrap-around with two ranges)
        # Target colors between #620000 (darker) and #D40000 (brighter)
        # Range split to handle H wrap-around around 0/179
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

        self.declare_parameter('debug_mask', True)
        self.declare_parameter('debug_image', True)

        # Publishers / subscribers
        image_topic = self.get_parameter('image_topic').value

        self.bridge = CvBridge()

        self.pos_pub = self.create_publisher(PointStamped, '/mangueira/position', 10)
        self.angle_pub = self.create_publisher(Float64, '/mangueira/angle', 10)
        self.detections_pub = self.create_publisher(Detection2DArray, '/mangueira/detections', 10)
        self.image_pub = self.create_publisher(Image, '/mangueira_detector/image', 10)
        self.mask_pub = self.create_publisher(Image, '/mangueira_detector/mask', _DBG_QOS)

        self.sub = self.create_subscription(CompressedImage, image_topic, self.image_callback, 10)

        self.get_logger().info(f'Mangueira rough-line detector started. Subscribed to {image_topic}')

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
        try:
            frame = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as exc:
            self.get_logger().error(f'Failed to convert image: {exc}')
            return

        resize_width = int(self.get_parameter('resize_width').value)
        scale = resize_width / float(frame.shape[1])
        frame = cv2.resize(frame, (resize_width, int(frame.shape[0] * scale)))
        height, width = frame.shape[:2]

        mask = self._build_red_mask(frame)

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

        # Edge detection and Hough lines
        canny_low = int(self.get_parameter('canny_low').value)
        canny_high = int(self.get_parameter('canny_high').value)
        edges = cv2.Canny(mask, canny_low, canny_high)

        rho = 1
        theta = np.pi / 180.0
        threshold = int(self.get_parameter('hough_threshold').value)
        min_line_length = int(self.get_parameter('min_line_length').value)
        max_line_gap = int(self.get_parameter('max_line_gap').value)

        lines = cv2.HoughLinesP(edges, rho, theta, threshold, minLineLength=min_line_length, maxLineGap=max_line_gap)

        annotated = frame.copy()
        detection_array = Detection2DArray()
        detection_array.header = msg.header

        best_line = None
        best_len = 0.0

        # Draw all lines faintly
        if lines is not None:
            for l in lines:
                x1, y1, x2, y2 = l[0]
                length = math.hypot(x2 - x1, y2 - y1)
                cv2.line(annotated, (x1, y1), (x2, y2), (255, 0, 0), 1)
                if length > best_len:
                    best_len = length
                    best_line = (x1, y1, x2, y2)

        # If a best line found, publish position and angle
        if best_line is not None:
            x1, y1, x2, y2 = best_line
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            # Normalized center [0,1]
            norm_x = float(cx) / float(width)
            norm_y = float(cy) / float(height)

            # Angle relative to vertical: use atan2(dx, dy) so 0 = vertical
            dx = float(x2 - x1)
            dy = float(y2 - y1)
            angle = math.atan2(dx, dy) if (dx != 0.0 or dy != 0.0) else 0.0
            # Normalize to [-pi/2, pi/2]
            while angle > math.pi/2:
                angle -= math.pi
            while angle < -math.pi/2:
                angle += math.pi

            # Confidence proportional to line length normalized by image diagonal
            diag = math.hypot(width, height)
            confidence = float(best_len / diag)

            # Publish PointStamped (x,y normalized, z=confidence)
            pos_msg = PointStamped()
            pos_msg.header = msg.header
            pos_msg.point.x = norm_x
            pos_msg.point.y = norm_y
            pos_msg.point.z = confidence
            self.pos_pub.publish(pos_msg)

            angle_msg = Float64()
            angle_msg.data = float(angle)
            self.angle_pub.publish(angle_msg)

            # Build a simple Detection2D for consumers expecting array
            det = Detection2D()
            det.header = msg.header
            # bounding box centered on line midpoint with small size
            bbox = BoundingBox2D()
            bbox.center.position.x = norm_x
            bbox.center.position.y = norm_y
            bbox.size_x = min(0.5, best_len / float(width))
            bbox.size_y = min(0.2, best_len / float(height))
            det.bbox = bbox
            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id = 'mangueira'
            hyp.hypothesis.score = float(min(0.99, confidence))
            det.results.append(hyp)
            detection_array.detections.append(det)

            # Annotate best line prominently
            cv2.line(annotated, (x1, y1), (x2, y2), (0, 255, 0), 3)
            # Arrow to indicate direction (pt2 is arrow tip)
            arrow_tip = (int(x2), int(y2))
            # Compute small arrow
            arrow_len = int(min(30, best_len * 0.2))
            ang = angle
            left = (int(arrow_tip[0] + arrow_len * math.sin(ang + 0.5)),
                    int(arrow_tip[1] + arrow_len * math.cos(ang + 0.5)))
            right = (int(arrow_tip[0] + arrow_len * math.sin(ang - 0.5)),
                     int(arrow_tip[1] + arrow_len * math.cos(ang - 0.5)))
            cv2.line(annotated, arrow_tip, left, (0, 255, 0), 2)
            cv2.line(annotated, arrow_tip, right, (0, 255, 0), 2)

            cv2.circle(annotated, (cx, cy), 5, (0, 0, 255), -1)
            cv2.putText(annotated, f'angle={math.degrees(angle):.1f}deg conf={confidence:.2f}',
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        else:
            cv2.putText(annotated, 'no line detected', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Publish detection array
        self.detections_pub.publish(detection_array)

        # Publish annotated image if requested
        if bool(self.get_parameter('debug_image').value):
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
