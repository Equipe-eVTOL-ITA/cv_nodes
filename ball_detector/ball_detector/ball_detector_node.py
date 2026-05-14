from collections import deque
import math

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from custom_msgs.msg import BallDetection
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import CompressedImage


_DBG_QOS = QoSProfile(
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=5,
    reliability=QoSReliabilityPolicy.BEST_EFFORT,
    durability=QoSDurabilityPolicy.VOLATILE,
)


class BallDetectorNode(Node):
    """Detect the orange competition ball in the horizontal camera stream."""

    def __init__(self):
        super().__init__('ball_detector')

        self.declare_parameter('image_topic', '/horizontal_camera/compressed')
        self.declare_parameter('resize_width', 600)
        self.declare_parameter('track_buffer', 64)
        self.declare_parameter('min_radius', 10)
        self.declare_parameter('min_area', 100.0)
        self.declare_parameter('max_area', 30000.0)
        self.declare_parameter('min_circularity', 0.55)
        self.declare_parameter('orange_h_min', 5)
        self.declare_parameter('orange_h_max', 30)  # widened: log shows dom H≈25-30
        self.declare_parameter('orange_s_min', 80)
        self.declare_parameter('orange_v_min', 50)
        self.declare_parameter('morph_kernel_size', 5)

        self.declare_parameter('debug_mask', True)
        self.declare_parameter('debug_image', True)

        self.publisher_ = self.create_publisher(BallDetection, 'ball_detection', 10)
        self.debug_pub_ = self.create_publisher(CompressedImage, 'ball_detection_image/compressed', _DBG_QOS)
        self.mask_pub_ = self.create_publisher(CompressedImage, 'ball_detector/mask/compressed', _DBG_QOS)

        image_topic = self.get_parameter('image_topic').value
        self.subscription = self.create_subscription(
            CompressedImage,
            image_topic,
            self.image_callback,
            10,
        )

        self.br = CvBridge()
        self.pts = deque(maxlen=int(self.get_parameter('track_buffer').value))
        self._frame_count = 0
        self.get_logger().info(f'Orange ball detector started. Subscribed to {image_topic}')

    def _pub_debug(self, publisher, image, header):
        try:
            msg = self.br.cv2_to_compressed_imgmsg(image)
            msg.header = header
            publisher.publish(msg)
        except Exception as exc:
            self.get_logger().error(f'Failed to publish debug image: {exc}')

    def _build_orange_mask(self, frame_bgr):
        h_min = int(self.get_parameter('orange_h_min').value)
        h_max = int(self.get_parameter('orange_h_max').value)
        s_min = int(self.get_parameter('orange_s_min').value)
        v_min = int(self.get_parameter('orange_v_min').value)
        kernel_size = int(self.get_parameter('morph_kernel_size').value)

        blurred = cv2.GaussianBlur(frame_bgr, (11, 11), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        lower_orange = np.array([h_min, s_min, v_min], dtype=np.uint8)
        upper_orange = np.array([h_max, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower_orange, upper_orange)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return mask

    def image_callback(self, msg):
        try:
            frame = self.br.compressed_imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as exc:
            self.get_logger().error(f'Failed to convert image: {exc}')
            return

        self._frame_count += 1

        resize_width = int(self.get_parameter('resize_width').value)
        scale = resize_width / float(frame.shape[1])
        frame = cv2.resize(frame, (resize_width, int(frame.shape[0] * scale)))
        height, width = frame.shape[:2]

        mask = self._build_orange_mask(frame)

        if bool(self.get_parameter('debug_mask').value):
            mask_dbg = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            cv2.putText(mask_dbg, f'orange px={int(np.count_nonzero(mask))}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            self._pub_debug(self.mask_pub_, mask_dbg, msg.header)

        contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        output = frame.copy()
        det_msg = BallDetection()
        det_msg.header.stamp = self.get_clock().now().to_msg()
        det_msg.header.frame_id = msg.header.frame_id
        det_msg.is_detected = False
        det_msg.x_error = 0.0
        det_msg.y_error = 0.0
        det_msg.target_score = 0.0
        det_msg.tracking_mode = 0
        det_msg.center_position.x = 0.0
        det_msg.center_position.y = 0.0
        det_msg.center_position.z = 0.0
        det_msg.distance_estimate = -1.0

        best_contour = None
        best_area = 0.0
        best_circularity = 0.0

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < float(self.get_parameter('min_area').value):
                continue

            perimeter = cv2.arcLength(contour, True)
            if perimeter <= 0.0:
                continue

            circularity = float(4.0 * math.pi * area / (perimeter * perimeter))
            if circularity < float(self.get_parameter('min_circularity').value):
                continue

            if area > best_area:
                best_contour = contour
                best_area = float(area)
                best_circularity = circularity

        if best_contour is not None:
            ((x, y), radius) = cv2.minEnclosingCircle(best_contour)
            moments = cv2.moments(best_contour)

            if radius >= float(self.get_parameter('min_radius').value) and moments['m00'] > 0.0:
                cx = int(moments['m10'] / moments['m00'])
                cy = int(moments['m01'] / moments['m00'])

                self.pts.appendleft((cx, cy))

                cv2.circle(output, (int(x), int(y)), int(radius), (0, 165, 255), 2)
                cv2.circle(output, (cx, cy), 5, (0, 0, 255), -1)
                cv2.drawMarker(output, (cx, cy), (255, 255, 255), cv2.MARKER_CROSS, int(radius * 2), 2)

                for i in range(1, len(self.pts)):
                    if self.pts[i - 1] is None or self.pts[i] is None:
                        continue
                    thickness = int(np.sqrt(len(self.pts) / float(i + 1)) * 2.5)
                    cv2.line(output, self.pts[i - 1], self.pts[i], (0, 0, 255), thickness)

                x_error = (cx - width / 2.0) / (width / 2.0)
                y_error = (cy - height / 2.0) / (height / 2.0)

                det_msg.is_detected = True
                det_msg.x_error = float(x_error)
                det_msg.y_error = float(y_error)
                det_msg.target_score = float(best_area)
                det_msg.tracking_mode = 0
                det_msg.center_position.x = float(cx)
                det_msg.center_position.y = float(cy)
                det_msg.center_position.z = 0.0
                det_msg.distance_estimate = -1.0

                cv2.putText(output, f'orange ball area={best_area:.0f} circ={best_circularity:.2f}',
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                self.pts.appendleft(None)
                cv2.putText(output, 'ball rejected', (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            self.pts.appendleft(None)
            cv2.putText(output, 'no orange ball detected', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        self.publisher_.publish(det_msg)

        if bool(self.get_parameter('debug_image').value):
            self._pub_debug(self.debug_pub_, output, msg.header)


def main(args=None):
    rclpy.init(args=args)
    node = BallDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
