from abc import abstractmethod

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)
from sensor_msgs.msg import CompressedImage


class Detector(Node):
    """
    Base class for ROS2 computer-vision detector nodes.

    Subclass and implement process_frame(). Everything else — camera
    subscription, processing-rate throttle, and debug-image helpers —
    is handled here and configured via ROS2 parameters / YAML.

    Parameters (all overridable via YAML or --ros-args):
      image_topic          (str)   /vertical_camera/compressed
      processing_frequency (float) 10.0  Hz; 0 = every frame
      debug_image          (bool)  True
      debug_mask           (bool)  False
      debug_publish_rate   (float) 5.0   Hz; 0 = every processed frame
      debug_max_width      (int)   320   px; 0 = no resize
      debug_jpeg_quality   (int)   60    1-100
    """

    def __init__(self, name: str):
        super().__init__(name)

        # ── Camera ────────────────────────────────────────────────────────
        self.declare_parameter('image_topic', '/vertical_camera/compressed')
        self.declare_parameter('processing_frequency', 10.0)

        # ── Debug ─────────────────────────────────────────────────────────
        self.declare_parameter('debug_image', True)
        self.declare_parameter('debug_mask', False)
        self.declare_parameter('debug_publish_rate', 5.0)
        self.declare_parameter('debug_max_width', 320)
        self.declare_parameter('debug_jpeg_quality', 60)

        image_topic = self.get_parameter('image_topic').value
        hz = float(self.get_parameter('processing_frequency').value)
        self._proc_interval_ns = int(1e9 / hz) if hz > 0.0 else 0
        self._last_proc_time = self.get_clock().now()

        dbg_rate = float(self.get_parameter('debug_publish_rate').value)
        self._debug_min_period = (1.0 / dbg_rate) if dbg_rate > 0.0 else 0.0
        self._last_debug_time = 0.0

        sensor_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=5,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
        )
        self.image_subscription = self.create_subscription(
            CompressedImage,
            image_topic,
            self._image_callback_base,
            sensor_qos,
        )

        self.bridge = CvBridge()
        self.get_logger().info(
            f'{name} subscribed to {image_topic} @ '
            f'{"unlimited" if hz <= 0 else f"{hz:.1f}"} Hz'
        )

    # ── Internal callback ─────────────────────────────────────────────────

    def _image_callback_base(self, msg: CompressedImage) -> None:
        now = self.get_clock().now()
        if (self._proc_interval_ns > 0 and
                (now - self._last_proc_time).nanoseconds < self._proc_interval_ns):
            return
        self._last_proc_time = now

        try:
            frame = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as exc:
            self.get_logger().error(f'Failed to convert image: {exc}')
            return

        self.process_frame(frame, msg.header)

    # ── Debug helpers (call from process_frame) ───────────────────────────

    def _debug_should_publish(self) -> bool:
        """Time-gated check: True when debug rate allows publishing."""
        if self._debug_min_period <= 0.0:
            return True
        now_s = self.get_clock().now().nanoseconds * 1e-9
        if now_s - self._last_debug_time >= self._debug_min_period:
            self._last_debug_time = now_s
            return True
        return False

    def _pub_debug(self, publisher, image: np.ndarray, header) -> None:
        """Resize + JPEG-encode + publish a debug image."""
        try:
            max_w = int(self.get_parameter('debug_max_width').value)
            if max_w > 0 and image.shape[1] > max_w:
                scale = max_w / float(image.shape[1])
                new_size = (max_w, max(1, int(image.shape[0] * scale)))
                image = cv2.resize(image, new_size)

            q = int(self.get_parameter('debug_jpeg_quality').value)
            ok, buf = cv2.imencode('.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), q])
            if not ok:
                return

            msg = CompressedImage()
            msg.format = 'jpeg'
            msg.data = buf.tobytes()
            msg.header = header
            publisher.publish(msg)
        except Exception as exc:
            self.get_logger().error(f'Failed to publish debug image: {exc}')

    # ── Abstract interface ────────────────────────────────────────────────

    @abstractmethod
    def process_frame(self, frame: np.ndarray, header) -> None:
        """
        Called for each frame that passes the processing-rate gate.

        Args:
            frame:  BGR image as numpy array.
            header: Original CompressedImage header (stamp + frame_id).

        Use self._debug_should_publish() and self._pub_debug() for throttled
        debug image publishing. Check get_parameter('debug_image').value and
        get_parameter('debug_mask').value before building debug images.
        """


def main(args=None):
    """Minimal example showing how to subclass Detector."""
    rclpy.init(args=args)

    class GrayscaleDetector(Detector):
        def __init__(self):
            super().__init__('grayscale_example')

        def process_frame(self, frame, header):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cv2.imshow('gray', gray)
            cv2.waitKey(1)

    node = GrayscaleDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
