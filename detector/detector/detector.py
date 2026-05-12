from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
import numpy as np
from rclpy.qos import (
    QoSProfile,
    QoSReliabilityPolicy,
    QoSHistoryPolicy,
    QoSDurabilityPolicy
)
from abc import abstractmethod

class Detector(Node):
    def __init__(self, name: str, camera_topic: str, qos_depth: int = 1, processing_hz: float = 30.0):
        super().__init__(name)

        self.camera_topic = camera_topic
        self.processing_interval = (1.0/processing_hz) if processing_hz > 0 else 0.0
        self.last_processed_time = self.get_clock().now()

        self._qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=qos_depth,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE
        )

        self.image_subscription = self.create_subscription(
            CompressedImage,
            self.camera_topic,
            self._image_callback_base,
            self._qos
        )

        self.bridge = CvBridge()
    

    def _image_callback_base(self, msg: CompressedImage) -> None:
        current_time = self.get_clock().now()
        if(current_time-self.last_processed_time).nanoseconds < self.processing_interval*1e9:
            return # descarta o frame
        
        self.last_processed_time = current_time

        try:
            frame = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'Failed to convert image: {e}')
            return

        self.process_frame(frame, msg.header)
    
    @abstractmethod
    def process_frame(self, frame: np.ndarray, header) -> None:
        pass

def main(args=None):
    import cv2
    import rclpy

    class BallDetector(Detector):
        def __init__(self):
            super().__init__('ball_detector', '/vertical_camera/compressed', qos_depth=1, processing_hz=30)

        def process_frame(self, frame, header):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cv2.imshow('Escala de cinza', gray)
            cv2.waitKey(1)
    
    ball_detector_node = BallDetector()

    rclpy.init(args=args)
    try:
        rclpy.spin(ball_detector_node)
    except KeyboardInterrupt:
        pass
    finally:
        ball_detector_node.destroy_node()
        rclpy.try_shutdown()

if __name__ == '__main__':
    main()
