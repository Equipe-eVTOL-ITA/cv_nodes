import cv2 as cv
import math

import rclpy
from rclpy.node import Node
import cv2
import numpy as np
from rclpy.qos import QoSProfile, ReliabilityPolicy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import CompressedImage
from vision_msgs.msg import Detection2DArray, Detection2D
from vision_msgs.msg import BoundingBox2D
from std_msgs.msg import String

class BaseDetector(Node):
    def __init__(self):
        super().__init__('base_detector_node')

        self.declare_parameter('image_topic', '/vertical_camera/compressed')
        self.image_topic = self.get_parameter('image_topic').get_parameter_value().string_value

        qos_profile = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT
        )

        self.bridge = CvBridge()
        self.latest_image_msg = None  # To store the latest image message
        self.subscriber = self.create_subscription(
            CompressedImage,
            self.image_topic,
            self.base_callback,
            qos_profile
        )
    
    def base_callback(self, msg):
        try:
            frame = self.bridge.compressed_imgmsg_to_cv2(self.latest_image_msg, "bgr8")
            cinza = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

            blured = cv.GaussianBlur(cinza, (7, 7), 0)

            sobel_x = cv.Sobel(blured, cv.CV_64F, 1, 0, ksize=3)
            sobel_y = cv.Sobel(blured, cv.CV_64F, 0, 1, ksize=3)

            sobel_x_abs = cv.convertScaleAbs(sobel_x)
            sobel_y_abs = cv.convertScaleAbs(sobel_y)

            sobel_completo = cv.bitwise_or(sobel_x_abs, sobel_y_abs)

            _, binario = cv.threshold(sobel_completo, 50, 255, cv.THRESH_BINARY)

            contornos, _ = cv.findContours(binario, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

            circulos_encontrados = []
            for c in contornos:
                area = cv.contourArea(c)

                if area < 100:
                    continue # filtra os círculos muito pequenos (noise)

                perimetro = cv.arcLength(c, True)

                if perimetro == 0:
                    continue

                circularidade = (4*math.pi*area)/(perimetro*perimetro)
                print(circularidade)

                if 0.6 < circularidade < 1.2:
                    circulos_encontrados.append(c)

            cv.drawContours(frame, circulos_encontrados, -1, (0,255,0), thickness=2)

            cv.imshow('deteccao', frame)
            cv.imshow('binario', binario)

        except CvBridgeError as e:
            self.get_logger().error(f"Failed to convert image: {str(e)}")
            return

def main(args=None):
    rclpy.init(args=args)
    node = BaseDetector()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()