import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from custom_msgs.msg import BallDetection
from cv_bridge import CvBridge
import cv2
import numpy as np

class BallDetectorNode(Node):
    def __init__(self):
        super().__init__('ball_detector')
        
        # Publisher & Subscriber
        self.publisher_ = self.create_publisher(BallDetection, 'ball_detection', 10)
        self.subscription = self.create_subscription(
            Image,
            'horizontal_camera',
            self.image_callback,
            10)
        self.br = CvBridge()
        
        self.get_logger().info('Ball Detector Node has been started.')

    def image_callback(self, msg):
        try:
            # Convert ROS Image message to OpenCV image
            current_frame = self.br.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'Failed to convert image: {e}')
            return
            
        hsv = cv2.cvtColor(current_frame, cv2.COLOR_BGR2HSV)
        
        # Calibrar certinho isso aqui (está full carteado) 
        lower_red = np.array([0, 120, 70])
        upper_red = np.array([10, 255, 255])
        mask = cv2.inRange(hsv, lower_red, upper_red)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        det_msg = BallDetection()
        det_msg.header.stamp = self.get_clock().now().to_msg()
        det_msg.header.frame_id = 'horizontal_camera_link'
        
        if contours:
            # pega o maior contorno
            c = max(contours, key=cv2.contourArea)
            if cv2.contourArea(c) > 100: # min area threshold
                M = cv2.moments(c)
                if M['m00'] != 0:
                    cx = int(M['m10']/M['m00'])
                    cy = int(M['m01']/M['m00'])
                    
                    det_msg.is_detected = True
                    det_msg.center_position.x = float(cx)
                    det_msg.center_position.y = float(cy)
                    det_msg.center_position.z = 0.0
                    
                    # estimando a distância
                    _, radius = cv2.minEnclosingCircle(c)
                    det_msg.distance_estimate = float(100.0 / (radius + 1e-5)) # placeholder equation
                    
                    self.publisher_.publish(det_msg)
                    return
        
        det_msg.is_detected = False
        det_msg.center_position.x = 0.0
        det_msg.center_position.y = 0.0
        det_msg.center_position.z = 0.0
        det_msg.distance_estimate = -1.0
        self.publisher_.publish(det_msg)

def main(args=None):
    rclpy.init(args=args)
    node = BallDetectorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
