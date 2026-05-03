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
        
        output_frame = current_frame.copy()
        
        for cnt in contours:
            if cv2.contourArea(cnt) > 50:
                c_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
                cv2.drawContours(c_mask, [cnt], -1, 255, -1)
                mean_hsv = cv2.mean(hsv, mask=c_mask)[:3]
                
                cv2.drawContours(output_frame, [cnt], -1, (0, 255, 0), 2)
                M = cv2.moments(cnt)
                if M['m00'] != 0:
                    cx_cnt = int(M['m10']/M['m00'])
                    cy_cnt = int(M['m01']/M['m00'])
                    label = f"H:{int(mean_hsv[0])} S:{int(mean_hsv[1])} V:{int(mean_hsv[2])}"
                    cv2.putText(output_frame, label, (cx_cnt - 20, cy_cnt - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        cv2.imshow("Ball Detector", output_frame)
        cv2.waitKey(1)
        
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
