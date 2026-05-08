import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from custom_msgs.msg import BallDetection
from cv_bridge import CvBridge
import cv2
import numpy as np
import math

class BallDetectorNode(Node):
    def __init__(self):
        super().__init__('ball_detector')
        
        # Parâmetros Dinâmicos (HSV, Morfologia, Transição)
        self.declare_parameter('h_min', 0)
        self.declare_parameter('h_max', 20)
        self.declare_parameter('s_min', 100)
        self.declare_parameter('s_max', 255)
        self.declare_parameter('v_min', 100)
        self.declare_parameter('v_max', 255)
        self.declare_parameter('morph_kernel_size', 5)
        self.declare_parameter('l_transicao', 150) # limiar para transição longe/perto

        # Publisher & Subscriber
        self.publisher_ = self.create_publisher(BallDetection, 'ball_detection', 10)
        self.debug_pub_ = self.create_publisher(CompressedImage, 'ball_detection_image', 10)
        self.subscription = self.create_subscription(
            CompressedImage,
            '/horizontal_camera/compressed',
            self.image_callback,
            10)
        self.br = CvBridge()
        
        self.get_logger().info('Ball Detector Node (Missao CV) has been started.')

    def image_callback(self, msg):
        try:
            current_frame = self.br.compressed_imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'Failed to convert image: {e}')
            return
            
        height, width = current_frame.shape[:2]
        
        # 1. Conversão HSV
        hsv = cv2.cvtColor(current_frame, cv2.COLOR_BGR2HSV)
        
        # Leitura dos Parâmetros
        h_min = self.get_parameter('h_min').value
        h_max = self.get_parameter('h_max').value
        s_min = self.get_parameter('s_min').value
        s_max = self.get_parameter('s_max').value
        v_min = self.get_parameter('v_min').value
        v_max = self.get_parameter('v_max').value
        k_size = self.get_parameter('morph_kernel_size').value
        
        lower_orange = np.array([h_min, s_min, v_min])
        upper_orange = np.array([h_max, s_max, v_max])
        
        # 2. Segmentação Binarizada
        mask = cv2.inRange(hsv, lower_orange, upper_orange)
        
        # 3. Tratamento Morfológico (Fechamento)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_size, k_size))
        mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # 4. Detecção da Mangueira (Hough)
        lines = cv2.HoughLinesP(mask_closed, 1, np.pi/180, threshold=50, minLineLength=50, maxLineGap=20)
        
        horizontal_lines = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = abs(math.degrees(math.atan2(y2 - y1, x2 - x1)))
                # Filtro: reter retas com inclinação próxima a 0 (horizontais)
                if angle < 15 or angle > 165:
                    horizontal_lines.append(line[0])
                    
        det_msg = BallDetection()
        det_msg.header.stamp = self.get_clock().now().to_msg()
        det_msg.header.frame_id = msg.header.frame_id
        det_msg.is_detected = False
        
        output_frame = current_frame.copy()
        
        if horizontal_lines:
            # 5. Extração de ROI (Bounding Box dinâmico)
            min_x = min([min(l[0], l[2]) for l in horizontal_lines])
            max_x = max([max(l[0], l[2]) for l in horizontal_lines])
            min_y = min([min(l[1], l[3]) for l in horizontal_lines])
            max_y = max([max(l[1], l[3]) for l in horizontal_lines])
            
            # Margem de segurança (padding)
            pad = 30
            min_x = max(0, min_x - pad)
            max_x = min(width, max_x + pad)
            min_y = max(0, min_y - pad)
            max_y = min(height, max_y + pad)
            
            cv2.rectangle(output_frame, (min_x, min_y), (max_x, max_y), (255, 0, 0), 2)
            
            roi_mask = mask_closed[min_y:max_y, min_x:max_x]
            
            # 6. Algoritmo de Detecção do Alvo (Modo Longe) - Forma Geométrica
            roi_mask = mask_closed[min_y:max_y, min_x:max_x]
            
            contours, _ = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            best_contour = None
            best_circularity = 0.0
            best_cx, best_cy = 0, 0
            best_radius = 5
            
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < 5:  # Filtra ruídos muito pequenos
                    continue
                    
                perimeter = cv2.arcLength(cnt, True)
                if perimeter == 0:
                    continue
                    
                circularity = 4 * math.pi * (area / (perimeter * perimeter))
                
                # Circularidade 1.0 é um círculo perfeito. Limiar de 0.4 acomoda deformações na lente e ruído
                if circularity > best_circularity and circularity > 0.4: 
                    best_circularity = circularity
                    best_contour = cnt
                    
                    M = cv2.moments(cnt)
                    if M['m00'] != 0:
                        best_cx = int(M['m10']/M['m00'])
                        best_cy = int(M['m01']/M['m00'])
                        
                        _, radius = cv2.minEnclosingCircle(cnt)
                        best_radius = max(5, int(radius))

            if best_contour is not None:
                # Transformação Global
                alvo_x_global = min_x + best_cx
                alvo_y_global = min_y + best_cy
                
                cv2.circle(output_frame, (int(alvo_x_global), int(alvo_y_global)), best_radius, (0, 0, 255), -1)
                
                # Cálculo de Erro normalizado [-1, 1]
                x_error = (alvo_x_global - width / 2.0) / (width / 2.0)
                y_error = (alvo_y_global - height / 2.0) / (height / 2.0)
                
                # Score de Confiança adaptado (Area)
                target_score = float(cv2.contourArea(best_contour))
                
                det_msg.is_detected = True
                det_msg.x_error = float(x_error)
                det_msg.y_error = float(y_error)
                det_msg.target_score = target_score
                det_msg.tracking_mode = 0  # 0 = Longe (Shape)
                
                # Campos de compatibilidade legada
                det_msg.center_position.x = float(alvo_x_global)
                det_msg.center_position.y = float(alvo_y_global)
                det_msg.center_position.z = 0.0
                det_msg.distance_estimate = -1.0
                    
        #cv2.imshow("Ball Detector", output_frame)
        #cv2.waitKey(1)
        
        if not det_msg.is_detected:
            det_msg.x_error = 0.0
            det_msg.y_error = 0.0
            det_msg.target_score = 0.0
            det_msg.tracking_mode = 0
            
            det_msg.center_position.x = 0.0
            det_msg.center_position.y = 0.0
            det_msg.center_position.z = 0.0
            det_msg.distance_estimate = -1.0
            
            cv2.putText(output_frame, "nao detectada", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            cv2.putText(output_frame, "detectada", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
        self.publisher_.publish(det_msg)
        
        # Publicar a imagem de debug
        try:
            debug_msg = self.br.cv2_to_compressed_imgmsg(output_frame)#, encoding="bgr8")
            debug_msg.header = msg.header
            self.debug_pub_.publish(debug_msg)
        except Exception as e:
            self.get_logger().error(f'Failed to publish debug image: {e}')

def main(args=None):
    rclpy.init(args=args)
    node = BallDetectorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
