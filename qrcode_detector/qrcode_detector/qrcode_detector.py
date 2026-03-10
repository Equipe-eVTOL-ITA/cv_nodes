import rclpy
from rclpy.node import Node
import cv2
import numpy as np
from sensor_msgs.msg import CompressedImage, Image
from vision_msgs.msg import Detection2DArray, Detection2D, BoundingBox2D
from std_msgs.msg import String
from std_srvs.srv import Trigger
from cv_bridge import CvBridge
import os
from datetime import datetime

class QRCodeDetectionNode(Node):
    def __init__(self):
        super().__init__('qr_code_detection_node')

        # CV Bridge for converting between ROS and OpenCV images
        self.bridge = CvBridge()

        # Declare parameters for topic and message type
        self.declare_parameter('image_topic', '/depth_camera/image_raw')
        self.declare_parameter('use_compressed', False)
        
        image_topic = self.get_parameter('image_topic').get_parameter_value().string_value
        use_compressed = self.get_parameter('use_compressed').get_parameter_value().bool_value
        
        # ROS 2 Subscriber for camera images (use module-level imports to avoid scope issues)
        if use_compressed:
            self.image_sub = self.create_subscription(
                CompressedImage,
                image_topic,
                self.compressed_image_callback,
                10
            )
            self.get_logger().info(f'Subscribed to COMPRESSED image topic: {image_topic}')
        else:
            self.image_sub = self.create_subscription(
                Image,
                image_topic,
                self.raw_image_callback,
                10
            )
            self.get_logger().info(f'Subscribed to RAW image topic: {image_topic}')

        # ROS 2 Publishers
        self.qr_location_pub = self.create_publisher(Detection2DArray, '/vertical_classification', 10)
        self.annotated_image_pub = self.create_publisher(CompressedImage, '/annotated_image/compressed', 10)
        self.qr_string_pub = self.create_publisher(String, '/qr_code_string', 10)

        # OpenCV QR Code Detector
        self.qr_detector = cv2.QRCodeDetector()

        # Configurar diretório para salvar imagens
        self.save_directory = os.path.expanduser("~/qr_code_images")
        os.makedirs(self.save_directory, exist_ok=True)
        
        # Controle de salvamento (evita salvar múltiplas imagens do mesmo QR)
        self.last_saved_content = ""
        self.save_cooldown = 0
        
        # Parâmetros configuráveis
        self.declare_parameter('save_images', True)
        self.declare_parameter('save_cooldown_frames', 50)  # Evita salvar o mesmo QR por 5s
        
        self.save_images_enabled = self.get_parameter('save_images').get_parameter_value().bool_value
        self.cooldown_frames = self.get_parameter('save_cooldown_frames').get_parameter_value().integer_value

        # Serviço para capturar imagem sob demanda
        self.capture_service = self.create_service(
            Trigger, 
            '/capture_qr_image', 
            self.capture_qr_image_callback
        )
        
        # Armazenar último frame para captura sob demanda
        self.last_frame = None
        self.last_qr_detected = False
        self.last_qr_content = ""
        self.last_qr_bbox = None

        self.get_logger().info("QR Code Detector Node initialized")

    def raw_image_callback(self, msg):
        """Callback for raw (uncompressed) images from Gazebo"""
        try:
            # Convert ROS Image message to OpenCV format
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.detect_qr_codes(frame)
        except Exception as e:
            self.get_logger().error(f"Error in raw_image_callback: {str(e)}")
    
    def compressed_image_callback(self, msg):
        """Callback for compressed images (original callback)"""
        try:
            # Convert compressed image to OpenCV format
            np_arr = np.frombuffer(msg.data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if frame is None:
                self.get_logger().warn("Failed to decode image")
                return
            
            self.detect_qr_codes(frame)
            
        except Exception as e:
            self.get_logger().error(f"Error in compressed_image_callback: {str(e)}")
    
    def image_callback(self, msg):
        """Legacy callback - redirects to compressed callback for backward compatibility"""
        self.compressed_image_callback(msg)

    def detect_qr_codes(self, frame):
        """Process frame to detect QR codes"""

        # Armazenar último frame para serviço de captura
        self.last_frame = frame.copy()

        # Detectar e decodificar QR Codes
        decoded_text, bbox, _ = self.qr_detector.detectAndDecode(frame)
        
        detections_msg = Detection2DArray()
        detections_msg.header.stamp = self.get_clock().now().to_msg()
        annotated_frame = frame.copy()

        if bbox is not None and len(decoded_text) > 0:
            # QR Code detectado
            bbox = bbox[0].astype(int)  # Converter para inteiros
            
            # Calcular bounding box retangular
            x_min = int(np.min(bbox[:, 0]))
            y_min = int(np.min(bbox[:, 1]))
            x_max = int(np.max(bbox[:, 0]))
            y_max = int(np.max(bbox[:, 1]))
            
            # Coordenadas normalizadas
            height, width = frame.shape[:2]
            x_center = ((x_min + x_max) / 2) / width
            y_center = ((y_min + y_max) / 2) / height
            bbox_width = (x_max - x_min) / width
            bbox_height = (y_max - y_min) / height

            # Criar Detection2D
            detection = Detection2D()
            detection.bbox = BoundingBox2D()
            detection.bbox.center.position.x = x_center
            detection.bbox.center.position.y = y_center
            detection.bbox.size_x = bbox_width
            detection.bbox.size_y = bbox_height
            detections_msg.detections.append(detection)

            # Desenhar bounding box
            cv2.rectangle(annotated_frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(annotated_frame, f"QR: {decoded_text}", (x_min, y_min-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Atualizar informações para serviço de captura
            self.last_qr_detected = True
            self.last_qr_content = decoded_text
            self.last_qr_bbox = (x_min, y_min, x_max, y_max)

            # Salvar imagem do QR Code se habilitado
            if self.save_images_enabled:
                self.save_qr_image(frame, decoded_text, bbox, x_min, y_min, x_max, y_max)

            # Publicar texto do QR Code
            qr_string_msg = String()
            qr_string_msg.data = decoded_text
            self.qr_string_pub.publish(qr_string_msg)

        else:
            # Nenhum QR Code detectado
            self.last_qr_detected = False
            self.last_qr_content = ""
            self.last_qr_bbox = None
            
            default_detection = Detection2D()
            default_detection.bbox = BoundingBox2D()
            default_detection.bbox.center.position.x = 0.0
            default_detection.bbox.center.position.y = 0.0
            default_detection.bbox.size_x = 0.0
            default_detection.bbox.size_y = 0.0
            detections_msg.detections.append(default_detection)

            # Publicar string vazia
            qr_string_msg = String()
            qr_string_msg.data = ""
            self.qr_string_pub.publish(qr_string_msg)

        # Publicar detecções
        self.qr_location_pub.publish(detections_msg)

        # Publicar imagem anotada
        compressed_image_msg = CompressedImage()
        compressed_image_msg.header.stamp = self.get_clock().now().to_msg()
        compressed_image_msg.format = "jpeg"
        compressed_image_msg.data = np.array(cv2.imencode('.jpg', annotated_frame)[1]).tobytes()
        self.annotated_image_pub.publish(compressed_image_msg)
        
        # Decrementar cooldown
        if self.save_cooldown > 0:
            self.save_cooldown -= 1

    def save_qr_image(self, frame, qr_content, bbox_points, x_min, y_min, x_max, y_max):
        """Salva a imagem do QR Code detectado"""
        
        # Verificar cooldown para evitar salvar o mesmo QR repetidamente
        if (qr_content == self.last_saved_content and self.save_cooldown > 0) or not qr_content.strip():
            return
            
        try:
            # Timestamp para nome único
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # milliseconds
            
            # Nome do arquivo seguro (remover caracteres especiais do conteúdo do QR)
            safe_content = "".join(c for c in qr_content if c.isalnum() or c in (' ', '-', '_')).rstrip()
            safe_content = safe_content[:50]  # Limitar tamanho
            if not safe_content:
                safe_content = "unknown_qr"
            
            # Salvar imagem completa com anotação
            full_filename = f"qr_full_{timestamp}_{safe_content}.jpg"
            full_path = os.path.join(self.save_directory, full_filename)
            
            # Criar imagem anotada
            annotated_image = frame.copy()
            cv2.rectangle(annotated_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 3)
            cv2.putText(annotated_image, f"QR: {qr_content}", (x_min, y_min-15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # Adicionar timestamp na imagem
            cv2.putText(annotated_image, timestamp, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imwrite(full_path, annotated_image)
            
            # Salvar também recorte do QR Code
            margin = 20  # Margem ao redor do QR
            crop_x_min = max(0, x_min - margin)
            crop_y_min = max(0, y_min - margin) 
            crop_x_max = min(frame.shape[1], x_max + margin)
            crop_y_max = min(frame.shape[0], y_max + margin)
            
            cropped_qr = frame[crop_y_min:crop_y_max, crop_x_min:crop_x_max]
            
            crop_filename = f"qr_crop_{timestamp}_{safe_content}.jpg"
            crop_path = os.path.join(self.save_directory, crop_filename)
            cv2.imwrite(crop_path, cropped_qr)
            
            # Log do salvamento
            self.get_logger().info(f"QR Code images saved:")
            self.get_logger().info(f"  Full: {full_path}")
            self.get_logger().info(f"  Crop: {crop_path}")
            self.get_logger().info(f"  Content: '{qr_content}'")
            
            # Atualizar controle de cooldown
            self.last_saved_content = qr_content
            self.save_cooldown = self.cooldown_frames
            
        except Exception as e:
            self.get_logger().error(f"Error saving QR Code image: {str(e)}")

    def capture_qr_image_callback(self, request, response):
        """Serviço para capturar imagem do QR Code sob demanda"""
        
        if self.last_frame is None:
            response.success = False
            response.message = "No frame available for capture"
            return response
            
        if not self.last_qr_detected or not self.last_qr_bbox:
            response.success = False  
            response.message = "No QR Code currently detected"
            return response
            
        try:
            # Forçar salvamento da imagem atual
            x_min, y_min, x_max, y_max = self.last_qr_bbox
            
            # Timestamp para captura manual
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            safe_content = "".join(c for c in self.last_qr_content if c.isalnum() or c in (' ', '-', '_')).rstrip()[:50]
            if not safe_content:
                safe_content = "manual_capture"
                
            # Salvar imagem completa
            full_filename = f"qr_manual_{timestamp}_{safe_content}.jpg"
            full_path = os.path.join(self.save_directory, full_filename)
            
            annotated_image = self.last_frame.copy()
            cv2.rectangle(annotated_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 3)
            cv2.putText(annotated_image, f"QR: {self.last_qr_content}", (x_min, y_min-15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(annotated_image, f"MANUAL CAPTURE - {timestamp}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            cv2.imwrite(full_path, annotated_image)
            
            response.success = True
            response.message = f"QR Code image saved: {full_path}\nContent: '{self.last_qr_content}'"
            
            self.get_logger().info(f"Manual QR Code capture: {full_path}")
            
        except Exception as e:
            response.success = False
            response.message = f"Error capturing QR Code: {str(e)}"
            self.get_logger().error(f"Manual capture error: {str(e)}")
            
        return response

    def __del__(self):
        # No need to release camera since we're using ROS2 topic subscription
        pass


def main(args=None):
    rclpy.init(args=args)
    node = QRCodeDetectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()