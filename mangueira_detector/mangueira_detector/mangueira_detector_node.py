#!/usr/bin/env python3
"""
Red Hose (Mangueira) Detector

This ROS 2 node detects a red hose (1.25cm diameter) using the vertical camera
and publishes hose position information for the drone to align with it.

The detector uses HSV color segmentation to identify red pixels, then finds
thin, elongated contours that match the hose characteristics.

Optimized for very thin objects with specialized morphological operations
and confidence scoring based on shape characteristics.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage, Image
from geometry_msgs.msg import PointStamped
from std_msgs.msg import Float64
from vision_msgs.msg import Detection2D, Detection2DArray, BoundingBox2D, ObjectHypothesisWithPose
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
import math

class MangueiraDetector(Node):
    def __init__(self):
        super().__init__('mangueira_detector')

        # Declare parameters
        self.declare_parameter('image_topic', '/vertical_camera/compressed')
        self.declare_parameter('processing_frequency', 10.0)  # Hz
        self.declare_parameter('confidence_threshold', 0.5)
        
        # Hose detection parameters (optimized for very thin objects)
        self.declare_parameter('min_hose_area', 50)    # Very small for thin hose
        self.declare_parameter('max_hose_area', 1000)  # Maximum area to avoid large objects
        self.declare_parameter('morph_kernel_size', 3)
        self.declare_parameter('morph_iterations', 2)
        
        # Hose shape constraints (very relaxed for very thin objects)
        self.declare_parameter('min_aspect_ratio', 0.0)   # Very relaxed
        self.declare_parameter('max_aspect_ratio', 200.0) # Very high for thin objects
        self.declare_parameter('min_hose_length', 1)      # Very small minimum
        self.declare_parameter('max_hose_width', 800)     # Very relaxed
        
        # Red hose HSV parameters
        self.declare_parameter('red_lower1_h', 0)
        self.declare_parameter('red_lower1_s', 50)
        self.declare_parameter('red_lower1_v', 30)
        self.declare_parameter('red_upper1_h', 10)
        self.declare_parameter('red_upper1_s', 255)
        self.declare_parameter('red_upper1_v', 255)
        self.declare_parameter('red_lower2_h', 170)
        self.declare_parameter('red_lower2_s', 50)
        self.declare_parameter('red_lower2_v', 30)
        self.declare_parameter('red_upper2_h', 200)
        self.declare_parameter('red_upper2_s', 255)
        self.declare_parameter('red_upper2_v', 255)
        
        # Get parameter values
        self.image_topic = self.get_parameter('image_topic').get_parameter_value().string_value
        self.confidence_threshold = self.get_parameter('confidence_threshold').get_parameter_value().double_value
        processing_freq = self.get_parameter('processing_frequency').get_parameter_value().double_value
        
        # Hose detection parameters
        self.min_hose_area = self.get_parameter('min_hose_area').get_parameter_value().integer_value
        self.max_hose_area = self.get_parameter('max_hose_area').get_parameter_value().integer_value
        self.morph_kernel_size = self.get_parameter('morph_kernel_size').get_parameter_value().integer_value
        self.morph_iterations = self.get_parameter('morph_iterations').get_parameter_value().integer_value
        
        # Shape constraints
        self.min_aspect_ratio = self.get_parameter('min_aspect_ratio').get_parameter_value().double_value
        self.max_aspect_ratio = self.get_parameter('max_aspect_ratio').get_parameter_value().double_value
        self.min_hose_length = self.get_parameter('min_hose_length').get_parameter_value().integer_value
        self.max_hose_width = self.get_parameter('max_hose_width').get_parameter_value().integer_value
        
        # HSV color ranges for red hose (two ranges due to hue wrap-around)
        self.red_lower1 = np.array([
            self.get_parameter('red_lower1_h').get_parameter_value().integer_value,
            self.get_parameter('red_lower1_s').get_parameter_value().integer_value,
            self.get_parameter('red_lower1_v').get_parameter_value().integer_value
        ])
        self.red_upper1 = np.array([
            self.get_parameter('red_upper1_h').get_parameter_value().integer_value,
            self.get_parameter('red_upper1_s').get_parameter_value().integer_value,
            self.get_parameter('red_upper1_v').get_parameter_value().integer_value
        ])
        self.red_lower2 = np.array([
            self.get_parameter('red_lower2_h').get_parameter_value().integer_value,
            self.get_parameter('red_lower2_s').get_parameter_value().integer_value,
            self.get_parameter('red_lower2_v').get_parameter_value().integer_value
        ])
        self.red_upper2 = np.array([
            self.get_parameter('red_upper2_h').get_parameter_value().integer_value,
            self.get_parameter('red_upper2_s').get_parameter_value().integer_value,
            self.get_parameter('red_upper2_v').get_parameter_value().integer_value
        ])

        # Setup QoS profile
        qos_profile = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT
        )

        # Setup subscriber, publishers, and timer
        self.bridge = CvBridge()
        self.latest_image_msg = None
        
        self.subscriber = self.create_subscription(
            CompressedImage,
            self.image_topic,
            self.image_callback,
            qos_profile
        )

        # Create dynamic topic names
        base_topic = self.image_topic.replace('/compressed', '').replace('/', '')
        if base_topic.startswith('_'):
            base_topic = base_topic[1:]
        
        # Publishers
        self.hose_position_publisher = self.create_publisher(PointStamped, '/mangueira/position', 10)
        self.detection_publisher = self.create_publisher(Detection2DArray, '/mangueira/detections', 10)
        self.hose_angle_publisher = self.create_publisher(Float64, '/mangueira/angle', 10)  # NEW: Publish hose angle
        self.image_publisher = self.create_publisher(Image, f'/mangueira_detector/{base_topic}/image', 10)

        # Set up timer for processing images
        timer_period = 1.0 / processing_freq
        self.timer = self.create_timer(timer_period, self.process_image_callback)
        
        self.get_logger().info(f'Mangueira Detector initialized')
        self.get_logger().info(f'Subscribing to: {self.image_topic}')
        self.get_logger().info(f'Publishing position to: /mangueira/position')
        self.get_logger().info(f'Publishing detections to: /mangueira/detections')
        self.get_logger().info(f'Red HSV Range 1: H={self.red_lower1[0]}-{self.red_upper1[0]}, S={self.red_lower1[1]}-{self.red_upper1[1]}, V={self.red_lower1[2]}-{self.red_upper1[2]}')
        self.get_logger().info(f'Red HSV Range 2: H={self.red_lower2[0]}-{self.red_upper2[0]}, S={self.red_lower2[1]}-{self.red_upper2[1]}, V={self.red_lower2[2]}-{self.red_upper2[2]}')

    def image_callback(self, msg):
        """Store the latest image message"""
        self.latest_image_msg = msg

    def segment_red_hose(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Segment red hose from the image using specialized algorithms for very thin objects
        
        Args:
            image: Input BGR image
            
        Returns:
            mask: Binary mask of red hose pixels
            colored_mask: Colored mask for visualization
        """
        # Convert BGR to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Create masks for both red ranges (handle hue wrap-around)
        mask1 = cv2.inRange(hsv, self.red_lower1, self.red_upper1)
        mask2 = cv2.inRange(hsv, self.red_lower2, self.red_upper2)
        
        # Combine red masks
        mask = cv2.bitwise_or(mask1, mask2)
        
        # For very thin hose detection, use specialized morphological operations
        # Use thin kernels to connect fragmented hose segments without losing thin structures
        
        # Horizontal kernel for connecting broken horizontal hose segments
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 1))
        # Vertical kernel for connecting broken vertical hose segments  
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 7))
        # Diagonal kernels for diagonal hose orientations
        diagonal_kernel1 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.uint8)
        diagonal_kernel2 = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=np.uint8)
        
        # Apply directional closing operations
        mask_h = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, horizontal_kernel, iterations=2)
        mask_v = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, vertical_kernel, iterations=2)
        mask_d1 = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, diagonal_kernel1, iterations=1)
        mask_d2 = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, diagonal_kernel2, iterations=1)
        
        # Combine all directional results
        mask = cv2.bitwise_or(cv2.bitwise_or(mask_h, mask_v), cv2.bitwise_or(mask_d1, mask_d2))
        
        # Standard morphological operations with small kernel for thin objects
        small_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                                (self.morph_kernel_size, self.morph_kernel_size))
        
        # Dilate slightly to make thin objects more visible for contour detection
        mask = cv2.dilate(mask, small_kernel, iterations=1)
        
        # Close to connect nearby segments
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, small_kernel, iterations=self.morph_iterations)
        
        # Final opening to remove small noise while preserving thin structures
        tiny_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, tiny_kernel, iterations=1)
        
        # Create colored mask for visualization (red)
        colored_mask = np.zeros_like(image)
        colored_mask[mask > 0] = (0, 0, 255)  # Red in BGR
        
        return mask, colored_mask

    def analyze_hose_contours(self, mask: np.ndarray, image_shape: Tuple[int, int]) -> List[Dict]:
        """
        Analyze contours to find hose-like objects with angle detection
        
        Args:
            mask: Binary mask of potential hose pixels
            image_shape: (height, width) of the original image
            
        Returns:
            List of hose detection dictionaries with angle information
        """
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detections = []
        height, width = image_shape
        
        for contour in contours:
            # Calculate contour area
            area = cv2.contourArea(contour)
            
            # Very relaxed area filtering for thin objects
            if area < self.min_hose_area or area > self.max_hose_area:
                continue
            
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Calculate aspect ratio (length/width) - for thin hose, this can vary greatly
            aspect_ratio = max(w, h) / max(min(w, h), 1)  # Avoid division by zero
            
            # Very relaxed aspect ratio constraints
            if aspect_ratio < self.min_aspect_ratio or aspect_ratio > self.max_aspect_ratio:
                continue
            
            # Relaxed size constraints
            length = max(w, h)
            width_px = min(w, h)
            
            if length < self.min_hose_length or width_px > self.max_hose_width:
                continue
            
            # Calculate hose angle using contour fitting
            hose_angle = 0.0  # Default angle
            
            # Method 1: Try to fit ellipse for orientation
            if len(contour) >= 5:
                try:
                    ellipse = cv2.fitEllipse(contour)
                    # Get ellipse angle in degrees, convert to radians
                    ellipse_angle_deg = ellipse[2]
                    
                    # Convert to our coordinate system: 
                    # - 0° = vertical (pointing up in image = drone front)
                    # - positive angles = clockwise rotation from drone front
                    # - range: -π/2 to π/2
                    
                    # OpenCV ellipse angle is in degrees (0-180), where 0° is horizontal
                    # Convert to our system: subtract 90° to make 0° vertical (drone front)
                    adjusted_angle_deg = -ellipse_angle_deg
                    
                    # Convert to radians
                    hose_angle = math.radians(adjusted_angle_deg)
                    
                    # Normalize to [-π/2, π/2] to avoid 180° flips
                    # This ensures consistent angle relative to drone front
                    while hose_angle > math.pi/2:
                        hose_angle -= math.pi
                    while hose_angle < -math.pi/2:
                        hose_angle += math.pi
                        
                except:
                    # Method 2: Use line fitting as fallback
                    if len(contour) >= 10:
                        try:
                            # Reshape contour for cv2.fitLine
                            points = contour.reshape(-1, 2)
                            [vx, vy, x0, y0] = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01)
                            
                            # Calculate angle from vertical (drone front direction)
                            hose_angle = math.atan2(vx, vy)
                            
                            # Normalize to [-π/2, π/2] to avoid 180° flips
                            while hose_angle > math.pi/2:
                                hose_angle -= math.pi
                            while hose_angle < -math.pi/2:
                                hose_angle += math.pi
                        except:
                            hose_angle = 0.0
            
            # Calculate additional metrics for thin object validation
            
            # 1. Solidity (how solid/filled the contour is)
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0
            
            # 2. Extent (ratio of contour area to bounding rectangle area)
            rect_area = w * h
            extent = area / rect_area if rect_area > 0 else 0
            
            # 3. Perimeter-based metrics for thin objects
            perimeter = cv2.arcLength(contour, True)
            compactness = (perimeter * perimeter) / (4 * np.pi * area) if area > 0 else float('inf')
            
            # Calculate centroid
            M = cv2.moments(contour)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
            else:
                cx, cy = x + w//2, y + h//2
            
            # Advanced confidence calculation for thin hose detection
            confidence = 0.0
            
            # Factor 1: Red pixel density in bounding box
            roi = mask[y:y+h, x:x+w]
            red_pixel_ratio = np.sum(roi > 0) / (w * h) if (w * h) > 0 else 0
            confidence += 0.25 * red_pixel_ratio
            
            # Factor 2: Aspect ratio score (prefer elongated but not extremely thin)
            # Optimal range for thin hose: 3-30
            if 3 <= aspect_ratio <= 30:
                aspect_score = 1.0
            elif aspect_ratio < 3:
                aspect_score = aspect_ratio / 3.0
            else:
                aspect_score = max(0.3, 30.0 / aspect_ratio)
            confidence += 0.20 * aspect_score
            
            # Factor 3: Size score (prefer medium-sized objects)
            if 50 <= area <= 300:
                size_score = 1.0
            elif area < 50:
                size_score = area / 50.0
            else:
                size_score = max(0.4, 300.0 / area)
            confidence += 0.20 * size_score
            
            # Factor 4: Shape regularity (thin objects should have reasonable solidity)
            if 0.3 <= solidity <= 0.8:
                solidity_score = 1.0
            else:
                solidity_score = max(0.3, 1.0 - abs(solidity - 0.55) / 0.55)
            confidence += 0.15 * solidity_score
            
            # Factor 5: Compactness (thin objects have higher compactness)
            compactness_score = min(1.0, compactness / 15.0) if compactness < float('inf') else 0.5
            confidence += 0.10 * compactness_score
            
            # Factor 6: Position score (center objects score slightly higher)
            center_x, center_y = width / 2, height / 2
            distance_from_center = np.sqrt((cx - center_x)**2 + (cy - center_y)**2)
            max_distance = np.sqrt(center_x**2 + center_y**2)
            position_score = 1.0 - (distance_from_center / max_distance) * 0.5  # Less weight on position
            confidence += 0.10 * position_score
            
            # Normalize coordinates
            norm_cx = cx / width
            norm_cy = cy / height
            norm_x = x / width
            norm_y = y / height
            norm_w = w / width
            norm_h = h / height
            
            detection = {
                'bbox': [norm_x, norm_y, norm_w, norm_h],
                'center': [norm_cx, norm_cy],
                'confidence': min(0.99, max(0.01, confidence)),  # Clamp confidence
                'area': area,
                'aspect_ratio': aspect_ratio,
                'solidity': solidity,
                'extent': extent,
                'compactness': compactness,
                'angle': hose_angle,  # Hose angle in radians (-π/2 to π/2)
                'length_px': length,
                'width_px': width_px,
                'red_pixel_ratio': red_pixel_ratio,
                'pixel_bbox': [x, y, w, h]
            }
            
            detections.append(detection)
        
        # Sort by confidence and return best detections
        detections.sort(key=lambda x: x['confidence'], reverse=True)
        
        return detections

    def process_image_callback(self):
        """Process the latest image for hose detection"""
        if self.latest_image_msg is None:
            return

        try:
            # Convert ROS Image message to OpenCV image
            current_frame = self.bridge.compressed_imgmsg_to_cv2(self.latest_image_msg, "bgr8")
        except CvBridgeError as e:
            self.get_logger().error(f"Failed to convert image: {str(e)}")
            return

        # Segment red hose
        mask, colored_mask = self.segment_red_hose(current_frame)
        
        # Analyze hose contours
        detections = self.analyze_hose_contours(mask, current_frame.shape[:2])
        
        # Filter by confidence threshold
        valid_detections = [d for d in detections if d['confidence'] >= self.confidence_threshold]
        
        # Create annotated image
        annotated_frame = current_frame.copy()
        
        # Apply transparent mask overlay
        alpha = 0.3
        annotated_frame = cv2.addWeighted(annotated_frame, 1 - alpha, colored_mask, alpha, 0)
        
        # Create Detection2DArray message
        detection_array = Detection2DArray()
        detection_array.header = self.latest_image_msg.header
        
        best_hose_position = None
        
        for i, detection in enumerate(valid_detections[:3]):  # Limit to top 3 detections
            pixel_bbox = detection['pixel_bbox']
            norm_bbox = detection['bbox']
            confidence = detection['confidence']
            
            x, y, w, h = pixel_bbox
            
            # Draw bounding box
            cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            
            # Draw center point
            center_x = x + w // 2
            center_y = y + h // 2
            cv2.circle(annotated_frame, (center_x, center_y), 5, (0, 255, 0), -1)
            
            # Add detailed label with metrics
            label = f'Hose: {confidence:.2f}'
            details = f'AR:{detection["aspect_ratio"]:.1f} A:{detection["area"]:.0f}'
            
            label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            details_size, _ = cv2.getTextSize(details, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            
            # Background rectangle for text
            text_height = label_size[1] + details_size[1] + baseline + 10
            text_width = max(label_size[0], details_size[0])
            cv2.rectangle(annotated_frame, 
                         (x, y - text_height), 
                         (x + text_width, y), 
                         (0, 0, 255), cv2.FILLED)
            
            # Main label
            cv2.putText(annotated_frame, label, 
                       (x, y - details_size[1] - baseline - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Details
            cv2.putText(annotated_frame, details, 
                       (x, y - 2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # Create Detection2D message
            det_msg = Detection2D()
            det_msg.header = self.latest_image_msg.header
            
            det_msg.bbox = BoundingBox2D()
            det_msg.bbox.center.position.x = norm_bbox[0] + norm_bbox[2]/2
            det_msg.bbox.center.position.y = norm_bbox[1] + norm_bbox[3]/2
            det_msg.bbox.size_x = norm_bbox[2]
            det_msg.bbox.size_y = norm_bbox[3]

            # Create hypothesis
            hypothesis = ObjectHypothesisWithPose()
            hypothesis.hypothesis.class_id = "mangueira"
            hypothesis.hypothesis.score = confidence

            det_msg.results.append(hypothesis)
            detection_array.detections.append(det_msg)
            
            # Store best detection for position publishing
            if i == 0:  # Best detection
                best_hose_position = (detection['center'][0], detection['center'][1], confidence)

        # Publish detections
        self.detection_publisher.publish(detection_array)
        
        # Publish best hose position and angle
        if best_hose_position is not None:
            position_msg = PointStamped()
            position_msg.header = self.latest_image_msg.header
            position_msg.point.x = best_hose_position[0]  # Normalized x
            position_msg.point.y = best_hose_position[1]  # Normalized y
            position_msg.point.z = best_hose_position[2]  # Confidence
            self.hose_position_publisher.publish(position_msg)
            
            # Publish hose angle for the best detection
            if valid_detections:
                angle_msg = Float64()
                angle_msg.data = valid_detections[0]['angle']  # Angle in radians (-π/2 to π/2)
                self.hose_angle_publisher.publish(angle_msg)
            
            # Log every 10th detection to avoid spam
            if hasattr(self, '_detection_count'):
                self._detection_count += 1
            else:
                self._detection_count = 1
                
            if self._detection_count % 10 == 0:
                angle_deg = math.degrees(valid_detections[0]['angle']) if valid_detections else 0.0
                self.get_logger().info(f"Hose detected: pos=({best_hose_position[0]:.2f}, {best_hose_position[1]:.2f}), conf={best_hose_position[2]:.2f}, angle={angle_deg:.1f}°")
        else:
            cv2.putText(annotated_frame, 'No hose detected', (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Add detection count and parameters info
        cv2.putText(annotated_frame, f'Detections: {len(valid_detections)}', 
                   (10, current_frame.shape[0] - 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                   
        cv2.putText(annotated_frame, f'Conf threshold: {self.confidence_threshold:.2f}', 
                   (10, current_frame.shape[0] - 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                   
        cv2.putText(annotated_frame, f'Area range: {self.min_hose_area}-{self.max_hose_area}', 
                   (10, current_frame.shape[0] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Publish annotated image
        try:
            annotated_msg = self.bridge.cv2_to_imgmsg(annotated_frame, encoding="bgr8")
            annotated_msg.header = self.latest_image_msg.header
            self.image_publisher.publish(annotated_msg)
        except CvBridgeError as e:
            self.get_logger().error(f"Failed to convert annotated image: {str(e)}")


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