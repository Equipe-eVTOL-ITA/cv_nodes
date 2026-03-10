#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from std_msgs.msg import UInt8, Bool
from sensor_msgs.msg import CompressedImage, Image
from geometry_msgs.msg import Point

import cv2
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
import time

class WindowDetector(Node):
    def __init__(self):
        super().__init__('window_detector')

        self._centroid_publisher = self.create_publisher(Point, 'centroid', 10)
        self._threshold_publisher = self.create_publisher(UInt8, 'threshold', 10)
        self._window_found_publisher = self.create_publisher(Bool, 'window_found', 10)

        # Declare parameters for flexible configuration
        self.declare_parameter('depth_topic', '/depth_camera/image_raw')
        self.declare_parameter('use_compressed', False)
        self.declare_parameter('scale', 0.5)
        self.declare_parameter('debug_fps', 4.0)
        self.declare_parameter('depth_min_m', 0.2)
        self.declare_parameter('depth_max_m', 10.0)
        self.declare_parameter('depth_unit_scale', 0.001)  # For 16UC1 mm -> meters

        # Get parameters
        depth_topic = self.get_parameter('depth_topic').get_parameter_value().string_value
        use_compressed = self.get_parameter('use_compressed').get_parameter_value().bool_value
        self.scale = float(self.get_parameter('scale').get_parameter_value().double_value)
        debug_fps = float(self.get_parameter('debug_fps').get_parameter_value().double_value)
        self.depth_min = float(self.get_parameter('depth_min_m').get_parameter_value().double_value)
        self.depth_max = float(self.get_parameter('depth_max_m').get_parameter_value().double_value)
        self.depth_unit_scale = float(self.get_parameter('depth_unit_scale').get_parameter_value().double_value)
        self._debug_period = 1.0 / debug_fps if debug_fps > 0 else 0.0
        self._last_debug_pub_ts = 0.0

        # Set up subscription based on whether the topic is compressed or raw
        if use_compressed:
            self._subscriber = self.create_subscription(
                CompressedImage,
                depth_topic,
                self.compressed_depth_callback,
                10
            )
            self.get_logger().info(f'Subscribed to COMPRESSED depth topic: {depth_topic}')
        else:
            self._subscriber = self.create_subscription(
                Image,
                depth_topic,
                self.raw_depth_callback,
                10
            )
            self.get_logger().info(f'Subscribed to RAW depth topic: {depth_topic}')

        self.bridge = CvBridge()

        self.threshold = UInt8()
        self.threshold.data = int(8) # Initial threshold value for edge detection

        # Debug publishers for visualization in tools like RQt Image Viewer
        self._annotated_pub = self.create_publisher(Image, '/window_detector/annotated', 1)
        self._mask_pub = self.create_publisher(Image, '/window_detector/mask', 1)

    def raw_depth_callback(self, msg):
        """Callback for raw (uncompressed) depth images."""
        try:
            enc = (msg.encoding or '').lower()
            if enc in ('32fc1', '32fc'):
                depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1')
            elif enc in ('16uc1', 'mono16'):
                d16 = self.bridge.imgmsg_to_cv2(msg, desired_encoding='16UC1')
                depth = d16.astype(np.float32) * self.depth_unit_scale
                depth[d16 == 0] = np.nan
            else:
                self.get_logger().warn(f'Depth topic has non-depth encoding "{msg.encoding}". Trying best effort.')
                try:
                    depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1')
                except CvBridgeError:
                    d16 = self.bridge.imgmsg_to_cv2(msg, desired_encoding='16UC1')
                    depth = d16.astype(np.float32) * self.depth_unit_scale
                    depth[d16 == 0] = np.nan
        except CvBridgeError as e:
            self.get_logger().error(f'CvBridge Error (raw): {e}')
            return

        self.process_depth_image(depth)

    def compressed_depth_callback(self, msg):
        """Callback for compressed depth images (PNG format with 12-byte header)."""
        try:
            # Assumes the standard 12-byte header for compressed depth in ROS
            png_data = msg.data[12:]
            np_arr = np.frombuffer(png_data, np.uint8)
            depth_meters = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)

            if depth_meters is None:
                self.get_logger().error('Failed to decode compressed depth image.')
                return
            
            # Compressed depth can be 16-bit (mm). If so, convert to float meters.
            if depth_meters.dtype == np.uint16:
                depth_meters = depth_meters.astype(np.float32) * self.depth_unit_scale
                depth_meters[depth_meters == 0] = np.nan
            
            self.process_depth_image(depth_meters)
        except Exception as e:
            self.get_logger().error(f'Error processing compressed depth: {e}')
            return

    def process_depth_image(self, depth_meters):
        if depth_meters is None:
            return

        # Downscale for performance
        if self.scale != 1.0:
            depth_meters = cv2.resize(depth_meters, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_NEAREST)

        h, w = depth_meters.shape[:2]
        center_x = w // 2
        center_y = h // 2

        # Prepare depth data for processing
        d = depth_meters.astype(np.float32).copy()
        d[~np.isfinite(d)] = np.nan
        valid = np.isfinite(d) & (d >= self.depth_min) & (d <= self.depth_max)

        # Normalize depth image for visualization and edge detection
        # Far objects become bright, near objects become dark
        if np.any(valid):
            span = max(self.depth_max - self.depth_min, 1e-6)
            clamped = np.clip(d, self.depth_min, self.depth_max)
            n = (1.0 - (clamped - self.depth_min) / span) * 255.0
            n = np.clip(n, 0, 255).astype(np.uint8)
            n[~np.isfinite(n)] = 0
        else:
            n = np.zeros_like(d, dtype=np.uint8)

        # --- ADVANCED WINDOW DETECTION LOGIC ---

        # 1. Detect Edges using Sobel Filter
        grad_x = cv2.Sobel(n, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(n, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = cv2.magnitude(grad_x, grad_y)
        MAX_VALUE = 255
        if np.max(magnitude) > 0:
            magnitude = cv2.normalize(magnitude, None, 0, MAX_VALUE, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        else:
            magnitude = np.zeros_like(magnitude, dtype=np.uint8)
        
        _, binary_mask = cv2.threshold(magnitude, self.threshold.data, MAX_VALUE, cv2.THRESH_BINARY)
        self._threshold_publisher.publish(self.threshold)

        # 2. Clean the Mask with Morphological Closing
        # This crucial step connects broken edges of the window frame.
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        closed_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)

        # 3. Find Contours and Analyze Hierarchy
        # cv2.RETR_TREE builds a full hierarchy of parent-child contours.
        contornos, hierarquia = cv2.findContours(
            closed_mask, # Use the cleaned mask for better results
            cv2.RETR_TREE,
            cv2.CHAIN_APPROX_SIMPLE
        )

        output_contornos = cv2.cvtColor(n, cv2.COLOR_GRAY2BGR) # Debug on normalized depth
        window_candidates = []

        # The hierarchy is [Next, Previous, First_Child, Parent]
        if hierarquia is not None:
            for i, (contorno, hierarchy_info) in enumerate(zip(contornos, hierarquia[0])):
                # KEY LOGIC: A window is a "hole", so it must have a parent contour (the wall).
                parent_index = hierarchy_info[3]
                if parent_index != -1:
                    area = cv2.contourArea(contorno)
                    x, y, w_box, h_box = cv2.boundingRect(contorno)
                    
                    # Filter out small, noisy contours
                    if area > 1000 and w_box > 50 and h_box > 50:
                        # Optional but powerful: Filter by shape. Windows are typically rectangular.
                        peri = cv2.arcLength(contorno, True)
                        approx = cv2.approxPolyDP(contorno, 0.02 * peri, True)
                        
                        # Check if the shape is roughly a quadrilateral
                        if len(approx) >= 4 and len(approx) <= 6:
                             window_candidates.append(contorno)
                             # Draw this valid candidate in blue for debugging
                             cv2.drawContours(output_contornos, [contorno], -1, (255, 0, 0), 2)

        largest_centroid = None
        # 4. Select the best candidate (the largest one that meets all criteria)
        if window_candidates:
            largest_window = max(window_candidates, key=cv2.contourArea)
            
            # Draw the final chosen contour in bright green
            cv2.drawContours(output_contornos, [largest_window], -1, (0, 255, 0), 3)
            
            M = cv2.moments(largest_window)
            if M["m00"] != 0:
                cx_cont = int(M["m10"] / M["m00"])
                cy_cont = int(M["m01"] / M["m00"])
                largest_centroid = (cx_cont - center_x, cy_cont - center_y)
                # Draw the final centroid in red
                cv2.circle(output_contornos, (cx_cont, cy_cont), 7, (0, 0, 255), -1)

        # --- END OF DETECTION LOGIC ---

        # Publishing logic
        cx, cy = None, None
        if largest_centroid is not None:
            cx, cy = largest_centroid

        if cx is not None and cy is not None:
            cx_normalized = float(cx) / (w / 2.0)
            cy_normalized = float(cy) / (h / 2.0)
            centroid = Point()
            centroid.x = -1.0 * cy_normalized # Y-axis in image frame corresponds to -X in robot frame
            centroid.y = cx_normalized      # X-axis in image frame corresponds to Y in robot frame
            centroid.z = 0.0
            self._centroid_publisher.publish(centroid)
            self._window_found_publisher.publish(Bool(data=True))
            self.get_logger().info(f'Window Centroid Found (normalized): x={centroid.x:.3f}, y={centroid.y:.3f}')
        else:
            self._window_found_publisher.publish(Bool(data=False))
            # self.get_logger().info("No valid window found in this frame")

        # Publish debug images at a controlled rate
        now = time.time()
        if self._debug_period > 0 and (now - self._last_debug_pub_ts) >= self._debug_period:
            try:
                mask_msg = self.bridge.cv2_to_imgmsg(closed_mask, encoding='mono8')
                ann_msg = self.bridge.cv2_to_imgmsg(output_contornos, encoding='bgr8')
                self._mask_pub.publish(mask_msg)
                self._annotated_pub.publish(ann_msg)
            except CvBridgeError as e:
                self.get_logger().error(f'CvBridge Error (publish debug): {e}')
            self._last_debug_pub_ts = now

def main(args=None):
    rclpy.init(args=args)
    window_finder = WindowDetector()
    rclpy.spin(window_finder)
    window_finder.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()