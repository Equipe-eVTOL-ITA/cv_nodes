import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
import numpy as np
import open3d as o3d
import cv2 as cv


class RansacWindowDetectorNode(Node):
    def __init__(self):
        super().__init__('ransac_window_detector')

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.sub = self.create_subscription(
            PointCloud2,
            '/camera/depth/points',
            self.cloud_callback,
            qos
        )

        self.last_process_time = self.get_clock().now()
        self.process_interval = 2.0
        self.get_logger().info('RANSAC window detector simulation started. Waiting for point cloud...')

    def cloud_callback(self, msg):
        now = self.get_clock().now()
        elapsed = (now - self.last_process_time).nanoseconds / 1e9
        if elapsed < self.process_interval:
            return
        self.last_process_time = now

        self.get_logger().info('Point cloud received - running RANSAC window detection...')

        raw = pc2.read_points(msg, field_names=('x', 'y', 'z'), skip_nans=True)
        pts = np.array([[p[0], p[1], p[2]] for p in raw], dtype=np.float32)

        mask = np.isfinite(pts).all(axis=1)
        pts = pts[mask]

        if len(pts) < 100:
            self.get_logger().warn(f'Too few valid points ({len(pts)}), skipping.')
            return

        self.get_logger().info(f'Valid points: {len(pts)}')
        self.get_logger().info(f'Min XYZ: {pts.min(axis=0)}')
        self.get_logger().info(f'Max XYZ: {pts.max(axis=0)}')

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)

        bbox = pcd.get_axis_aligned_bounding_box()
        extent = np.asarray(bbox.get_extent())
        voxel_size = max(float(extent.max()) / 100.0, 0.01)
        self.get_logger().info(f'Scene extent: {extent}, using voxel_size={voxel_size:.4f}')

        pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
        pts_ds = np.asarray(pcd.points)
        self.get_logger().info(f'After downsampling: {len(pts_ds)} points')

        plane_model, inliers = pcd.segment_plane(
            distance_threshold=0.02,
            ransac_n=3,
            num_iterations=1000
        )

        [a, b, c, d] = plane_model
        self.get_logger().info(
            f'Wall plane: {a:.3f}x + {b:.3f}y + {c:.3f}z + {d:.3f} = 0 '
            f'({len(inliers)} inliers)'
        )

        wall_cloud = pcd.select_by_index(inliers)
        wall_points = np.asarray(wall_cloud.points)

        normal = np.array([a, b, c])
        normal = normal / np.linalg.norm(normal)

        world_up = np.array([0.0, 0.0, 1.0])
        right = np.cross(normal, world_up)
        if np.linalg.norm(right) < 1e-6:
            world_up = np.array([0.0, 1.0, 0.0])
            right = np.cross(normal, world_up)
        right = right / np.linalg.norm(right)
        up = np.cross(right, normal)
        up = up / np.linalg.norm(up)

        u_coords = wall_points @ right
        v_coords = wall_points @ up

        resolution = voxel_size
        u_min, u_max = u_coords.min(), u_coords.max()
        v_min, v_max = v_coords.min(), v_coords.max()

        width = int((u_max - u_min) / resolution) + 2
        height = int((v_max - v_min) / resolution) + 2

        if width < 5 or height < 5:
            self.get_logger().warn('Wall projection too small, skipping.')
            return

        wall_image = np.zeros((height, width), dtype=np.uint8)
        u_idx = np.clip(((u_coords - u_min) / resolution).astype(int), 0, width - 1)
        v_idx = np.clip(((v_coords - v_min) / resolution).astype(int), 0, height - 1)
        wall_image[v_idx, u_idx] = 255

        kernel = np.ones((3, 3), np.uint8)
        wall_image = cv.dilate(wall_image, kernel, iterations=2)
        wall_holes = cv.bitwise_not(wall_image)

        border = 5
        wall_holes[:border, :] = 0
        wall_holes[-border:, :] = 0
        wall_holes[:, :border] = 0
        wall_holes[:, -border:] = 0

        contours, _ = cv.findContours(
            wall_holes, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
        )

        self.get_logger().info(f'Found {len(contours)} hole regions')

        window_found = False
        for i, contour in enumerate(contours):
            area_px = cv.contourArea(contour)
            if area_px < 20:
                continue

            x, y, w, h = cv.boundingRect(contour)
            width_m = w * resolution
            height_m = h * resolution
            aspect = width_m / height_m if height_m > 0 else 0

            self.get_logger().info(
                f'  Hole {i}: {width_m:.2f}m x {height_m:.2f}m  aspect={aspect:.2f}'
            )

            if (0.3 < width_m < 3.0 and
                0.3 < height_m < 3.0 and
                0.2 < aspect < 5.0):
                cx_2d = u_min + (x + w / 2) * resolution
                cy_2d = v_min + (y + h / 2) * resolution
                dist_to_plane = -d / (np.dot(normal, normal))
                center_3d = cx_2d * right + cy_2d * up + dist_to_plane * normal

                self.get_logger().info(
                    f'  *** WINDOW DETECTED ***\n'
                    f'      Size: {width_m:.2f}m x {height_m:.2f}m\n'
                    f'      Center 3D: ({center_3d[0]:.2f}, '
                    f'{center_3d[1]:.2f}, {center_3d[2]:.2f})\n'
                    f'      Approach vector: ({-normal[0]:.2f}, '
                    f'{-normal[1]:.2f}, {-normal[2]:.2f})'
                )
                window_found = True

        if not window_found:
            self.get_logger().info(
                'No window detected. Move the camera closer to the wall.'
            )

        self.get_logger().info('--- cycle complete ---\n')


def main(args=None):
    rclpy.init(args=args)
    node = RansacWindowDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()