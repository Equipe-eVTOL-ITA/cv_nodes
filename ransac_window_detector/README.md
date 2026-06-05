# RANSAC Window Detector

Real-time window opening detection for autonomous drone systems using 3D point cloud processing and geometric plane segmentation.

## What It Does

Detects rectangular window openings in walls using an RGB-D camera. The node subscribes to a depth point cloud, identifies the dominant wall plane via RANSAC, projects it to 2D, and uses OpenCV contour detection to find holes — publishing the 3D position and approach vector of detected windows.

## Pipeline

```
┌─────────────────┐
│  RGB-D Camera   │ (PointCloud2 from depth sensor)
└────────┬────────┘
         │
         v
┌─────────────────────────┐
│  NaN / Inf Filtering    │ (Remove invalid depth points)
└────────┬────────────────┘
         │
         v
┌─────────────────────────┐
│  Voxel Downsampling     │ (Reduce density for performance)
└────────┬────────────────┘
         │
         v
┌─────────────────────────┐
│  RANSAC Plane Detection │ (Find dominant wall plane via Open3D)
└────────┬────────────────┘
         │
         v
┌─────────────────────────┐
│  Normal Vector Filter   │ (Reject floor/ceiling — keep walls only)
└────────┬────────────────┘
         │
         v
┌──────────────────────────────────────┐
│  3D → 2D Projection                  │ (Project wall points onto local plane)
└────────┬─────────────────────────────┘
         │
         v
┌──────────────────────────────────────┐
│  OpenCV Contour Detection            │ (Find rectangular holes in wall image)
└────────┬─────────────────────────────┘
         │
         v
┌──────────────────────────────────────┐
│  Size / Aspect Ratio Filter          │ (0.3m–3.0m, aspect 0.2–5.0)
└────────┬─────────────────────────────┘
         │
         v
┌──────────────────────────────────────┐
│  3D Unprojection                     │ (Recover window center in world frame)
└────────┬─────────────────────────────┘
         │
         v
┌───────────────────────────────────────────────┐
│  Publish: PoseStamped + terminal log           │
└───────────────────────────────────────────────┘
```

## Published ROS2 Topics

| Topic          | Type                        | Description                                                  |
| -------------- | --------------------------- | ------------------------------------------------------------ |
| `/window/pose` | `geometry_msgs/PoseStamped` | 3D center position + approach orientation of detected window |

## Subscribed ROS2 Topics

| Topic                  | Type                      | Description             |
| ---------------------- | ------------------------- | ----------------------- |
| `/camera/depth/points` | `sensor_msgs/PointCloud2` | Input depth point cloud |

> For Intel RealSense hardware, change to `/camera/depth/color/points`.

## Parameters

| Parameter            | Type  | Default | Description                                                            |
| -------------------- | ----- | ------- | ---------------------------------------------------------------------- |
| `process_interval`   | float | `2.0`   | Seconds between RANSAC processing cycles                               |
| `distance_threshold` | float | `0.02`  | RANSAC inlier threshold in meters (increase to 0.05 for real hardware) |
| `ransac_iterations`  | int   | `1000`  | Number of RANSAC iterations                                            |
| `min_points`         | int   | `100`   | Minimum valid points required to attempt detection                     |
| `min_window_size`    | float | `0.3`   | Minimum window dimension in meters                                     |
| `max_window_size`    | float | `3.0`   | Maximum window dimension in meters                                     |
| `min_aspect_ratio`   | float | `0.2`   | Minimum width/height ratio                                             |
| `max_aspect_ratio`   | float | `5.0`   | Maximum width/height ratio                                             |
| `normal_z_threshold` | float | `0.3`   | Max Z component of plane normal — rejects floor/ceiling                |

## Dependencies

### ROS2 packages

```bash
sudo apt install ros-humble-sensor-msgs-py
```

### Python

```bash
pip3 install open3d --break-system-packages
pip3 install "numpy<2" --break-system-packages   # Required — NumPy 2.x breaks cv_bridge
pip3 install opencv-python --break-system-packages
```

> **NumPy version:** `cv_bridge` in ROS2 Humble was compiled against NumPy 1.x. Installing NumPy 2.x will cause `_ARRAY_API not found` on import. Always use `numpy<2`.

## Building

```bash
cd ~/your_ws
colcon build --symlink-install --packages-select ransac_window_detector
source install/setup.bash
```

## Running

```bash
# Terminal 1 — launch simulation or hardware sensor
# (Gazebo or real RealSense)

# Terminal 2 — static TF (simulation only, replace with odometry for real drone)
ros2 run tf2_ros static_transform_publisher 0 0 0 0 0 0 world camera_link

# Terminal 3 — run the detector
ros2 run ransac_window_detector ransac_node
```

Expected output when a window is detected:

```
[INFO] Valid points: 18920
[INFO] Wall plane: 0.011x + 0.999y + 0.002z + -3.614 = 0 (2198 inliers)
[INFO] Found 3 hole regions
[INFO]   Hole 1: 1.28m x 0.98m  aspect=1.31
[INFO]   *** WINDOW DETECTED ***
[INFO]       Size: 1.28m x 0.98m
[INFO]       Center 3D: (-0.03, 3.61, 1.02)
[INFO]       Approach vector: (-0.01, -1.00, -0.00)
```

## QoS Configuration

The node subscribes with `BEST_EFFORT` reliability to match the depth camera publisher. This is required — using `RELIABLE` (ROS2 default) will result in no messages received even when the topic is active.

## Simulation Setup

Tested with **Gazebo Classic 11** + **ROS2 Humble** using `libgazebo_ros_camera.so` with `type="depth"`.

> `libgazebo_ros_openni_kinect.so` does **not** exist in ROS2 Humble. Use `libgazebo_ros_camera.so` with a depth sensor type instead.

Minimum world file camera sensor config:

```xml
<sensor name="depth_camera" type="depth">
  <update_rate>10</update_rate>
  <always_on>true</always_on>
  <plugin name="depth_camera_plugin" filename="libgazebo_ros_camera.so">
    <ros>
      <namespace>/camera</namespace>
      <remapping>~/image_raw:=depth/image_raw</remapping>
      <remapping>~/points:=depth/points</remapping>
    </ros>
    <camera_name>depth</camera_name>
    <frame_name>camera_link</frame_name>
    <min_depth>0.05</min_depth>
    <max_depth>10.0</max_depth>
  </plugin>
</sensor>
```

## Hardware (Real Drone)

Tested sensor: **Intel RealSense D435 / D455**

```bash
sudo apt install ros-humble-realsense2-camera

ros2 launch realsense2_camera rs_launch.py \
  pointcloud.enable:=true \
  align_depth.enable:=true
```

Change the subscribed topic in the node from `/camera/depth/points` to `/camera/depth/color/points`.

Recommended parameter adjustments for real hardware:

```python
distance_threshold = 0.05   # Walls have more surface irregularity
min_area_px = 50            # More noise from real sensors
```

## How RANSAC Works Here

RANSAC (_Random Sample Consensus_) finds the dominant plane in the point cloud by:

1. Randomly sampling 3 points and fitting a plane through them
2. Counting how many other points lie within `distance_threshold` of that plane (inliers)
3. Repeating 1000 times and keeping the plane with the most inliers

The plane equation `ax + by + cz + d = 0` is then used to:

- Isolate wall points from the rest of the scene
- Compute the wall normal vector (approach direction for the drone)
- Project wall points to 2D for OpenCV contour detection

The normal vector filter `abs(normal[2]) > 0.3` discards horizontal planes (floor/ceiling) so RANSAC always returns a vertical wall even when the floor has more points in view.

## Troubleshooting

| Problem                              | Cause                                 | Fix                                                                             |
| ------------------------------------ | ------------------------------------- | ------------------------------------------------------------------------------- |
| No topic `/camera/depth/points`      | Gazebo launched without `ros2 launch` | Use `ros2 launch gazebo_ros gazebo.launch.py`                                   |
| Point cloud invisible in RViz        | Reliability Policy mismatch           | Set **Reliability Policy → Best Effort** in RViz display                        |
| `Frame [camera_link] does not exist` | No TF publisher                       | Run `ros2 run tf2_ros static_transform_publisher 0 0 0 0 0 0 world camera_link` |
| `_ARRAY_API not found` on import     | NumPy 2.x installed                   | `pip install "numpy<2" --break-system-packages`                                 |
| RANSAC detects floor instead of wall | Camera too far from wall              | Move camera closer; normal filter handles this automatically                    |
| Window not detected                  | Size filters too strict               | Lower `min_window_size` or widen aspect ratio range                             |
| Node crashes on large point clouds   | Memory pressure                       | Reduce camera resolution or increase `process_interval`                         |

## Integration with Mission Node

The node publishes `/window/pose` as a `PoseStamped`. A mission state machine can subscribe to this and use the position as a navigation target:

```python
from geometry_msgs.msg import PoseStamped

self.window_sub = self.create_subscription(
    PoseStamped,
    '/window/pose',
    self.window_callback,
    10
)

def window_callback(self, msg):
    # msg.pose.position  -> 3D center of window
    # msg.pose.orientation -> approach direction (face the wall)
    self.window_position = msg.pose.position
    self.transition(MissionState.APPROACH)
```

## References

- [Open3D RANSAC documentation](http://www.open3d.org/docs/release/tutorial/geometry/pointcloud.html#Plane-segmentation)
- [PX4 ROS2 User Guide](https://docs.px4.io/main/en/ros2/user_guide.html)
- [ROS2 Humble sensor_msgs](https://docs.ros2.org/humble/api/sensor_msgs/)
- [Intel RealSense ROS2 wrapper](https://github.com/IntelRealSense/realsense-ros)

---

**Maintainer:** eVTOL ITA Navigation Team  
**ROS2 Distro:** Humble  
**Status:** Simulation validated ✅ | Hardware pending 🔧
